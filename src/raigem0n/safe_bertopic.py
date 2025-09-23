"""
Memory-safe wrapper for BERTopic operations with error isolation and cleanup.

This module provides a robust wrapper around BERTopic to prevent Bus Errors
and memory crashes on Apple Silicon and other platforms.
"""

import gc
import logging
import multiprocessing
import os
import signal
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SafeBERTopicWrapper:
    """Safe wrapper for BERTopic operations with memory management and error isolation."""
    
    def __init__(self, 
                 timeout_seconds: int = 300,
                 max_memory_mb: int = 4096,
                 enable_multiprocessing: bool = False):
        """Initialize safe BERTopic wrapper.
        
        Args:
            timeout_seconds: Maximum time to allow for BERTopic operations
            max_memory_mb: Maximum memory usage before forcing cleanup
            enable_multiprocessing: Whether to allow multiprocessing (risky on macOS)
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.enable_multiprocessing = enable_multiprocessing
        
        # Disable multiprocessing on macOS by default due to Bus Error issues
        if not enable_multiprocessing and sys.platform == "darwin":
            self._disable_multiprocessing()
    
    def _disable_multiprocessing(self) -> None:
        """Disable problematic multiprocessing settings."""
        # Force single-threaded execution
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        
        # Disable nested parallelism
        os.environ["OMP_NESTED"] = "FALSE"
        
        # Use 'fork' start method if available (more stable than 'spawn')
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # Already set
        
        logger.info("Multiprocessing disabled for stability on macOS")
    
    @contextmanager
    def _memory_monitor(self, operation_name: str):
        """Context manager to monitor memory usage during operations."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Starting {operation_name} with {initial_memory:.1f}MB memory")
        
        try:
            yield
        finally:
            # Force cleanup
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Completed {operation_name}: {initial_memory:.1f}MB → {final_memory:.1f}MB")
    
    @contextmanager
    def _timeout_handler(self, timeout_seconds: int):
        """Context manager to handle operation timeouts."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        if sys.platform != "win32":  # Signal handling not available on Windows
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            yield
    
    def safe_topic_discovery(self,
                           texts: List[str],
                           discovery_params: Dict[str, Any] = None) -> Tuple[List[int], List[float], Optional[Dict[str, Any]]]:
        """Perform topic discovery with memory safety and error isolation.
        
        Args:
            texts: List of texts to analyze
            discovery_params: Parameters for BERTopic discovery
            
        Returns:
            Tuple of (topic_ids, probabilities, topic_info) or safe fallbacks on error
        """
        discovery_params = discovery_params or {}
        
        try:
            with self._memory_monitor("topic_discovery"):
                with self._timeout_handler(self.timeout_seconds):
                    return self._execute_topic_discovery(texts, discovery_params)
                    
        except TimeoutError as e:
            logger.error(f"BERTopic discovery timed out: {e}")
            return self._fallback_topic_assignment(texts)
            
        except MemoryError as e:
            logger.error(f"BERTopic discovery ran out of memory: {e}")
            return self._fallback_topic_assignment(texts)
            
        except Exception as e:
            logger.error(f"BERTopic discovery failed: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return self._fallback_topic_assignment(texts)
    
    def _execute_topic_discovery(self,
                               texts: List[str],
                               discovery_params: Dict[str, Any]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Execute the actual topic discovery with safety measures."""
        from .discovery_topic_processor import DiscoveryTopicProcessor
        
        # Configure safe parameters for Apple Silicon
        safe_params = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'min_cluster_size': max(10, len(texts) // 50),  # Adaptive cluster size
            'min_samples': 5,
            'n_neighbors': min(15, len(texts) // 10),  # Adaptive neighbors
            'n_components': 5,
            'max_features': 500,  # Reduced features for memory safety
            'device': 'cpu',  # Force CPU for stability
            'language': 'english',
            'top_k_words': 10,
            'nr_topics': min(20, len(texts) // 30) if len(texts) > 100 else None,  # Limit topics
            'min_topic_size': max(5, len(texts) // 100)
        }
        
        # Update with user parameters but keep safe defaults
        safe_params.update(discovery_params)
        
        logger.info(f"Starting topic discovery with safe parameters: {safe_params}")
        
        # Initialize processor with safe parameters
        processor = DiscoveryTopicProcessor(**safe_params)
        
        # Filter texts to reduce memory load
        filtered_texts = [text[:1000] for text in texts if text and len(text.strip()) > 10]  # Truncate long texts
        
        if len(filtered_texts) < safe_params['min_cluster_size']:
            logger.warning(f"Not enough texts for clustering ({len(filtered_texts)}), using fallback")
            return self._fallback_topic_assignment(texts)
        
        # Execute discovery
        topic_ids, probabilities = processor.discover_topics(filtered_texts)
        
        # Pad results to match original text count
        final_topic_ids = []
        final_probabilities = []
        filtered_idx = 0
        
        for original_text in texts:
            if original_text and len(original_text.strip()) > 10:
                if filtered_idx < len(topic_ids):
                    final_topic_ids.append(topic_ids[filtered_idx])
                    final_probabilities.append(probabilities[filtered_idx])
                else:
                    final_topic_ids.append(-1)
                    final_probabilities.append(0.0)
                filtered_idx += 1
            else:
                final_topic_ids.append(-1)
                final_probabilities.append(0.0)
        
        # Get topic information
        topic_info = processor.get_topic_info_detailed()
        
        # Clean up processor to free memory
        del processor
        gc.collect()
        
        logger.info(f"Topic discovery completed successfully")
        return final_topic_ids, final_probabilities, topic_info
    
    def _fallback_topic_assignment(self, texts: List[str]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Provide fallback topic assignment when BERTopic fails."""
        logger.info("Using fallback topic assignment")
        
        # Simple fallback: assign all texts to "outliers" topic
        topic_ids = [-1] * len(texts)
        probabilities = [0.0] * len(texts)
        
        topic_info = {
            'total_topics': 1,
            'total_documents': len(texts),
            'outliers_count': len(texts),
            'topics': [{
                'topic_id': -1,
                'label': 'outliers',
                'size': len(texts),
                'words': [],
                'representative_docs': []
            }],
            'model_params': {
                'fallback': True,
                'reason': 'BERTopic failed or timed out'
            }
        }
        
        return topic_ids, probabilities, topic_info
    
    def safe_dataframe_processing(self, 
                                df: pd.DataFrame, 
                                text_column: str = "text",
                                discovery_params: Dict[str, Any] = None) -> pd.DataFrame:
        """Process dataframe with safe topic discovery.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            discovery_params: Parameters for BERTopic discovery
            
        Returns:
            DataFrame with added discovery topic columns
        """
        logger.info(f"Processing {len(df)} messages for safe topic discovery")
        
        # Extract texts
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # Perform safe discovery
        topic_ids, topic_probs, topic_info = self.safe_topic_discovery(texts, discovery_params)
        
        # Create result dataframe copy to avoid memory issues
        result_df = df.copy()
        
        # Add results
        result_df['discovery_topic_id'] = topic_ids
        result_df['discovery_topic_probability'] = topic_probs
        
        # Add topic labels
        if topic_info and 'topics' in topic_info:
            topic_labels = {}
            for topic in topic_info['topics']:
                topic_labels[topic['topic_id']] = topic['label']
            
            result_df['discovery_topic_label'] = result_df['discovery_topic_id'].map(topic_labels).fillna("unknown")
        else:
            result_df['discovery_topic_label'] = "unknown"
        
        # Log summary
        topic_distribution = result_df['discovery_topic_label'].value_counts()
        valid_topics = sum(1 for tid in topic_ids if tid != -1)
        
        logger.info(f"Safe topic discovery completed.")
        logger.info(f"Valid topic assignments: {valid_topics}/{len(df)} ({valid_topics/len(df)*100:.1f}%)")
        logger.info(f"Top discovered topics: {dict(topic_distribution.head())}")
        
        return result_df


def create_safe_discovery_processor(**kwargs) -> SafeBERTopicWrapper:
    """Factory function to create a safe BERTopic wrapper."""
    return SafeBERTopicWrapper(**kwargs)