"""
Unsupervised topic discovery processor using BERTopic.
Part of Milestone 5 implementation for hybrid topic classification.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP

logger = logging.getLogger(__name__)


class DiscoveryTopicProcessor:
    """Discover topics using unsupervised clustering with BERTopic."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 10,
        min_samples: int = 5,
        n_neighbors: int = 15,
        n_components: int = 5,
        max_features: int = 1000,
        device: str = "cpu",
        language: str = "english",
        top_k_words: int = 10,
        nr_topics: Optional[int] = None,
        min_topic_size: int = 10,
    ) -> None:
        """Initialize discovery topic processor.
        
        Args:
            embedding_model: SentenceTransformer model name for text embeddings
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples in a cluster for HDBSCAN  
            n_neighbors: Number of neighbors for UMAP
            n_components: Number of components for UMAP dimensionality reduction
            max_features: Maximum features for CountVectorizer
            device: Device for embeddings ("cpu", "cuda", "mps")
            language: Language for stop words
            top_k_words: Number of words per topic
            nr_topics: Target number of topics (None for automatic)
            min_topic_size: Minimum topic size after merging
        """
        self.embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.max_features = max_features
        self.device = device
        self.language = language
        self.top_k_words = top_k_words
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        
        # Initialize components
        self.embedding_model = None
        self.bertopic_model = None
        self.cluster_labels = None
        self.topic_info = None
        self.embeddings = None
        
    def _initialize_components(self) -> None:
        """Initialize BERTopic components."""
        logger.info("Initializing BERTopic components...")
        
        # Sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        
        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric='cosine',
            low_memory=False,
            random_state=42
        )
        
        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # CountVectorizer for topic representation
        vectorizer_model = CountVectorizer(
            max_features=self.max_features,
            stop_words=self.language,
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Initialize BERTopic
        self.bertopic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language=self.language,
            calculate_probabilities=True,
            verbose=False
        )
        
        logger.info("BERTopic components initialized successfully")
        
    def discover_topics(
        self, 
        texts: List[str], 
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[float]]:
        """Discover topics from a collection of texts.
        
        Args:
            texts: List of texts to analyze
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Tuple of (topic_labels, topic_probabilities)
        """
        if self.bertopic_model is None:
            self._initialize_components()
        
        logger.info(f"Discovering topics from {len(texts)} documents...")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and len(text.strip()) > 0]
        if len(valid_texts) < self.min_cluster_size:
            logger.warning(f"Not enough valid texts ({len(valid_texts)}) for clustering")
            return [-1] * len(texts), [0.0] * len(texts)
        
        try:
            # Fit BERTopic model
            if embeddings is not None:
                topics, probabilities = self.bertopic_model.fit_transform(valid_texts, embeddings)
            else:
                topics, probabilities = self.bertopic_model.fit_transform(valid_texts)
            
            # Store results
            self.cluster_labels = topics
            self.topic_info = self.bertopic_model.get_topic_info()
            
            # Reduce topics if specified
            if self.nr_topics is not None and len(self.topic_info) > self.nr_topics + 1:
                logger.info(f"Reducing topics from {len(self.topic_info)} to {self.nr_topics}")
                self.bertopic_model.reduce_topics(valid_texts, nr_topics=self.nr_topics)
                topics = self.bertopic_model.topics_
                self.topic_info = self.bertopic_model.get_topic_info()
            
            # Handle texts that were filtered out
            final_topics = []
            final_probs = []
            valid_idx = 0
            
            for original_text in texts:
                if original_text and len(original_text.strip()) > 0:
                    final_topics.append(topics[valid_idx] if valid_idx < len(topics) else -1)
                    final_probs.append(probabilities[valid_idx] if valid_idx < len(probabilities) else 0.0)
                    valid_idx += 1
                else:
                    final_topics.append(-1)
                    final_probs.append(0.0)
            
            logger.info(f"Topic discovery completed. Found {len(self.topic_info)} topics.")
            logger.info(f"Topic distribution: {dict(pd.Series(final_topics).value_counts().head())}")
            
            return final_topics, final_probs
            
        except Exception as e:
            logger.error(f"Topic discovery failed: {e}")
            return [-1] * len(texts), [0.0] * len(texts)
    
    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels from discovered topics.
        
        Returns:
            Dictionary mapping topic IDs to descriptive labels
        """
        if self.bertopic_model is None or self.topic_info is None:
            return {}
        
        topic_labels = {}
        
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            
            if topic_id == -1:
                topic_labels[topic_id] = "outliers"
            else:
                # Get topic words and create label
                topic_words = self.bertopic_model.get_topic(topic_id)
                if topic_words:
                    # Take top 3 words for label
                    top_words = [word for word, _ in topic_words[:3]]
                    topic_labels[topic_id] = "_".join(top_words)
                else:
                    topic_labels[topic_id] = f"topic_{topic_id}"
        
        return topic_labels
    
    def get_topic_info_detailed(self) -> Dict[str, Any]:
        """Get detailed information about discovered topics.
        
        Returns:
            Dictionary with topic information and statistics
        """
        if self.bertopic_model is None or self.topic_info is None:
            return {}
        
        topic_labels = self.get_topic_labels()
        
        # Get topic details
        topics_detail = []
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            
            topic_detail = {
                'topic_id': int(topic_id),
                'label': topic_labels.get(topic_id, f"topic_{topic_id}"),
                'size': int(row['Count']),
                'representative_docs': []
            }
            
            # Get topic words
            if topic_id != -1:
                topic_words = self.bertopic_model.get_topic(topic_id)
                topic_detail['words'] = [
                    {'word': word, 'weight': float(weight)}
                    for word, weight in topic_words[:self.top_k_words]
                ]
                
                # Get representative documents
                try:
                    repr_docs = self.bertopic_model.get_representative_docs(topic_id)
                    topic_detail['representative_docs'] = [
                        doc[:200] + "..." if len(doc) > 200 else doc
                        for doc in (repr_docs[:3] if repr_docs else [])
                    ]
                except:
                    topic_detail['representative_docs'] = []
            else:
                topic_detail['words'] = []
                topic_detail['representative_docs'] = []
            
            topics_detail.append(topic_detail)
        
        return {
            'total_topics': len(self.topic_info),
            'total_documents': int(self.topic_info['Count'].sum()),
            'outliers_count': int(self.topic_info[self.topic_info['Topic'] == -1]['Count'].sum()) if -1 in self.topic_info['Topic'].values else 0,
            'topics': topics_detail,
            'model_params': {
                'embedding_model': self.embedding_model_name,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'n_neighbors': self.n_neighbors,
                'n_components': self.n_components
            }
        }
    
    def map_to_ontology_topics(self, ontology_topics: List[str], threshold: float = 0.3) -> Dict[int, str]:
        """Map discovered topics to predefined ontology topics.
        
        Args:
            ontology_topics: List of predefined topic categories
            threshold: Minimum similarity threshold for mapping
            
        Returns:
            Dictionary mapping discovery topic IDs to ontology topics
        """
        if self.bertopic_model is None or self.topic_info is None:
            return {}
        
        from transformers import pipeline
        
        # Use zero-shot classification to map topics
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == "cuda" else -1
        )
        
        topic_mapping = {}
        topic_labels = self.get_topic_labels()
        
        for topic_id in topic_labels.keys():
            if topic_id == -1:
                topic_mapping[topic_id] = "other"
                continue
            
            # Get representative text for the topic
            topic_words = self.bertopic_model.get_topic(topic_id)
            if not topic_words:
                topic_mapping[topic_id] = "other"
                continue
            
            # Create description from top words
            top_words = [word for word, _ in topic_words[:5]]
            topic_description = f"A topic about {', '.join(top_words)}"
            
            try:
                # Classify against ontology topics
                result = classifier(topic_description, ontology_topics)
                
                best_match = result['labels'][0]
                best_score = result['scores'][0]
                
                if best_score >= threshold:
                    topic_mapping[topic_id] = best_match
                else:
                    topic_mapping[topic_id] = "other"
                    
            except Exception as e:
                logger.warning(f"Failed to map topic {topic_id}: {e}")
                topic_mapping[topic_id] = "other"
        
        logger.info(f"Mapped {len(topic_mapping)} discovery topics to ontology")
        return topic_mapping
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add discovery topic information.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added discovery topic columns
        """
        logger.info(f"Processing {len(df)} messages for topic discovery")
        
        # Extract texts
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # Discover topics
        topic_ids, topic_probs = self.discover_topics(texts)
        
        # Add results to dataframe
        df = df.copy()
        df['discovery_topic_id'] = topic_ids
        df['discovery_topic_probability'] = topic_probs
        
        # Add topic labels
        topic_labels = self.get_topic_labels()
        df['discovery_topic_label'] = df['discovery_topic_id'].map(topic_labels).fillna("unknown")
        
        # Log summary
        topic_distribution = df['discovery_topic_label'].value_counts()
        valid_topics = sum(1 for tid in topic_ids if tid != -1)
        
        logger.info(f"Topic discovery completed.")
        logger.info(f"Valid topic assignments: {valid_topics}/{len(df)} ({valid_topics/len(df)*100:.1f}%)")
        logger.info(f"Top discovered topics: {dict(topic_distribution.head())}")
        
        return df
    
    def export_discovery_analysis(self, df: pd.DataFrame, output_path: str) -> None:
        """Export topic discovery analysis to JSON file.
        
        Args:
            df: DataFrame with discovery results
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        
        analysis_data = {
            'metadata': {
                'total_messages': len(df),
                'extraction_date': pd.Timestamp.now().isoformat(),
                'processor_version': '1.0.0',
                'embedding_model': self.embedding_model_name,
                'clustering_params': {
                    'min_cluster_size': self.min_cluster_size,
                    'min_samples': self.min_samples,
                    'n_neighbors': self.n_neighbors,
                    'n_components': self.n_components
                }
            },
            'topic_info': self.get_topic_info_detailed(),
            'discovery_stats': self.get_discovery_stats(df)
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Discovery analysis exported to {output_path}")
    
    def get_discovery_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about topic discovery results.
        
        Args:
            df: DataFrame with discovery results
            
        Returns:
            Dictionary with discovery statistics
        """
        if 'discovery_topic_id' not in df.columns:
            return {}
        
        stats = {
            'total_messages': len(df),
            'messages_with_topics': sum(1 for tid in df['discovery_topic_id'] if tid != -1),
            'outlier_messages': sum(1 for tid in df['discovery_topic_id'] if tid == -1),
            'unique_topics': len(df['discovery_topic_id'].unique()) - (1 if -1 in df['discovery_topic_id'].values else 0),
            'avg_topic_probability': float(df['discovery_topic_probability'].mean()),
            'topic_distribution': dict(df['discovery_topic_label'].value_counts().head(10))
        }
        
        stats['topic_coverage'] = (stats['messages_with_topics'] / stats['total_messages']) * 100
        
        return stats