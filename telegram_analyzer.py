#!/usr/bin/env python3
"""
Raigem0n: Telegram Stance & Language Analysis Pipeline

A neutral, reproducible analytics pipeline for analyzing public Telegram channels
to extract stance classification, topic analysis, and linguistic patterns while
maintaining attribution accuracy.

Usage:
    python telegram_analyzer.py --config config/config.yaml --input data/channel.json --out out/run-$(date +%Y%m%d-%H%M)
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from raigem0n.config import Config
from raigem0n.data_loader import DataLoader
from raigem0n.processors import NERProcessor, SentimentProcessor, StyleProcessor, TopicProcessor, ToxicityProcessor, StanceProcessor, LinksProcessor, QuoteProcessor
from raigem0n.checkpoint_manager import CheckpointManager
from raigem0n.safe_bertopic import SafeBERTopicWrapper


def setup_logging(level: str = "INFO", redact_sensitive: bool = True) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Add file handler if output directory exists
    if Path("out").exists():
        log_file = Path("out") / f"raigem0n_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {level} level")


def detect_device() -> str:
    """Detect available compute device based on environment and hardware."""
    import torch
    
    # Check environment variables (Exo configuration)
    if os.getenv("CLANG") == "1":
        logger.info("Using CPU (CLANG environment variable set)")
        return "cpu"
    
    if os.getenv("CUDA") and torch.cuda.is_available():
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        return "cuda"
    
    # Check for Apple MPS
    if torch.backends.mps.is_available() and not os.getenv("CUDA"):
        logger.info("Using Apple MPS GPU")
        return "mps"
    
    # Default to CPU
    logger.info("Using CPU (no GPU available or configured)")
    return "cpu"


class TelegramAnalyzer:
    """Main class for Telegram analysis pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration."""
        self.config = Config(config_path)
        self.device = None
        self.checkpoint_manager = None
        
    def analyze_channel(
        self,
        input_path: str,
        output_dir: str,
        batch_size: Optional[int] = None,
        skip_langdetect: bool = False,
        verbose: bool = False,
    ) -> None:
        """Run the complete analysis pipeline."""
        
        # Override config with parameters
        if batch_size:
            self.config.set("processing.batch_size", batch_size)
        if skip_langdetect:
            self.config.set("processing.skip_langdetect", True)
        
        # Set up output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(output_path, enable_resume=True)
        
        # Save config snapshot for reproducibility
        self.config.save_snapshot(output_path)
        
        # Detect compute device
        device = detect_device()
        if not self.config.prefer_gpu:
            device = "cpu"
        self.device = device
        
        logger.info(f"Starting analysis pipeline with device: {device}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Batch size: {self.config.batch_size}")
        
        # Check for resume point
        resume_from = self.checkpoint_manager.get_resume_point()
        if resume_from:
            logger.info(f"Resuming pipeline from step: {resume_from}")
            return self._resume_pipeline(resume_from, input_path, output_path)
        else:
            logger.info("Starting pipeline from the beginning")
        
        # Step 1: Load and normalize data
        if not self.checkpoint_manager.is_step_completed("data_loading"):
            logger.info("=" * 50)
            logger.info("Step 1: Loading and normalizing data")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("data_loading", "start")
            
            data_loader = DataLoader(max_text_length=self.config.max_text_length)
            df = data_loader.load_data(
                input_path=input_path,
                format_type=self.config.input_format,
                text_col=self.config.text_column,
                id_col=self.config.id_column,
                date_col=self.config.date_column,
            )
            
            # Print data summary
            summary = data_loader.get_data_summary(df)
            logger.info(f"Data summary: {summary}")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("data_loading", df, summary)
            self.checkpoint_manager.force_garbage_collection("data_loading")
        else:
            logger.info("Loading data from checkpoint...")
            df, summary = self.checkpoint_manager.load_checkpoint("data_loading")
            data_loader = DataLoader(max_text_length=self.config.max_text_length)
        
        # Step 2: Language detection (optional)
        if not self.checkpoint_manager.is_step_completed("language_detection"):
            if not self.config.skip_langdetect:
                logger.info("=" * 50)
                logger.info("Step 2: Language detection")
                logger.info("=" * 50)
                
                self.checkpoint_manager.log_memory_usage("language_detection", "start")
                
                try:
                    from langdetect import detect
                    logger.info("Running language detection...")
                    df["language"] = df["text"].apply(
                        lambda x: detect(x) if len(x.strip()) > 10 else "unknown"
                    )
                    lang_counts = df["language"].value_counts()
                    lang_stats = {"language_distribution": dict(lang_counts.head())}
                    logger.info(f"Language distribution: {lang_stats['language_distribution']}")
                except Exception as e:
                    logger.warning(f"Language detection failed: {e}")
                    df["language"] = "unknown"
                    lang_stats = {"language_detection_failed": str(e)}
            else:
                df["language"] = "unknown"
                lang_stats = {"language_detection_skipped": True}
                logger.info("Skipping language detection")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("language_detection", df, lang_stats)
            self.checkpoint_manager.force_garbage_collection("language_detection")
        else:
            logger.info("Loading language detection from checkpoint...")
            df, lang_stats = self.checkpoint_manager.load_checkpoint("language_detection")
        
        # Step 2.5: Quote Detection and Speaker Attribution
        if not self.checkpoint_manager.is_step_completed("quote_detection"):
            logger.info("=" * 50)
            logger.info("Step 2.5: Quote Detection and Speaker Attribution")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("quote_detection", "start")
            
            quote_processor = QuoteProcessor(
                detect_forwarded=self.config.quote_aware,
                detect_quoted_spans=True,
                attribute_forwarded_to_source=True
            )
            
            df = quote_processor.process_dataframe(df)
            quote_stats = quote_processor.get_quote_stats(df)
            logger.info(f"Quote detection stats: {quote_stats}")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("quote_detection", df, quote_stats)
            self.checkpoint_manager.force_garbage_collection("quote_detection")
        else:
            logger.info("Loading quote detection from checkpoint...")
            df, quote_stats = self.checkpoint_manager.load_checkpoint("quote_detection")
        
        # Step 3: Named Entity Recognition
        if not self.checkpoint_manager.is_step_completed("entity_extraction"):
            logger.info("=" * 50)
            logger.info("Step 3: Named Entity Recognition")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("entity_extraction", "start")
            
            ner_processor = NERProcessor(
                model_name=self.config.ner_model,
                max_entities_per_msg=self.config.max_entities_per_msg,
                device=device,
            )
            
            # Load entity aliases
            aliases = self.config.load_aliases()
            if aliases:
                ner_processor.load_aliases(aliases)
            
            df = ner_processor.process_dataframe(df)
            entity_stats = ner_processor.get_entity_stats(df)
            logger.info(f"Entity extraction stats: {entity_stats}")
            
            # Save checkpoint with entity counts as additional data
            additional_data = {}
            if entity_stats and "top_entities" in entity_stats:
                entity_counts_df = pd.DataFrame(
                    list(entity_stats["top_entities"].items()), 
                    columns=["entity", "count"]
                )
                additional_data["entity_counts"] = entity_counts_df
            
            self.checkpoint_manager.save_checkpoint("entity_extraction", df, entity_stats, additional_data)
            
            # Clean up processor to free memory
            del ner_processor
            self.checkpoint_manager.force_garbage_collection("entity_extraction")
        else:
            logger.info("Loading entity extraction from checkpoint...")
            df, entity_stats = self.checkpoint_manager.load_checkpoint("entity_extraction")
        
        # Step 4: Sentiment Analysis
        if not self.checkpoint_manager.is_step_completed("sentiment_analysis"):
            logger.info("=" * 50)
            logger.info("Step 4: Sentiment Analysis")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("sentiment_analysis", "start")
            
            sentiment_processor = SentimentProcessor(
                model_name=self.config.sentiment_model,
                device=device,
            )
            
            df = sentiment_processor.process_dataframe(df)
            sentiment_stats = sentiment_processor.get_sentiment_stats(df)
            logger.info(f"Sentiment analysis stats: {sentiment_stats}")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("sentiment_analysis", df, sentiment_stats)
            
            # Clean up processor to free memory
            del sentiment_processor
            self.checkpoint_manager.force_garbage_collection("sentiment_analysis")
        else:
            logger.info("Loading sentiment analysis from checkpoint...")
            df, sentiment_stats = self.checkpoint_manager.load_checkpoint("sentiment_analysis")
        
        # Step 5: Toxicity Detection
        if not self.checkpoint_manager.is_step_completed("toxicity_detection"):
            logger.info("=" * 50)
            logger.info("Step 5: Toxicity Detection")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("toxicity_detection", "start")
            
            toxicity_processor = ToxicityProcessor(
                model_name=self.config.toxicity_model,
                device=device,
            )
            
            df = toxicity_processor.process_dataframe(df)
            toxicity_stats = toxicity_processor.get_toxicity_stats(df)
            logger.info(f"Toxicity analysis stats: {toxicity_stats}")
            
            # Save checkpoint with top toxic messages
            additional_data = {}
            if "toxicity_score" in df.columns:
                toxic_messages = toxicity_processor.get_most_toxic_messages(df, top_k=20)
                if not toxic_messages.empty:
                    additional_data["toxic_messages"] = toxic_messages
            
            self.checkpoint_manager.save_checkpoint("toxicity_detection", df, toxicity_stats, additional_data)
            
            # Clean up processor to free memory  
            del toxicity_processor
            self.checkpoint_manager.force_garbage_collection("toxicity_detection")
        else:
            logger.info("Loading toxicity detection from checkpoint...")
            df, toxicity_stats = self.checkpoint_manager.load_checkpoint("toxicity_detection")
        
        # Step 6: Stance Classification  
        if not self.checkpoint_manager.is_step_completed("stance_classification"):
            logger.info("=" * 50)
            logger.info("Step 6: Stance Classification")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("stance_classification", "start")
            
            stance_processor = StanceProcessor(
                model_name=self.config.stance_model,
                stance_threshold=self.config.stance_threshold,
                device=device,
            )
            
            df = stance_processor.process_dataframe(df)
            stance_stats = stance_processor.get_stance_stats(df)
            logger.info(f"Stance classification stats: {stance_stats}")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("stance_classification", df, stance_stats)
            
            # Clean up processor to free memory
            del stance_processor
            self.checkpoint_manager.force_garbage_collection("stance_classification")
        else:
            logger.info("Loading stance classification from checkpoint...")
            df, stance_stats = self.checkpoint_manager.load_checkpoint("stance_classification")
        
        # Step 7: Style Feature Extraction
        if not self.checkpoint_manager.is_step_completed("style_extraction"):
            logger.info("=" * 50)
            logger.info("Step 7: Style Feature Extraction")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("style_extraction", "start")
            
            style_processor = StyleProcessor()
            
            df = style_processor.process_dataframe(df)
            style_stats = style_processor.get_style_stats(df)
            logger.info(f"Style feature extraction stats: {style_stats}")
            
            # Export style features to JSON immediately
            style_json_path = output_path / "channel_style_features.json"
            style_processor.export_style_features_json(df, style_json_path)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("style_extraction", df, style_stats)
            
            # Clean up processor to free memory
            del style_processor
            self.checkpoint_manager.force_garbage_collection("style_extraction")
        else:
            logger.info("Loading style extraction from checkpoint...")
            df, style_stats = self.checkpoint_manager.load_checkpoint("style_extraction")
        
        # Step 8: Topic Classification
        if not self.checkpoint_manager.is_step_completed("topic_classification"):
            logger.info("=" * 50)
            logger.info("Step 8: Topic Classification")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("topic_classification", "start")
            
            # Check if topic discovery is enabled
            enable_discovery = getattr(self.config, 'enable_topic_discovery', False)
            discovery_params = getattr(self.config, 'topic_discovery', {})
            
            try:
                # Use safe wrapper for discovery if enabled
                if enable_discovery:
                    logger.info("Using safe BERTopic wrapper for topic discovery")
                    
                    # Initialize safe wrapper with conservative parameters
                    safe_bertopic = SafeBERTopicWrapper(
                        timeout_seconds=600,  # 10 minutes max
                        max_memory_mb=4096,    # 4GB memory limit
                        enable_multiprocessing=False  # Disabled for stability
                    )
                    
                    # Process dataframe with safe discovery
                    df = safe_bertopic.safe_dataframe_processing(
                        df, 
                        text_column="text", 
                        discovery_params=discovery_params
                    )
                    
                    # Create mock topic stats for consistency
                    topic_stats = {
                        "discovery_enabled": True,
                        "safe_mode": True,
                        "total_messages": len(df),
                        "topic_distribution": dict(df['discovery_topic_label'].value_counts().head())
                    }
                    
                    # Export discovery analysis immediately
                    discovery_json_path = output_path / "channel_discovery_topics.json"
                    discovery_analysis = {
                        "metadata": {
                            "total_messages": len(df),
                            "safe_mode": True,
                            "extraction_date": pd.Timestamp.now().isoformat()
                        },
                        "discovery_stats": topic_stats
                    }
                    
                    with open(discovery_json_path, 'w') as f:
                        json.dump(discovery_analysis, f, indent=2, default=str)
                    
                    # Clean up safe wrapper
                    del safe_bertopic
                
                else:
                    # Use regular ontology-based topic classification only
                    topic_processor = TopicProcessor(
                        model_name=self.config.topic_model,
                        confidence_threshold=self.config.topic_threshold,
                        device=device,
                        batch_size=self.config.batch_size // 2,
                        enable_discovery=False,  # Disabled for safety
                        discovery_params={}
                    )
                    
                    # Load custom topics if available
                    custom_topics = self.config.load_topics()
                    if custom_topics:
                        topic_labels = [topic.get("label", topic.get("name", "")) for topic in custom_topics]
                        topic_processor.topics = topic_labels
                        logger.info(f"Using {len(topic_labels)} custom topics")
                    
                    df = topic_processor.process_dataframe(df)
                    topic_stats = topic_processor.get_topic_stats(df)
                    
                    # Clean up processor
                    del topic_processor
                
                logger.info(f"Topic classification stats: {topic_stats}")
                
                # Export topic analysis to JSON immediately
                topic_json_path = output_path / "channel_topic_analysis.json"
                topic_analysis = {
                    "metadata": {
                        "total_messages": len(df),
                        "extraction_date": pd.Timestamp.now().isoformat(),
                        "discovery_enabled": enable_discovery
                    },
                    "topic_stats": topic_stats
                }
                
                with open(topic_json_path, 'w') as f:
                    json.dump(topic_analysis, f, indent=2, default=str)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint("topic_classification", df, topic_stats)
                
            except Exception as e:
                logger.error(f"Topic classification failed: {e}")
                # Continue without topic classification
                topic_stats = {"error": str(e), "topic_classification_failed": True}
                self.checkpoint_manager.save_checkpoint("topic_classification", df, topic_stats)
            
            self.checkpoint_manager.force_garbage_collection("topic_classification")
        else:
            logger.info("Loading topic classification from checkpoint...")
            df, topic_stats = self.checkpoint_manager.load_checkpoint("topic_classification")
        
        # Step 9: Links & Domains Extraction
        if not self.checkpoint_manager.is_step_completed("link_extraction"):
            logger.info("=" * 50)
            logger.info("Step 9: Links & Domains Extraction")
            logger.info("=" * 50)
            
            self.checkpoint_manager.log_memory_usage("link_extraction", "start")
            
            links_processor = LinksProcessor()
            
            df = links_processor.process_dataframe(df)
            
            # Export domain analysis to JSON immediately
            domain_json_path = output_path / "channel_domain_analysis.json"
            links_processor.export_domain_analysis(df, domain_json_path)
            
            # Create stats for checkpoint
            links_stats = {
                "total_links": int(df['total_links'].sum()) if 'total_links' in df.columns else 0,
                "unique_domains": int(df['unique_domains'].sum()) if 'unique_domains' in df.columns else 0,
                "messages_with_links": int(df[df['total_links'] > 0].shape[0]) if 'total_links' in df.columns else 0
            }
            
            # Prepare domain counts for additional data
            additional_data = {}
            if "total_links" in df.columns:
                all_domain_counts = {}
                for _, row in df.iterrows():
                    if row['total_links'] > 0 and 'domain_counts' in row:
                        for domain, count in row['domain_counts'].items():
                            all_domain_counts[domain] = all_domain_counts.get(domain, 0) + count
                
                if all_domain_counts:
                    domain_counts_df = pd.DataFrame(
                        list(all_domain_counts.items()), 
                        columns=["domain", "count"]
                    ).sort_values('count', ascending=False)
                    additional_data["domain_counts"] = domain_counts_df
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("link_extraction", df, links_stats, additional_data)
            
            # Clean up processor
            del links_processor
            self.checkpoint_manager.force_garbage_collection("link_extraction")
        else:
            logger.info("Loading link extraction from checkpoint...")
            df, links_stats = self.checkpoint_manager.load_checkpoint("link_extraction")
        
        # Step 10: Save final results
        logger.info("=" * 50)
        logger.info("Step 10: Saving final results")
        logger.info("=" * 50)
        
        self.checkpoint_manager.log_memory_usage("final_output", "start")
        
        # Save main enriched dataset
        main_output = output_path / "posts_enriched.parquet"
        if hasattr(data_loader, 'save_processed_data'):
            data_loader.save_processed_data(df, main_output)
        else:
            df.to_parquet(main_output, index=False, compression="snappy")
        
        logger.info(f"Main dataset saved: {main_output}")
        
        # Generate and save derived outputs
        try:
            # Daily summary
            daily_summary = create_daily_summary(df)
            daily_summary.to_csv(output_path / "channel_daily_summary.csv", index=False)
            logger.info("Daily summary saved")
            
            # Entity stance timeline (if data available)
            if "stance_edges" in df.columns:
                entity_stance_counts = create_entity_stance_counts(df)
                if not entity_stance_counts.empty:
                    entity_stance_counts.to_csv(output_path / "channel_entity_stance_counts.csv", index=False)
                    logger.info("Entity stance counts saved")
                
                entity_stance_daily = create_entity_stance_daily(df)
                if not entity_stance_daily.empty:
                    entity_stance_daily.to_csv(output_path / "channel_entity_stance_daily.csv", index=False)
                    logger.info("Entity stance daily timeline saved")
            
            # Topic share daily (if data available)
            if any(col in df.columns for col in ["ontology_topic_label", "discovery_topic_label", "top_topic_label"]):
                topic_share_daily = create_topic_share_daily(df)
                if not topic_share_daily.empty:
                    topic_share_daily.to_csv(output_path / "channel_topic_share_daily.csv", index=False)
                    logger.info("Topic share daily timeline saved")
        
        except Exception as e:
            logger.error(f"Failed to generate derived outputs: {e}")
            logger.warning("Continuing with main results saved")
        
        # Generate final pipeline summary
        pipeline_summary = self.checkpoint_manager.get_pipeline_summary()
        with open(output_path / "pipeline_summary.json", 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        self.checkpoint_manager.log_memory_usage("final_output", "end")
        
        logger.info("=" * 50)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 50)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Main output: {main_output}")
        logger.info(f"Total messages processed: {len(df)}")
        logger.info(f"Pipeline summary: {pipeline_summary['progress_percent']:.1f}% complete")
    
    def _resume_pipeline(self, resume_from: str, input_path: str, output_path: Path) -> None:
        """Resume pipeline from a specific step.
        
        Args:
            resume_from: Name of the step to resume from
            input_path: Original input path (for reference)
            output_path: Output directory path
        """
        logger.info(f"Resuming pipeline from step: {resume_from}")
        
        # Load the last complete dataframe
        step_index = self.checkpoint_manager.pipeline_steps.index(resume_from)
        if step_index > 0:
            # Load from previous step
            prev_step = self.checkpoint_manager.pipeline_steps[step_index - 1]
            df, _ = self.checkpoint_manager.load_checkpoint(prev_step)
            if df is None:
                logger.error(f"Cannot resume: checkpoint for {prev_step} not found")
                return
        else:
            logger.error("Cannot resume: no previous checkpoint found")
            return
        
        logger.info(f"Loaded {len(df)} rows from checkpoint: {prev_step}")
        
        # Continue with the remaining steps by calling analyze_channel again
        # but with checkpoints already marking completed steps
        try:
            self.analyze_channel(
                input_path=input_path,
                output_dir=str(output_path),
                batch_size=None,
                skip_langdetect=False,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to resume pipeline: {e}")
            raise


def analyze_channel(
    config_path: str,
    input_path: str,
    output_dir: str,
    batch_size: Optional[int] = None,
    skip_langdetect: bool = False,
    verbose: bool = False,
) -> None:
    """Run the complete analysis pipeline (functional interface)."""
    analyzer = TelegramAnalyzer(config_path)
    analyzer.analyze_channel(
        input_path=input_path,
        output_dir=output_dir,
        batch_size=batch_size,
        skip_langdetect=skip_langdetect,
        verbose=verbose
    )
    


def create_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily summary statistics."""
    df["date_only"] = df["date"].dt.date
    
    # Base aggregations
    agg_dict = {
        "msg_id": "count",
        "text": lambda x: x.str.len().mean(),
        "entity_count": "mean",
    }
    
    # Add sentiment aggregations if available
    if "sentiment_score" in df.columns:
        agg_dict["sentiment_score"] = "mean"
    
    # Add toxicity aggregations if available
    if "toxicity_score" in df.columns:
        agg_dict["toxicity_score"] = ["mean", "max"]
    
    # Add links aggregations if available
    if "total_links" in df.columns:
        agg_dict["total_links"] = "sum"
        agg_dict["unique_domains"] = "sum"
    
    daily_stats = df.groupby("date_only").agg(agg_dict).round(3)
    
    # Flatten column names
    daily_stats.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col 
                          for col in daily_stats.columns]
    
    # Rename columns to match expected output format
    column_mapping = {
        "msg_id_count": "message_count",
        "text_<lambda>": "avg_text_length", 
        "entity_count_mean": "avg_entities",
        "sentiment_score_mean": "avg_sentiment",
        "toxicity_score_mean": "avg_toxicity",
        "toxicity_score_max": "max_toxicity",
        "total_links_sum": "total_links",
        "unique_domains_sum": "total_unique_domains"
    }
    
    daily_stats = daily_stats.rename(columns=column_mapping)
    daily_stats = daily_stats.reset_index()
    daily_stats["date"] = daily_stats["date_only"].astype(str)
    daily_stats = daily_stats.drop("date_only", axis=1)
    
    return daily_stats


def create_entity_stance_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create entity stance counts aggregation."""
    stance_counts = {}
    
    for _, row in df.iterrows():
        if row.get('stance_edges') and isinstance(row['stance_edges'], list):
            for edge in row['stance_edges']:
                if isinstance(edge, dict):
                    entity = edge.get('target', '')
                    label = edge.get('label', 'neutral')
                    score = edge.get('score', 0.0)
                    
                    if entity and label in ['support', 'oppose']:
                        key = (entity, label)
                        if key not in stance_counts:
                            stance_counts[key] = {'count': 0, 'total_score': 0.0}
                        stance_counts[key]['count'] += 1
                        stance_counts[key]['total_score'] += score
    
    if not stance_counts:
        return pd.DataFrame()
    
    # Convert to DataFrame
    rows = []
    for (entity, label), data in stance_counts.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0.0
        rows.append({
            'entity': entity,
            'stance': label,
            'count': data['count'],
            'avg_score': round(avg_score, 3)
        })
    
    result_df = pd.DataFrame(rows)
    return result_df.sort_values(['entity', 'count'], ascending=[True, False])


def create_entity_stance_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily entity stance timeline."""
    if 'date' not in df.columns:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['date_only'] = df_copy['date'].dt.date
    
    daily_stance = []
    
    for date in df_copy['date_only'].unique():
        day_df = df_copy[df_copy['date_only'] == date]
        
        entity_stances = {}
        for _, row in day_df.iterrows():
            if row.get('stance_edges') and isinstance(row['stance_edges'], list):
                for edge in row['stance_edges']:
                    if isinstance(edge, dict):
                        entity = edge.get('target', '')
                        label = edge.get('label', 'neutral')
                        score = edge.get('score', 0.0)
                        
                        if entity and label in ['support', 'oppose']:
                            key = (entity, label)
                            if key not in entity_stances:
                                entity_stances[key] = {'count': 0, 'total_score': 0.0}
                            entity_stances[key]['count'] += 1
                            entity_stances[key]['total_score'] += score
        
        for (entity, label), data in entity_stances.items():
            avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0.0
            daily_stance.append({
                'date': str(date),
                'entity': entity,
                'stance': label,
                'count': data['count'],
                'avg_score': round(avg_score, 3)
            })
    
    if not daily_stance:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(daily_stance)
    return result_df.sort_values(['date', 'entity', 'stance'])


def create_topic_share_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily topic distribution."""
    if 'date' not in df.columns or 'top_topic_label' not in df.columns:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['date_only'] = df_copy['date'].dt.date
    
    # Filter out messages without topics
    df_topics = df_copy[df_copy['top_topic_label'].notna() & (df_copy['top_topic_label'] != '')]
    
    if df_topics.empty:
        return pd.DataFrame()
    
    # Group by date and topic, count messages
    topic_daily = df_topics.groupby(['date_only', 'top_topic_label']).agg({
        'msg_id': 'count'
    }).reset_index()
    
    # Calculate total messages per day for percentage
    daily_totals = df_topics.groupby('date_only')['msg_id'].count().to_dict()
    
    topic_daily['total_messages_day'] = topic_daily['date_only'].map(daily_totals)
    topic_daily['percentage'] = (topic_daily['msg_id'] / topic_daily['total_messages_day'] * 100).round(2)
    
    # Rename columns
    topic_daily = topic_daily.rename(columns={
        'date_only': 'date',
        'top_topic_label': 'topic',
        'msg_id': 'count'
    })
    
    # Convert date to string
    topic_daily['date'] = topic_daily['date'].astype(str)
    
    # Drop intermediate column
    topic_daily = topic_daily.drop('total_messages_day', axis=1)
    
    return topic_daily.sort_values(['date', 'count'], ascending=[True, False])


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Raigem0n: Telegram Stance & Language Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python telegram_analyzer.py --config config/config.yaml --input data/channel.json --out out/run-001
  
  # Skip language detection for monolingual data
  python telegram_analyzer.py --config config.yaml --skip-langdetect --input data.json --out results/
  
  # Adjust batch size for available memory
  python telegram_analyzer.py --config config.yaml --batch-size 16 --input data.json --out results/
  
  # Verbose logging
  python telegram_analyzer.py --config config.yaml --verbose --input data.json --out results/
        """,
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input data file (JSON, JSONL, or CSV)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--skip-langdetect", 
        action="store_true",
        help="Skip language detection (for monolingual data)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Raigem0n Telegram Analysis Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        analyze_channel(
            config_path=args.config,
            input_path=args.input,
            output_dir=args.out,
            batch_size=args.batch_size,
            skip_langdetect=args.skip_langdetect,
            verbose=args.verbose,
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()