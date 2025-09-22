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
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from raigem0n.config import Config
from raigem0n.data_loader import DataLoader
from raigem0n.processors import NERProcessor, SentimentProcessor, StyleProcessor, TopicProcessor, ToxicityProcessor, StanceProcessor, LinksProcessor


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
        
        # Step 1: Load and normalize data
        logger.info("=" * 50)
        logger.info("Step 1: Loading and normalizing data")
        logger.info("=" * 50)
        
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
        
        # Step 2: Language detection (optional)
        if not self.config.skip_langdetect:
            logger.info("=" * 50)
            logger.info("Step 2: Language detection")
            logger.info("=" * 50)
            
            try:
                from langdetect import detect
                logger.info("Running language detection...")
                df["language"] = df["text"].apply(
                    lambda x: detect(x) if len(x.strip()) > 10 else "unknown"
                )
                lang_counts = df["language"].value_counts()
                logger.info(f"Language distribution: {dict(lang_counts.head())}")
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                df["language"] = "unknown"
        else:
            df["language"] = "unknown"
            logger.info("Skipping language detection")
        
        # Step 3: Named Entity Recognition
        logger.info("=" * 50)
        logger.info("Step 3: Named Entity Recognition")
        logger.info("=" * 50)
        
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
        
        # Step 4: Sentiment Analysis
        logger.info("=" * 50)
        logger.info("Step 4: Sentiment Analysis")
        logger.info("=" * 50)
        
        sentiment_processor = SentimentProcessor(
            model_name=self.config.sentiment_model,
            device=device,
        )
        
        df = sentiment_processor.process_dataframe(df)
        sentiment_stats = sentiment_processor.get_sentiment_stats(df)
        logger.info(f"Sentiment analysis stats: {sentiment_stats}")
        
        # Step 5: Toxicity Detection
        logger.info("=" * 50)
        logger.info("Step 5: Toxicity Detection")
        logger.info("=" * 50)
        
        toxicity_processor = ToxicityProcessor(
            model_name=self.config.toxicity_model,
            device=device,
        )
        
        df = toxicity_processor.process_dataframe(df)
        toxicity_stats = toxicity_processor.get_toxicity_stats(df)
        logger.info(f"Toxicity analysis stats: {toxicity_stats}")
        
        # Step 6: Stance Classification
        logger.info("=" * 50)
        logger.info("Step 6: Stance Classification")
        logger.info("=" * 50)
        
        stance_processor = StanceProcessor(
            model_name=self.config.stance_model,
            stance_threshold=self.config.stance_threshold,
            device=device,
        )
        
        df = stance_processor.process_dataframe(df)
        stance_stats = stance_processor.get_stance_stats(df)
        logger.info(f"Stance classification stats: {stance_stats}")
        
        # Step 7: Style Feature Extraction
        logger.info("=" * 50)
        logger.info("Step 7: Style Feature Extraction")
        logger.info("=" * 50)
        
        style_processor = StyleProcessor()
        
        df = style_processor.process_dataframe(df)
        style_stats = style_processor.get_style_stats(df)
        logger.info(f"Style feature extraction stats: {style_stats}")
        
        # Export style features to JSON
        style_json_path = output_path / "channel_style_features.json"
        style_processor.export_style_features_json(df, style_json_path)
        
        # Step 8: Topic Classification
        logger.info("=" * 50)
        logger.info("Step 8: Topic Classification")
        logger.info("=" * 50)
        
        topic_processor = TopicProcessor(
            model_name=self.config.topic_model,
            confidence_threshold=self.config.topic_threshold,
            device=device,
            batch_size=self.config.batch_size // 2,  # Use smaller batch for topic classification
        )
        
        # Load custom topics if available
        custom_topics = self.config.load_topics()
        if custom_topics:
            topic_labels = [topic.get("label", topic.get("name", "")) for topic in custom_topics]
            topic_processor.topics = topic_labels
            logger.info(f"Using {len(topic_labels)} custom topics")
        
        df = topic_processor.process_dataframe(df)
        topic_stats = topic_processor.get_topic_stats(df)
        logger.info(f"Topic classification stats: {topic_stats}")
        
        # Export topic analysis to JSON
        topic_json_path = output_path / "channel_topic_analysis.json"
        topic_processor.export_topic_analysis_json(df, topic_json_path)
        
        # Step 9: Links & Domains Extraction
        logger.info("=" * 50)
        logger.info("Step 9: Links & Domains Extraction")
        logger.info("=" * 50)
        
        links_processor = LinksProcessor()
        
        df = links_processor.process_dataframe(df)
        
        # Export domain analysis to JSON
        domain_json_path = output_path / "channel_domain_analysis.json"
        links_processor.export_domain_analysis(df, domain_json_path)
        
        # Step 10: Save results
        logger.info("=" * 50)
        logger.info("Step 10: Saving results")
        logger.info("=" * 50)
        
        # Save main enriched dataset
        main_output = output_path / "posts_enriched.parquet"
        data_loader.save_processed_data(df, main_output)
        
        # Save daily summary
        daily_summary = create_daily_summary(df)
        daily_summary.to_csv(output_path / "channel_daily_summary.csv", index=False)
        
        # Save entity counts
        if entity_stats:
            entity_counts_df = pd.DataFrame(list(entity_stats["top_entities"].items()), 
                                           columns=["entity", "count"])
            entity_counts_df.to_csv(output_path / "channel_entity_counts.csv", index=False)
        
        # Save most toxic messages
        if "toxicity_score" in df.columns:
            toxic_messages = toxicity_processor.get_most_toxic_messages(df, top_k=20)
            if not toxic_messages.empty:
                toxic_messages.to_csv(output_path / "channel_top_toxic_messages.csv", index=False)
        
        # Save domain counts
        if "total_links" in df.columns:
            # Aggregate domain counts across all messages
            all_domain_counts = {}
            for _, row in df.iterrows():
                if row['total_links'] > 0:
                    for domain, count in row['domain_counts'].items():
                        all_domain_counts[domain] = all_domain_counts.get(domain, 0) + count
            
            if all_domain_counts:
                domain_counts_df = pd.DataFrame(list(all_domain_counts.items()), 
                                               columns=["domain", "count"])
                domain_counts_df = domain_counts_df.sort_values('count', ascending=False)
                domain_counts_df.to_csv(output_path / "channel_domain_counts.csv", index=False)
        
        # Save entity stance counts and timeline
        if "stance_edges" in df.columns:
            entity_stance_counts = create_entity_stance_counts(df)
            if not entity_stance_counts.empty:
                entity_stance_counts.to_csv(output_path / "channel_entity_stance_counts.csv", index=False)
            
            entity_stance_daily = create_entity_stance_daily(df)
            if not entity_stance_daily.empty:
                entity_stance_daily.to_csv(output_path / "channel_entity_stance_daily.csv", index=False)
        
        # Save topic share daily
        if "top_topic_label" in df.columns:
            topic_share_daily = create_topic_share_daily(df)
            if not topic_share_daily.empty:
                topic_share_daily.to_csv(output_path / "channel_topic_share_daily.csv", index=False)
        
        logger.info("=" * 50)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 50)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Main output: {main_output}")
        logger.info(f"Total messages processed: {len(df)}")


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