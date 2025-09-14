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
from raigem0n.processors import NERProcessor


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
        
        # Step 4: Save intermediate results
        logger.info("=" * 50)
        logger.info("Saving results")
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
    
    daily_stats = df.groupby("date_only").agg({
        "msg_id": "count",
        "text": lambda x: x.str.len().mean(),
        "entity_count": "mean",
    }).round(2)
    
    daily_stats.columns = ["message_count", "avg_text_length", "avg_entities"]
    daily_stats = daily_stats.reset_index()
    daily_stats["date"] = daily_stats["date_only"].astype(str)
    daily_stats = daily_stats.drop("date_only", axis=1)
    
    return daily_stats


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