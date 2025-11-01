#!/usr/bin/env python3
"""
Command-line interface for raigem0n Telegram analysis pipeline.

This module provides the main entry point when raigem0n is installed as a package.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point for raigem0n package."""
    # Import telegram_analyzer from project root
    # We need to handle both installed package and dev mode
    try:
        # Try to import from installed location
        import telegram_analyzer
    except ImportError:
        # Fall back to project root for development
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        import telegram_analyzer

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Raigem0n: Telegram Stance & Language Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  raigem0n --config config/config.yaml --input data/channel.json --out out/run-$(date +%Y%m%d-%H%M)

  # Skip language detection
  raigem0n --config config.yaml --input data.json --out output/ --skip-langdetect

  # Verbose logging
  raigem0n --config config.yaml --input data.json --out output/ -v

For more information, visit: https://github.com/yourusername/v0lur
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/config.yaml)"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Telegram export (JSON, JSONL, or CSV)"
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for analysis results"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (overrides config)"
    )

    parser.add_argument(
        "--skip-langdetect",
        action="store_true",
        help="Skip language detection step"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="raigem0n 1.2.0 (Memory-Safe Release)"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    telegram_analyzer.setup_logging(level=log_level)

    # Create analyzer and run pipeline
    analyzer = telegram_analyzer.TelegramAnalyzer(args.config)

    try:
        analyzer.analyze_channel(
            input_path=args.input,
            output_dir=args.out,
            batch_size=args.batch_size,
            skip_langdetect=args.skip_langdetect,
            verbose=args.verbose
        )
        print(f"\n✅ Analysis complete! Results saved to: {args.out}")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
