#!/usr/bin/env python3
"""
Milestone 7: Evaluation Framework for v0lur
Main evaluation engine for systematic accuracy assessment.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from raigem0n.config import Config
from raigem0n.telegram_analyzer import TelegramAnalyzer

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Main evaluation orchestrator for v0lur accuracy assessment."""
    
    def __init__(self, gold_standard_path: str, config_path: str):
        """Initialize evaluation engine.
        
        Args:
            gold_standard_path: Path to gold standard annotations
            config_path: Path to system configuration
        """
        self.gold_standard_path = Path(gold_standard_path)
        self.config_path = Path(config_path)
        
        # Load gold standard data
        self.gold_data = self.load_gold_standard()
        self.config = Config(str(config_path))
        
        logger.info(f"Loaded {len(self.gold_data)} gold standard messages")
    
    def load_gold_standard(self) -> List[Dict[str, Any]]:
        """Load gold standard annotations from JSON file."""
        if not self.gold_standard_path.exists():
            raise FileNotFoundError(f"Gold standard file not found: {self.gold_standard_path}")
        
        with open(self.gold_standard_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_system_on_gold_data(self) -> pd.DataFrame:
        """Run the v0lur system on gold standard messages.
        
        Returns:
            DataFrame with system predictions for all gold messages
        """
        logger.info("Running v0lur system on gold standard data...")
        
        # Create temporary input file with gold standard messages
        temp_input = Path("temp_gold_input.json")
        
        # Convert gold standard to system input format
        system_input = []
        for item in self.gold_data:
            system_input.append({
                "msg_id": item["message_id"],
                "text": item["text"],
                "date": "2023-01-01T00:00:00",  # Dummy date
            })
        
        with open(temp_input, 'w', encoding='utf-8') as f:
            json.dump(system_input, f, indent=2)
        
        try:
            # Run system analysis
            analyzer = TelegramAnalyzer(str(self.config_path))
            analyzer.analyze_channel(
                input_path=str(temp_input),
                output_dir="temp_eval_output",
                batch_size=8,
                verbose=False
            )
            
            # Load results
            results_path = Path("temp_eval_output/posts_enriched.parquet")
            predictions_df = pd.read_parquet(results_path)
            
            # Clean up temporary files
            temp_input.unlink()
            
            return predictions_df
        
        except Exception as e:
            logger.error(f"System evaluation failed: {e}")
            if temp_input.exists():
                temp_input.unlink()
            raise
    
    def run_full_evaluation(self, predictions_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation suite.
        
        Args:
            predictions_df: Optional pre-computed system predictions
            
        Returns:
            Dictionary with all evaluation results
        """
        if predictions_df is None:
            predictions_df = self.run_system_on_gold_data()
        
        logger.info("Running comprehensive evaluation...")
        
        results = {
            'metadata': {
                'evaluation_time': datetime.now().isoformat(),
                'gold_standard_path': str(self.gold_standard_path),
                'config_path': str(self.config_path),
                'total_messages': len(self.gold_data),
            },
            'attribution_metrics': self.evaluate_attribution(predictions_df),
            'entity_metrics': self.evaluate_entities(predictions_df),
            'stance_metrics': self.evaluate_stance(predictions_df),
            'error_analysis': self.analyze_errors(predictions_df),
            'system_stats': self.compute_system_stats(predictions_df)
        }
        
        return results
    
    def evaluate_attribution(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate speaker attribution accuracy.
        
        Args:
            predictions_df: System predictions with quote spans
            
        Returns:
            Attribution accuracy metrics
        """
        logger.info("Evaluating speaker attribution accuracy...")
        
        # Import attribution metrics
        from metrics.attribution_metrics import compute_attribution_metrics
        
        # Align predictions with gold standard
        aligned_data = self.align_predictions_with_gold(predictions_df, 'attribution')
        
        return compute_attribution_metrics(
            predictions=aligned_data['predictions'],
            gold_standard=aligned_data['gold']
        )
    
    def evaluate_entities(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate named entity recognition accuracy.
        
        Args:
            predictions_df: System predictions with entities
            
        Returns:
            Entity recognition metrics
        """
        logger.info("Evaluating entity recognition accuracy...")
        
        from metrics.entity_metrics import compute_entity_metrics
        
        aligned_data = self.align_predictions_with_gold(predictions_df, 'entities')
        
        return compute_entity_metrics(
            predictions=aligned_data['predictions'],
            gold_standard=aligned_data['gold']
        )
    
    def evaluate_stance(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate stance classification accuracy.
        
        Args:
            predictions_df: System predictions with stance edges
            
        Returns:
            Stance classification metrics
        """
        logger.info("Evaluating stance classification accuracy...")
        
        from metrics.stance_metrics import compute_stance_metrics
        
        aligned_data = self.align_predictions_with_gold(predictions_df, 'stance')
        
        return compute_stance_metrics(
            predictions=aligned_data['predictions'],
            gold_standard=aligned_data['gold']
        )
    
    def analyze_errors(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed error analysis.
        
        Args:
            predictions_df: System predictions
            
        Returns:
            Error analysis results with categorized failures
        """
        logger.info("Performing error analysis...")
        
        from metrics.error_analysis import analyze_prediction_errors
        
        aligned_data = self.align_predictions_with_gold(predictions_df, 'all')
        
        return analyze_prediction_errors(
            predictions=aligned_data['predictions'],
            gold_standard=aligned_data['gold']
        )
    
    def compute_system_stats(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute system-level statistics.
        
        Args:
            predictions_df: System predictions
            
        Returns:
            System performance statistics
        """
        stats = {
            'total_predictions': len(predictions_df),
            'processing_coverage': len(predictions_df) / len(self.gold_data),
        }
        
        # Quote detection stats
        if 'has_quotes' in predictions_df.columns:
            stats['quote_detection'] = {
                'messages_with_quotes': int(predictions_df['has_quotes'].sum()),
                'quote_percentage': float(predictions_df['has_quotes'].mean() * 100),
                'multi_speaker_messages': int(predictions_df.get('multi_speaker', pd.Series([False] * len(predictions_df))).sum())
            }
        
        # Entity stats
        if 'entities' in predictions_df.columns:
            all_entities = [entity for entities in predictions_df['entities'] for entity in (entities or [])]
            stats['entity_extraction'] = {
                'total_entities': len(all_entities),
                'avg_entities_per_message': len(all_entities) / len(predictions_df),
                'unique_entities': len(set(entity.get('text', '') for entity in all_entities))
            }
        
        # Stance stats
        if 'stance_edges' in predictions_df.columns:
            all_edges = [edge for edges in predictions_df['stance_edges'] for edge in (edges or [])]
            stats['stance_classification'] = {
                'total_stance_edges': len(all_edges),
                'avg_edges_per_message': len(all_edges) / len(predictions_df),
                'stance_distribution': {}
            }
            
            # Count stance types
            for edge in all_edges:
                label = edge.get('label', 'unknown')
                stats['stance_classification']['stance_distribution'][label] = \
                    stats['stance_classification']['stance_distribution'].get(label, 0) + 1
        
        return stats
    
    def align_predictions_with_gold(self, predictions_df: pd.DataFrame, task: str) -> Dict[str, List]:
        """Align system predictions with gold standard annotations.
        
        Args:
            predictions_df: System predictions DataFrame
            task: Task type ('attribution', 'entities', 'stance', 'all')
            
        Returns:
            Dictionary with aligned predictions and gold standard data
        """
        predictions = []
        gold_standard = []
        
        # Create message ID mapping
        pred_by_id = {row['msg_id']: row for _, row in predictions_df.iterrows()}
        
        for gold_item in self.gold_data:
            message_id = gold_item['message_id']
            
            if message_id in pred_by_id:
                pred_row = pred_by_id[message_id]
                
                if task in ['attribution', 'all']:
                    # Extract attribution data
                    pred_spans = pred_row.get('quote_spans', [])
                    gold_spans = gold_item['ground_truth'].get('spans', [])
                    
                    predictions.append({
                        'message_id': message_id,
                        'spans': pred_spans
                    })
                    
                    gold_standard.append({
                        'message_id': message_id,
                        'spans': gold_spans
                    })
                
                elif task in ['entities', 'all']:
                    # Extract entity data
                    pred_entities = pred_row.get('entities', [])
                    gold_entities = gold_item['ground_truth'].get('entities', [])
                    
                    predictions.append({
                        'message_id': message_id,
                        'entities': pred_entities
                    })
                    
                    gold_standard.append({
                        'message_id': message_id,
                        'entities': gold_entities
                    })
                
                elif task in ['stance', 'all']:
                    # Extract stance data
                    pred_edges = pred_row.get('stance_edges', [])
                    gold_edges = gold_item['ground_truth'].get('stance_edges', [])
                    
                    predictions.append({
                        'message_id': message_id,
                        'stance_edges': pred_edges
                    })
                    
                    gold_standard.append({
                        'message_id': message_id,
                        'stance_edges': gold_edges
                    })
        
        return {
            'predictions': predictions,
            'gold': gold_standard
        }
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {output_file}")


def main():
    """Main evaluation script entry point."""
    parser = argparse.ArgumentParser(description="v0lur Evaluation Framework")
    parser.add_argument(
        "--gold-standard", 
        required=True,
        help="Path to gold standard annotations JSON file"
    )
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to system configuration file"
    )
    parser.add_argument(
        "--output", 
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Enable detailed error analysis"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Initialize evaluation engine
        engine = EvaluationEngine(
            gold_standard_path=args.gold_standard,
            config_path=args.config
        )
        
        # Run evaluation
        results = engine.run_full_evaluation()
        
        # Save results
        engine.save_evaluation_results(results, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Gold standard messages: {results['metadata']['total_messages']}")
        
        if 'attribution_metrics' in results:
            attr_metrics = results['attribution_metrics']
            if 'span_accuracy' in attr_metrics:
                print(f"Span attribution accuracy: {attr_metrics['span_accuracy']:.3f}")
        
        if 'entity_metrics' in results:
            entity_metrics = results['entity_metrics']
            if 'exact_match_f1' in entity_metrics:
                print(f"Entity exact match F1: {entity_metrics['exact_match_f1']:.3f}")
        
        if 'stance_metrics' in results:
            stance_metrics = results['stance_metrics']
            if 'overall_f1' in stance_metrics:
                print(f"Stance classification F1: {stance_metrics['overall_f1']:.3f}")
        
        print(f"\nDetailed results saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()