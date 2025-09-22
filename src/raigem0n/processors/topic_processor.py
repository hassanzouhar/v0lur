"""Topic classification processor for content theme analysis."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class TopicProcessor:
    """Classify text content into predefined topic categories."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        batch_size: int = 16,
        confidence_threshold: float = 0.3,
        custom_topics: Optional[List[str]] = None,
    ) -> None:
        """Initialize topic processor.
        
        Args:
            model_name: Hugging Face model for zero-shot classification
            device: Device to run inference on ("cpu", "cuda", "mps")
            batch_size: Batch size for processing
            confidence_threshold: Minimum confidence for topic assignment
            custom_topics: Custom topic labels to use instead of defaults
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Default topic categories for political/social media analysis
        self.default_topics = [
            "politics and government",
            "elections and voting", 
            "immigration and border security",
            "economy and finance",
            "healthcare and medical",
            "education and schools",
            "crime and law enforcement",
            "foreign policy and international relations",
            "military and defense",
            "civil rights and social justice",
            "environment and climate",
            "technology and innovation",
            "media and journalism",
            "religion and faith",
            "entertainment and culture",
            "sports and recreation",
            "business and industry",
            "science and research",
            "conspiracy theories and misinformation",
            "other news and current events"
        ]
        
        self.topics = custom_topics or self.default_topics
        self.classifier = None
        
    def load_model(self) -> None:
        """Load the zero-shot classification model."""
        logger.info(f"Loading topic classification model: {self.model_name}")
        
        try:
            # Use zero-shot classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                batch_size=self.batch_size,
            )
            
            # Test the model with a simple example
            test_result = self.classifier(
                "This is a test message about politics and elections.",
                self.topics[:5]  # Test with first 5 topics
            )
            
            logger.info("Topic classification model loaded successfully")
            logger.info(f"Available topics: {len(self.topics)}")
            
        except Exception as e:
            logger.error(f"Failed to load topic classification model: {e}")
            raise

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text into topic categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with topic classification results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result()
            
        if self.classifier is None:
            self.load_model()
        
        try:
            # Get predictions for all topics
            result = self.classifier(text, self.topics)
            
            # Extract results
            labels = result["labels"]
            scores = result["scores"]
            
            # Filter by confidence threshold
            confident_topics = []
            for label, score in zip(labels, scores):
                if score >= self.confidence_threshold:
                    confident_topics.append({
                        "topic": label,
                        "confidence": float(score)
                    })
            
            # Get top topic
            top_topic = labels[0] if labels else "other"
            top_confidence = float(scores[0]) if scores else 0.0
            
            return {
                "primary_topic": top_topic,
                "primary_confidence": top_confidence,
                "all_topics": confident_topics,
                "topic_count": len(confident_topics),
                "classification_method": "zero_shot_mnli"
            }
            
        except Exception as e:
            logger.warning(f"Topic classification failed for text: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty classification result."""
        return {
            "primary_topic": "other",
            "primary_confidence": 0.0,
            "all_topics": [],
            "topic_count": 0,
            "classification_method": "failed"
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of classification results
        """
        if self.classifier is None:
            self.load_model()
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # Get batch predictions
                batch_results = self.classifier(batch_texts, self.topics)
                
                # Handle single vs multiple results
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
                
                # Process each result
                for result in batch_results:
                    labels = result["labels"]
                    scores = result["scores"]
                    
                    # Filter by confidence
                    confident_topics = []
                    for label, score in zip(labels, scores):
                        if score >= self.confidence_threshold:
                            confident_topics.append({
                                "topic": label,
                                "confidence": float(score)
                            })
                    
                    # Get top topic
                    top_topic = labels[0] if labels else "other"
                    top_confidence = float(scores[0]) if scores else 0.0
                    
                    results.append({
                        "primary_topic": top_topic,
                        "primary_confidence": top_confidence,
                        "all_topics": confident_topics,
                        "topic_count": len(confident_topics),
                        "classification_method": "zero_shot_mnli"
                    })
                    
            except Exception as e:
                logger.warning(f"Batch topic classification failed: {e}")
                # Add empty results for failed batch
                for _ in batch_texts:
                    results.append(self._empty_result())
        
        return results

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add topic classifications.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added topic classification columns
        """
        logger.info(f"Processing {len(df)} messages for topic classification")
        
        # Extract texts for batch processing
        texts = df[text_column].tolist()
        
        # Classify all texts
        all_results = self.classify_batch(texts)
        
        # Add results to dataframe
        df = df.copy()
        
        # Add topic classification results as JSON column
        df["topic_classification"] = all_results
        
        # Add key columns for easier analysis
        df["primary_topic"] = [r["primary_topic"] for r in all_results]
        df["primary_topic_confidence"] = [r["primary_confidence"] for r in all_results]
        df["topic_count"] = [r["topic_count"] for r in all_results]
        
        # Log summary statistics
        topic_distribution = df["primary_topic"].value_counts()
        avg_confidence = df["primary_topic_confidence"].mean()
        confident_classifications = sum(1 for r in all_results if r["primary_confidence"] >= self.confidence_threshold)
        
        logger.info(f"Topic classification completed.")
        logger.info(f"Topic distribution: {dict(topic_distribution.head(10))}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info(f"Confident classifications: {confident_classifications}/{len(df)} ({confident_classifications/len(df)*100:.1f}%)")
        
        return df

    def get_topic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about topic classifications.
        
        Args:
            df: DataFrame with topic classification results
            
        Returns:
            Dictionary with topic statistics
        """
        if "topic_classification" not in df.columns:
            return {}

        all_results = df["topic_classification"].tolist()
        
        if not all_results:
            return {}

        # Topic distribution
        primary_topics = [r.get("primary_topic", "other") for r in all_results]
        topic_counts = pd.Series(primary_topics).value_counts()
        
        # Confidence statistics
        confidences = [r.get("primary_confidence", 0.0) for r in all_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Multi-topic messages
        multi_topic_count = sum(1 for r in all_results if r.get("topic_count", 0) > 1)
        
        # Confident classifications
        confident_count = sum(1 for c in confidences if c >= self.confidence_threshold)
        
        return {
            "total_messages": len(df),
            "topic_distribution": dict(topic_counts.head(15)),
            "unique_topics_found": len(topic_counts),
            "avg_confidence": avg_confidence,
            "confident_classifications": confident_count,
            "confident_percentage": (confident_count / len(df)) * 100 if len(df) > 0 else 0,
            "multi_topic_messages": multi_topic_count,
            "multi_topic_percentage": (multi_topic_count / len(df)) * 100 if len(df) > 0 else 0,
            "confidence_threshold": self.confidence_threshold
        }

    def export_topic_analysis_json(self, df: pd.DataFrame, output_path: str) -> None:
        """Export topic analysis to JSON file.
        
        Args:
            df: DataFrame with topic classification results
            output_path: Path to save the JSON file
        """
        import json
        from pathlib import Path
        
        if "topic_classification" not in df.columns:
            logger.warning("No topic classifications found in dataframe")
            return
            
        output_path = Path(output_path)
        
        # Create analysis structure
        topic_data = {
            "metadata": {
                "total_messages": len(df),
                "extraction_date": pd.Timestamp.now().isoformat(),
                "processor_version": "1.0.0",
                "model_name": self.model_name,
                "confidence_threshold": self.confidence_threshold,
                "available_topics": self.topics
            },
            "summary_stats": self.get_topic_stats(df),
            "message_topics": []
        }
        
        # Add per-message topic data
        for idx, row in df.iterrows():
            message_data = {
                "msg_id": str(row.get("msg_id", idx)),
                "date": row.get("date").isoformat() if pd.notna(row.get("date")) else None,
                "text_preview": row.get("text", "")[:100] + "..." if len(row.get("text", "")) > 100 else row.get("text", ""),
                "topic_classification": row["topic_classification"]
            }
            topic_data["message_topics"].append(message_data)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(topic_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Topic analysis exported to {output_path}")
        logger.info(f"Exported {len(topic_data['message_topics'])} message topic classifications")

    def get_topic_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get topic trends over time.
        
        Args:
            df: DataFrame with topic classification results and dates
            
        Returns:
            DataFrame with topic trends by date
        """
        if "topic_classification" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()
        
        # Create daily topic summary
        df["date_only"] = df["date"].dt.date
        
        # Get primary topics by date
        daily_topics = df.groupby(["date_only", "primary_topic"]).size().unstack(fill_value=0)
        
        # Add totals and percentages
        daily_topics["total_messages"] = daily_topics.sum(axis=1)
        
        # Convert to percentages
        percentage_cols = [col for col in daily_topics.columns if col != "total_messages"]
        for col in percentage_cols:
            daily_topics[f"{col}_pct"] = (daily_topics[col] / daily_topics["total_messages"] * 100).round(2)
        
        return daily_topics.reset_index()