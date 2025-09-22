"""Sentiment analysis processor using RoBERTa-based models."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class SentimentProcessor:
    """Sentiment analysis using Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[str] = None,
    ) -> None:
        """Initialize sentiment processor."""
        self.model_name = model_name
        self.device = device
        self.sentiment_pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Load sentiment analysis model."""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment analysis results with label and confidence
        """
        if not self.sentiment_pipeline:
            raise RuntimeError("Sentiment model not loaded")

        logger.debug(f"Analyzing sentiment for {len(texts)} texts")
        results = []

        for text in texts:
            try:
                # Get sentiment scores for all labels
                raw_results = self.sentiment_pipeline(text)
                
                # Process results into standard format
                sentiment_result = self._process_sentiment_result(raw_results[0])
                results.append(sentiment_result)

            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for text: {e}")
                results.append({
                    "sentiment_label": "unknown",
                    "sentiment_score": 0.0,
                    "all_scores": {}
                })

        return results

    def _process_sentiment_result(self, raw_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw sentiment analysis result into standard format.
        
        Args:
            raw_result: Raw result from sentiment pipeline
            
        Returns:
            Processed sentiment result
        """
        # Find the prediction with highest confidence
        best_prediction = max(raw_result, key=lambda x: x["score"])
        
        # Normalize label names
        label_mapping = {
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral", 
            "POSITIVE": "positive",
            "LABEL_0": "negative",  # Some models use numeric labels
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        }
        
        sentiment_label = label_mapping.get(
            best_prediction["label"].upper(), 
            best_prediction["label"].lower()
        )
        
        # Create scores dictionary for all labels
        all_scores = {}
        for prediction in raw_result:
            normalized_label = label_mapping.get(
                prediction["label"].upper(), 
                prediction["label"].lower()
            )
            all_scores[normalized_label] = float(prediction["score"])
        
        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": float(best_prediction["score"]),
            "all_scores": all_scores
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add sentiment information.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        logger.info(f"Processing {len(df)} messages for sentiment analysis")
        
        # Analyze sentiment in batches
        batch_size = 50  # Reasonable batch size for memory management
        all_results = []
        
        for i in range(0, len(df), batch_size):
            batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
            batch_results = self.analyze_sentiment(batch_texts)
            all_results.extend(batch_results)
            
            if (i + batch_size) % 200 == 0:
                logger.info(f"Processed {min(i + batch_size, len(df))} messages")
        
        # Add results to dataframe
        df = df.copy()
        
        # Extract individual columns from results
        df["sentiment_label"] = [result["sentiment_label"] for result in all_results]
        df["sentiment_score"] = [result["sentiment_score"] for result in all_results]
        df["sentiment_all_scores"] = [result["all_scores"] for result in all_results]
        
        # Log summary statistics
        sentiment_counts = df["sentiment_label"].value_counts()
        avg_confidence = df["sentiment_score"].mean()
        
        logger.info(f"Sentiment analysis completed.")
        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        return df

    def get_sentiment_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about sentiment analysis results.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with sentiment statistics
        """
        if "sentiment_label" not in df.columns:
            return {}

        sentiment_counts = df["sentiment_label"].value_counts()
        
        return {
            "total_messages": len(df),
            "sentiment_distribution": dict(sentiment_counts),
            "avg_confidence": df["sentiment_score"].mean() if len(df) > 0 else 0.0,
            "confidence_by_sentiment": df.groupby("sentiment_label")["sentiment_score"].mean().to_dict(),
        }