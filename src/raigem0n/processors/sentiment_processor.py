"""Sentiment analysis processor using RoBERTa-based models."""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore

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
        self._offline_mode = False
        self._load_model()

    def _load_model(self) -> None:
        """Load sentiment analysis model."""
        if pipeline is None:
            self.sentiment_pipeline = None
            self._offline_mode = True
            logger.warning("Transformers not available; using rule-based sentiment analysis")
            return

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
            # The heavy-weight model may be unavailable in constrained test
            # environments.  Falling back to a deterministic rule-based
            # implementation keeps the rest of the system operational.
            self.sentiment_pipeline = None
            self._offline_mode = True
            logger.warning(
                "Falling back to rule-based sentiment analysis: %s", e
            )

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment analysis results with label and confidence
        """
        if not texts:
            return []

        if self.sentiment_pipeline and not self._offline_mode:
            logger.debug(
                f"Analyzing sentiment for {len(texts)} texts with transformer pipeline"
            )
            results: List[Dict[str, Any]] = []
            for text in texts:
                try:
                    raw_results = self.sentiment_pipeline(text)
                    sentiment_result = self._process_sentiment_result(raw_results[0])
                    results.append(sentiment_result)
                except Exception as e:
                    logger.warning(f"Failed to analyze sentiment for text: {e}")
                    results.append(self._rule_based_sentiment(text))
            return results

        logger.debug("Analyzing sentiment with rule-based fallback")
        return [self._rule_based_sentiment(text) for text in texts]

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
            "sentiment": sentiment_label,
            "sentiment_score": float(best_prediction["score"]),
            "all_scores": all_scores
        }

    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback heuristic sentiment analysis for offline environments."""

        positive_words = {
            "love", "great", "excellent", "awesome", "wonderful",
            "good", "happy", "fantastic", "amazing", "like"
        }
        negative_words = {
            "hate", "terrible", "bad", "awful", "horrible",
            "sad", "angry", "disappointing", "worst", "poor"
        }

        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "all_scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            }

        pos_hits = sum(1 for token in tokens if token in positive_words)
        neg_hits = sum(1 for token in tokens if token in negative_words)

        if pos_hits == 0 and neg_hits == 0:
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "all_scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            }

        raw_score = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
        normalized_score = (raw_score + 1) / 2  # Map [-1,1] -> [0,1]

        if raw_score > 0.1:
            sentiment_label = "positive"
        elif raw_score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        neutral_confidence = max(0.0, min(1.0, 1 - abs(raw_score)))
        positive_confidence = max(0.0, min(1.0, normalized_score))
        negative_confidence = max(0.0, min(1.0, 1 - normalized_score))

        return {
            "sentiment": sentiment_label,
            "sentiment_score": float(max(0.0, min(1.0, normalized_score))),
            "all_scores": {
                "positive": positive_confidence,
                "neutral": neutral_confidence,
                "negative": negative_confidence,
            },
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

        if df.empty:
            df = df.copy()
            df["sentiment"] = []
            df["sentiment_score"] = []
            df["sentiment_all_scores"] = []
            df["sentiment_label"] = []
            return df

        all_results: List[Dict[str, Any]] = []

        for i in range(0, len(df), batch_size):
            batch_texts = df[text_column].iloc[i:i + batch_size].tolist()
            batch_results = self.analyze_sentiment(batch_texts)
            all_results.extend(batch_results)

            if (i + batch_size) % 200 == 0 and i != 0:
                logger.info(f"Processed {min(i + batch_size, len(df))} messages")

        # Add results to dataframe
        df = df.copy()

        df["sentiment"] = [result["sentiment"] for result in all_results]
        df["sentiment_label"] = df["sentiment"]  # Backwards compatibility
        df["sentiment_score"] = [result["sentiment_score"] for result in all_results]
        df["sentiment_all_scores"] = [result.get("all_scores", {}) for result in all_results]
        
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
        if "sentiment" not in df.columns:
            return {}

        sentiment_percentages = (
            df["sentiment"].value_counts(normalize=True) * 100
        ).round(2)

        return {
            "total_messages": len(df),
            "sentiment_distribution": sentiment_percentages.to_dict(),
            "avg_confidence": df["sentiment_score"].mean() if len(df) > 0 else 0.0,
            "confidence_by_sentiment": df.groupby("sentiment")["sentiment_score"].mean().to_dict(),
        }