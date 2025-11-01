"""Toxicity detection processor using BERT-based models."""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)


class ToxicityProcessor:
    """Toxicity detection using Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        toxicity_threshold: float = 0.5,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize toxicity processor.
        
        Args:
            model_name: Hugging Face model name for toxicity detection
            toxicity_threshold: Threshold above which content is considered toxic
            device: Device to run model on (cuda, mps, cpu)
        """
        self.model_name = model_name
        self.toxicity_threshold = threshold if threshold is not None else toxicity_threshold
        self.threshold = self.toxicity_threshold  # Backwards compatibility
        self.device = device
        self.toxicity_pipeline = None
        self._offline_mode = False
        self._load_model()

    def _load_model(self) -> None:
        """Load toxicity detection model."""
        if pipeline is None:
            self.toxicity_pipeline = None
            self._offline_mode = True
            logger.warning("Transformers not available; using heuristic toxicity detection")
            return

        try:
            logger.info(f"Loading toxicity model: {self.model_name}")
            self.toxicity_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
            )
            logger.info("Toxicity model loaded successfully")
        except Exception as e:
            # When the model cannot be downloaded (e.g. offline tests) we
            # gracefully fall back to a keyword-based heuristic.
            self.toxicity_pipeline = None
            self._offline_mode = True
            logger.warning("Falling back to heuristic toxicity detection: %s", e)

    def analyze_toxicity(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze toxicity for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of toxicity analysis results with scores and classifications
        """
        if not texts:
            return []

        if self.toxicity_pipeline and not self._offline_mode:
            logger.debug(f"Analyzing toxicity for {len(texts)} texts with transformer pipeline")
            results: List[Dict[str, Any]] = []

            for text in texts:
                try:
                    raw_results = self.toxicity_pipeline(text)
                    toxicity_result = self._process_toxicity_result(raw_results[0])
                    results.append(toxicity_result)
                except Exception as e:
                    logger.warning(f"Failed to analyze toxicity for text: {e}")
                    results.append(self._heuristic_toxicity(text))

            return results

        logger.debug("Analyzing toxicity with heuristic fallback")
        return [self._heuristic_toxicity(text) for text in texts]

    def _process_toxicity_result(self, raw_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw toxicity analysis result into standard format.
        
        Args:
            raw_result: Raw result from toxicity pipeline
            
        Returns:
            Processed toxicity result
        """
        # Different models may have different label formats
        toxic_score = 0.0
        
        # Handle different toxicity model output formats
        if isinstance(raw_result, list):
            # Find toxic/TOXIC label
            for prediction in raw_result:
                label = prediction["label"].upper()
                if label in ["TOXIC", "TOXICITY", "1"]:
                    toxic_score = float(prediction["score"])
                    break
        else:
            # Single prediction case
            if raw_result["label"].upper() in ["TOXIC", "TOXICITY", "1"]:
                toxic_score = float(raw_result["score"])
        
        # Create all scores dictionary
        all_scores = {}
        if isinstance(raw_result, list):
            for prediction in raw_result:
                label_normalized = self._normalize_toxicity_label(prediction["label"])
                all_scores[label_normalized] = float(prediction["score"])
        else:
            label_normalized = self._normalize_toxicity_label(raw_result["label"])
            all_scores[label_normalized] = float(raw_result["score"])
        
        return {
            "toxicity_score": toxic_score,
            "is_toxic": toxic_score >= self.toxicity_threshold,
            "all_scores": all_scores
        }

    def _heuristic_toxicity(self, text: str) -> Dict[str, Any]:
        """Offline fallback toxicity detection using keyword heuristics."""

        toxic_keywords = {
            "hate", "stupid", "idiot", "dumb", "terrible",
            "horrible", "worst", "kill", "trash", "awful"
        }

        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return {
                "toxicity_score": 0.0,
                "is_toxic": False,
                "all_scores": {"toxic": 0.0, "non_toxic": 1.0},
            }

        toxic_hits = sum(1 for token in tokens if token in toxic_keywords)
        score = min(1.0, toxic_hits / max(len(tokens), 1))
        is_toxic = score >= self.toxicity_threshold

        return {
            "toxicity_score": float(score),
            "is_toxic": bool(is_toxic),
            "all_scores": {"toxic": float(score), "non_toxic": float(1 - score)},
        }

    def _normalize_toxicity_label(self, label: str) -> str:
        """Normalize toxicity labels to standard format."""
        label_upper = label.upper()
        
        # Map various label formats to standard names
        if label_upper in ["TOXIC", "TOXICITY", "1", "LABEL_1"]:
            return "toxic"
        elif label_upper in ["NON_TOXIC", "NON-TOXIC", "NOT_TOXIC", "0", "LABEL_0"]:
            return "non_toxic"
        else:
            return label.lower()

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add toxicity information.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added toxicity columns
        """
        logger.info(f"Processing {len(df)} messages for toxicity detection")
        
        # Analyze toxicity in batches
        batch_size = 50  # Reasonable batch size for memory management

        if df.empty:
            df = df.copy()
            df["toxicity_score"] = []
            df["is_toxic"] = []
            df["toxicity_all_scores"] = []
            return df

        all_results: List[Dict[str, Any]] = []

        for i in range(0, len(df), batch_size):
            batch_texts = df[text_column].iloc[i:i + batch_size].tolist()
            batch_results = self.analyze_toxicity(batch_texts)
            all_results.extend(batch_results)

            if (i + batch_size) % 200 == 0 and i != 0:
                logger.info(f"Processed {min(i + batch_size, len(df))} messages")

        # Add results to dataframe
        df = df.copy()

        df["toxicity_score"] = [result["toxicity_score"] for result in all_results]
        df["is_toxic"] = [bool(result["is_toxic"]) for result in all_results]
        df["toxicity_all_scores"] = [result.get("all_scores", {}) for result in all_results]
        
        # Log summary statistics
        toxic_count = df["is_toxic"].sum()
        avg_toxicity = df["toxicity_score"].mean()
        max_toxicity = df["toxicity_score"].max()
        
        logger.info(f"Toxicity analysis completed.")
        logger.info(f"Toxic messages: {toxic_count}/{len(df)} ({toxic_count/len(df)*100:.1f}%)")
        logger.info(f"Average toxicity score: {avg_toxicity:.3f}")
        logger.info(f"Maximum toxicity score: {max_toxicity:.3f}")
        
        return df

    def get_toxicity_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about toxicity analysis results.
        
        Args:
            df: DataFrame with toxicity analysis results
            
        Returns:
            Dictionary with toxicity statistics
        """
        if "toxicity_score" not in df.columns:
            return {}

        toxic_count = int(df["is_toxic"].sum()) if "is_toxic" in df.columns else 0

        return {
            "total_messages": len(df),
            "toxic_messages": int(toxic_count),
            "toxic_count": int(toxic_count),
            "toxic_percentage": float(toxic_count / len(df) * 100) if len(df) > 0 else 0.0,
            "avg_toxicity_score": df["toxicity_score"].mean() if len(df) > 0 else 0.0,
            "max_toxicity_score": df["toxicity_score"].max() if len(df) > 0 else 0.0,
            "toxicity_threshold": self.toxicity_threshold,
        }

    def get_most_toxic_messages(self, df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Get the most toxic messages from the dataset.
        
        Args:
            df: DataFrame with toxicity analysis results
            top_k: Number of top toxic messages to return
            
        Returns:
            DataFrame with most toxic messages
        """
        if "toxicity_score" not in df.columns:
            return pd.DataFrame()

        # Sort by toxicity score and get top K
        toxic_messages = df.nlargest(top_k, "toxicity_score")
        
        # Select relevant columns for output
        output_columns = ["msg_id", "date", "toxicity_score", "text"]
        if "is_toxic" in df.columns:
            output_columns.insert(-1, "is_toxic")
        
        # Filter to only include available columns
        available_columns = [col for col in output_columns if col in toxic_messages.columns]
        
        return toxic_messages[available_columns].copy()