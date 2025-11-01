"""Tests for sentiment processor."""

import pandas as pd
import pytest

from raigem0n.processors.sentiment_processor import SentimentProcessor


class TestSentimentProcessor:
    """Test sentiment analysis functionality."""

    @pytest.fixture
    def processor(self):
        """Create a sentiment processor instance."""
        return SentimentProcessor(
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device="cpu"
        )

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "text": [
                "I love this! It's absolutely wonderful!",
                "This is terrible and disappointing.",
                "The weather is okay, nothing special.",
                "",  # Empty text
            ],
            "id": [1, 2, 3, 4]
        })

    def test_processor_initialization(self, processor):
        """Test that processor initializes correctly."""
        assert processor is not None
        assert processor.device == "cpu"
        assert processor.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def test_process_dataframe(self, processor, sample_df):
        """Test processing a DataFrame adds sentiment columns."""
        result_df = processor.process_dataframe(sample_df)

        # Check sentiment columns are added
        assert "sentiment" in result_df.columns
        assert "sentiment_score" in result_df.columns

        # Check all rows have sentiment
        assert result_df["sentiment"].notna().all()
        assert result_df["sentiment_score"].notna().all()

    def test_sentiment_values(self, processor, sample_df):
        """Test sentiment labels are valid."""
        result_df = processor.process_dataframe(sample_df)

        # Check sentiment values are valid
        valid_sentiments = {"positive", "negative", "neutral"}
        assert set(result_df["sentiment"].unique()).issubset(valid_sentiments)

        # Check scores are in valid range [0, 1]
        assert (result_df["sentiment_score"] >= 0).all()
        assert (result_df["sentiment_score"] <= 1).all()

    def test_empty_dataframe(self, processor):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame({"text": [], "id": []})
        result_df = processor.process_dataframe(empty_df)

        assert len(result_df) == 0
        assert "sentiment" in result_df.columns
        assert "sentiment_score" in result_df.columns

    def test_get_sentiment_stats(self, processor, sample_df):
        """Test sentiment statistics generation."""
        result_df = processor.process_dataframe(sample_df)
        stats = processor.get_sentiment_stats(result_df)

        assert "total_messages" in stats
        assert "sentiment_distribution" in stats
        assert stats["total_messages"] == len(sample_df)

        # Check sentiment distribution
        dist = stats["sentiment_distribution"]
        assert "positive" in dist
        assert "negative" in dist
        assert "neutral" in dist

        # Check percentages sum to ~100%
        total_pct = sum(dist.values())
        assert 99 <= total_pct <= 101  # Allow small rounding errors
