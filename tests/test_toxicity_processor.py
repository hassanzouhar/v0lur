"""Tests for toxicity processor."""

import pandas as pd
import pytest

from raigem0n.processors.toxicity_processor import ToxicityProcessor


class TestToxicityProcessor:
    """Test toxicity detection functionality."""

    @pytest.fixture
    def processor(self):
        """Create a toxicity processor instance."""
        return ToxicityProcessor(
            model_name="unitary/toxic-bert",
            device="cpu",
            threshold=0.5
        )

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "text": [
                "Hello, how are you today?",
                "This is a normal message.",
                "Great work everyone!",
                "",  # Empty text
            ],
            "id": [1, 2, 3, 4]
        })

    def test_processor_initialization(self, processor):
        """Test that processor initializes correctly."""
        assert processor is not None
        assert processor.device == "cpu"
        assert processor.threshold == 0.5
        assert processor.model_name == "unitary/toxic-bert"

    def test_process_dataframe(self, processor, sample_df):
        """Test processing a DataFrame adds toxicity columns."""
        result_df = processor.process_dataframe(sample_df)

        # Check toxicity columns are added
        assert "is_toxic" in result_df.columns
        assert "toxicity_score" in result_df.columns

        # Check all rows have toxicity data
        assert result_df["is_toxic"].notna().all()
        assert result_df["toxicity_score"].notna().all()

    def test_toxicity_values(self, processor, sample_df):
        """Test toxicity values are valid."""
        result_df = processor.process_dataframe(sample_df)

        # Check is_toxic is boolean
        assert result_df["is_toxic"].dtype == bool

        # Check scores are in valid range [0, 1]
        assert (result_df["toxicity_score"] >= 0).all()
        assert (result_df["toxicity_score"] <= 1).all()

    def test_empty_dataframe(self, processor):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame({"text": [], "id": []})
        result_df = processor.process_dataframe(empty_df)

        assert len(result_df) == 0
        assert "is_toxic" in result_df.columns
        assert "toxicity_score" in result_df.columns

    def test_get_toxicity_stats(self, processor, sample_df):
        """Test toxicity statistics generation."""
        result_df = processor.process_dataframe(sample_df)
        stats = processor.get_toxicity_stats(result_df)

        assert "total_messages" in stats
        assert "toxic_count" in stats
        assert "toxic_percentage" in stats
        assert stats["total_messages"] == len(sample_df)

        # Check counts are non-negative
        assert stats["toxic_count"] >= 0
        assert stats["toxic_percentage"] >= 0

    def test_threshold_parameter(self):
        """Test that threshold parameter works correctly."""
        processor_low = ToxicityProcessor(
            model_name="unitary/toxic-bert",
            device="cpu",
            threshold=0.3
        )
        processor_high = ToxicityProcessor(
            model_name="unitary/toxic-bert",
            device="cpu",
            threshold=0.7
        )

        assert processor_low.threshold == 0.3
        assert processor_high.threshold == 0.7
