"""Tests for checkpoint manager."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from raigem0n.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test checkpoint management functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a checkpoint manager instance."""
        return CheckpointManager(output_dir=temp_dir, enable_resume=True)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "text": ["message 1", "message 2", "message 3"],
            "id": [1, 2, 3]
        })

    def test_initialization(self, manager, temp_dir):
        """Test checkpoint manager initializes correctly."""
        assert manager.output_dir == temp_dir
        assert manager.checkpoint_dir == temp_dir / "checkpoints"
        assert manager.checkpoint_dir.exists()
        assert manager.enable_resume is True

    def test_pipeline_steps_defined(self, manager):
        """Test that pipeline steps are properly defined."""
        expected_steps = [
            "data_loading",
            "language_detection",
            "quote_detection",
            "entity_extraction",
            "sentiment_analysis",
            "toxicity_detection",
            "stance_classification",
            "style_extraction",
            "topic_classification",
            "link_extraction"
        ]
        assert manager.pipeline_steps == expected_steps

    def test_save_checkpoint(self, manager, sample_df):
        """Test saving a checkpoint."""
        step_name = "test_step"
        stats = {"count": 3, "processed": True}

        manager.save_checkpoint(
            step_name=step_name,
            dataframe=sample_df,
            stats=stats
        )

        # Check checkpoint file exists
        checkpoint_path = manager.get_checkpoint_path(step_name)
        assert checkpoint_path.exists()

        # Check stats file exists
        stats_path = manager.get_stats_path(step_name)
        assert stats_path.exists()

        # Verify stats content
        with open(stats_path, "r") as f:
            saved_stats = json.load(f)
        assert saved_stats["count"] == 3
        assert saved_stats["processed"] is True

    def test_load_checkpoint(self, manager, sample_df):
        """Test loading a checkpoint."""
        step_name = "test_step"

        # Save checkpoint first
        manager.save_checkpoint(step_name=step_name, dataframe=sample_df)

        # Load it back
        loaded_df, loaded_stats = manager.load_checkpoint(step_name)

        # Verify DataFrame
        assert loaded_df is not None
        assert len(loaded_df) == len(sample_df)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_checkpoint_exists(self, manager, sample_df):
        """Test checking if checkpoint exists."""
        step_name = "test_step"

        # Should not exist initially
        assert not manager.checkpoint_exists(step_name)

        # Save checkpoint
        manager.save_checkpoint(step_name=step_name, dataframe=sample_df)

        # Should exist now
        assert manager.checkpoint_exists(step_name)

    def test_get_memory_usage(self, manager):
        """Test memory usage tracking."""
        memory_info = manager.get_memory_usage()

        assert "rss_mb" in memory_info
        assert "vms_mb" in memory_info
        assert "percent" in memory_info
        assert "timestamp" in memory_info

        # Check values are reasonable
        assert memory_info["rss_mb"] > 0
        assert memory_info["percent"] > 0

    def test_log_memory_usage(self, manager):
        """Test logging memory usage for a step."""
        step_name = "test_step"

        manager.log_memory_usage(step_name, phase="start")

        # Check memory checkpoint was created
        key = f"{step_name}_start"
        assert key in manager.memory_checkpoints
        assert "rss_mb" in manager.memory_checkpoints[key]

    def test_resume_disabled(self, temp_dir):
        """Test checkpoint manager with resume disabled."""
        manager_no_resume = CheckpointManager(
            output_dir=temp_dir,
            enable_resume=False
        )

        assert manager_no_resume.enable_resume is False

    def test_empty_dataframe_checkpoint(self, manager):
        """Test saving checkpoint with empty DataFrame."""
        empty_df = pd.DataFrame()
        step_name = "empty_step"

        # Should not crash
        manager.save_checkpoint(step_name=step_name, dataframe=empty_df)

        # Should be able to load
        loaded_df, _ = manager.load_checkpoint(step_name)
        assert len(loaded_df) == 0
