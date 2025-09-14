"""Tests for configuration management."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from raigem0n.config import Config


class TestConfig:
    """Test configuration loading and management."""

    def test_load_valid_config(self):
        """Test loading a valid configuration."""
        config_data = {
            "io": {
                "input_path": "data/test.json",
                "format": "json",
                "out_path": "out/test.parquet",
            },
            "models": {
                "ner": "test-model",
            },
            "processing": {
                "batch_size": 16,
                "prefer_gpu": False,
            },
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config(config_path)
            
            assert config.input_path == Path("data/test.json")
            assert config.input_format == "json"
            assert config.batch_size == 16
            assert config.prefer_gpu is False
            assert config.ner_model == "test-model"
            
        finally:
            Path(config_path).unlink()

    def test_config_defaults(self):
        """Test configuration defaults."""
        config_data = {
            "io": {
                "input_path": "data/test.json",
            }
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config(config_path)
            
            # Test defaults
            assert config.batch_size == 32  # default
            assert config.prefer_gpu is False  # default
            assert config.max_entities_per_msg == 3  # default
            assert config.stance_threshold == 0.6  # default
            
        finally:
            Path(config_path).unlink()

    def test_config_get_set(self):
        """Test getting and setting configuration values."""
        config_data = {"test": {"nested": {"value": 42}}}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config(config_path)
            
            assert config.get("test.nested.value") == 42
            assert config.get("nonexistent", "default") == "default"
            
            config.set("new.setting", "test_value")
            assert config.get("new.setting") == "test_value"
            
        finally:
            Path(config_path).unlink()

    def test_load_aliases(self):
        """Test loading entity aliases."""
        config_data = {
            "io": {"input_path": "test.json"},
            "resources": {"aliases_path": "aliases.json"},
        }
        
        aliases_data = {
            "Test Person": {
                "aliases": ["Test", "TP"],
                "type": "PERSON",
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create config file
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            
            # Create aliases file
            aliases_path = tmpdir / "aliases.json"
            with open(aliases_path, "w") as f:
                json.dump(aliases_data, f)
            
            config = Config(config_path)
            aliases = config.load_aliases()
            
            assert len(aliases) == 1
            assert "Test Person" in aliases
            assert aliases["Test Person"]["type"] == "PERSON"

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            Config("nonexistent.yaml")