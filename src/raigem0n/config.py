"""Configuration management for the Telegram analysis pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the pipeline."""

    def __init__(self, config_path: Union[str, Path]) -> None:
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key."""
        keys = key.split(".")
        config_ref = self._config
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        config_ref[keys[-1]] = value

    # I/O Configuration
    @property
    def input_path(self) -> Path:
        """Input data file path."""
        return Path(self.get("io.input_path"))

    @property
    def output_path(self) -> Path:
        """Output file path."""
        return Path(self.get("io.out_path"))

    @property
    def input_format(self) -> str:
        """Input file format (json, jsonl, csv)."""
        return self.get("io.format", "json")

    @property
    def text_column(self) -> str:
        """Name of text column."""
        return self.get("io.text_col", "text")

    @property
    def id_column(self) -> str:
        """Name of ID column."""
        return self.get("io.id_col", "msg_id")

    @property
    def date_column(self) -> str:
        """Name of date column."""
        return self.get("io.date_col", "date")

    # Model Configuration
    @property
    def ner_model(self) -> str:
        """NER model name."""
        return self.get("models.ner", "dslim/bert-base-NER")

    @property
    def sentiment_model(self) -> str:
        """Sentiment analysis model name."""
        return self.get("models.sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest")

    @property
    def toxicity_model(self) -> str:
        """Toxicity detection model name."""
        return self.get("models.toxicity", "unitary/toxic-bert")

    @property
    def stance_model(self) -> str:
        """Stance classification model name."""
        return self.get("models.stance", "facebook/bart-large-mnli")

    @property
    def topic_model(self) -> str:
        """Topic classification model name."""
        return self.get("models.topic", "facebook/bart-large-mnli")

    # Processing Configuration
    @property
    def batch_size(self) -> int:
        """Processing batch size."""
        return self.get("processing.batch_size", 32)

    @property
    def prefer_gpu(self) -> bool:
        """Whether to prefer GPU processing."""
        return self.get("processing.prefer_gpu", False)

    @property
    def quote_aware(self) -> bool:
        """Whether to use quote-aware processing."""
        return self.get("processing.quote_aware", True)

    @property
    def skip_langdetect(self) -> bool:
        """Whether to skip language detection."""
        return self.get("processing.skip_langdetect", False)

    @property
    def max_entities_per_msg(self) -> int:
        """Maximum entities to extract per message."""
        return self.get("processing.max_entities_per_msg", 3)

    @property
    def stance_threshold(self) -> float:
        """Minimum confidence threshold for stance classification."""
        return self.get("processing.stance_threshold", 0.6)

    @property
    def topic_threshold(self) -> float:
        """Minimum confidence threshold for topic classification."""
        return self.get("processing.topic_threshold", 0.3)

    @property
    def max_text_length(self) -> int:
        """Maximum text length to process."""
        return self.get("processing.max_text_length", 8192)

    # Resource Configuration
    @property
    def aliases_path(self) -> Optional[Path]:
        """Path to entity aliases file."""
        aliases_path = self.get("resources.aliases_path")
        return Path(aliases_path) if aliases_path else None

    @property
    def topics_path(self) -> Optional[Path]:
        """Path to topic ontology file."""
        topics_path = self.get("resources.topics_path")
        return Path(topics_path) if topics_path else None

    # Logging Configuration
    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get("logging.level", "INFO")

    @property
    def redact_sensitive(self) -> bool:
        """Whether to redact sensitive information in logs."""
        return self.get("logging.redact_sensitive", True)

    def load_aliases(self) -> Dict[str, Dict[str, Any]]:
        """Load entity aliases from JSON file."""
        if not self.aliases_path or not self.aliases_path.exists():
            return {}

        try:
            with open(self.aliases_path, "r", encoding="utf-8") as f:
                aliases = json.load(f)
            logger.info(f"Loaded {len(aliases)} entity aliases")
            return aliases
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load aliases file: {e}")
            return {}

    def load_topics(self) -> List[Dict[str, Any]]:
        """Load topic ontology from JSON file."""
        if not self.topics_path or not self.topics_path.exists():
            return []

        try:
            with open(self.topics_path, "r", encoding="utf-8") as f:
                topics = json.load(f)
            logger.info(f"Loaded {len(topics)} topic definitions")
            return topics
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load topics file: {e}")
            return []

    def save_snapshot(self, output_dir: Path) -> None:
        """Save configuration snapshot for reproducibility."""
        snapshot_path = output_dir / "config_snapshot.yaml"
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=True)
            logger.info(f"Saved config snapshot to {snapshot_path}")
        except Exception as e:
            logger.warning(f"Could not save config snapshot: {e}")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path}, batch_size={self.batch_size})"