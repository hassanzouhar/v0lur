"""
Raigem0n: Telegram Stance & Language Analysis Pipeline

A neutral, reproducible analytics pipeline for analyzing public Telegram channels
to extract stance classification, topic analysis, and linguistic patterns while
maintaining attribution accuracy.
"""

import importlib
from typing import Any

__version__ = "1.2.0"
__author__ = "raigem0n"
__description__ = "Telegram Stance & Language Analysis Pipeline"

from .config import Config
from .data_loader import DataLoader

_PROCESSOR_MODULES = {
    "NERProcessor": "ner_processor",
    "SentimentProcessor": "sentiment_processor",
    "ToxicityProcessor": "toxicity_processor",
    "StanceProcessor": "stance_processor",
    "QuoteProcessor": "quote_processor",
    "StyleProcessor": "style_processor",
    "TopicProcessor": "topic_processor",
    "DiscoveryTopicProcessor": "discovery_topic_processor",
    "LinksProcessor": "links_processor",
}


def __getattr__(name: str) -> Any:
    """Lazily import processor classes on demand.

    Importing heavy transformer-based modules at package import time makes
    simple utilities like configuration loading fail in minimal test
    environments.  This lazy loader keeps backwards compatibility with the
    public API while only importing processor modules when explicitly
    requested.
    """

    if name in _PROCESSOR_MODULES:
        module = importlib.import_module(
            f".processors.{_PROCESSOR_MODULES[name]}",
            __name__,
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["Config", "DataLoader", *_PROCESSOR_MODULES.keys()]
