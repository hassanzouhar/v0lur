"""Processing modules for the Telegram analysis pipeline."""

import importlib
from typing import Any, Dict

_PROCESSOR_MODULES: Dict[str, str] = {
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
    """Lazily import processor classes to avoid heavy startup dependencies."""

    if name in _PROCESSOR_MODULES:
        module = importlib.import_module(f".{_PROCESSOR_MODULES[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> Any:  # pragma: no cover - compatibility helper
    return sorted(list(globals().keys()) + list(_PROCESSOR_MODULES.keys()))


__all__ = list(_PROCESSOR_MODULES.keys())
