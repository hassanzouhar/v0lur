"""
Raigem0n: Telegram Stance & Language Analysis Pipeline

A neutral, reproducible analytics pipeline for analyzing public Telegram channels
to extract stance classification, topic analysis, and linguistic patterns while
maintaining attribution accuracy.
"""

__version__ = "1.2.0"
__author__ = "raigem0n"
__description__ = "Telegram Stance & Language Analysis Pipeline"

from .config import Config
from .data_loader import DataLoader
from .processors import (
    NERProcessor,
)

__all__ = [
    "Config",
    "DataLoader",
    "NERProcessor",
]
