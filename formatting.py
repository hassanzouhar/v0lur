#!/usr/bin/env python3
"""
Formatting utilities for the Textual UI.

This module provides consistent number formatting, date formatting,
and color theme functions for displaying analysis results.
"""

from datetime import datetime
from typing import Optional, Union


def fmt_int(n: Union[int, float]) -> str:
    """Format integer with thousand separators."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return "0"


def fmt_float(x: Union[int, float], places: int = 2) -> str:
    """Format float with specified decimal places."""
    try:
        return f"{float(x):.{places}f}"
    except (ValueError, TypeError):
        return "0.00"


def fmt_pct(x: Union[int, float], places: int = 1) -> str:
    """Format percentage with specified decimal places."""
    try:
        return f"{float(x):.{places}f}%"
    except (ValueError, TypeError):
        return "0.0%"


def fmt_date(date_str: Union[str, datetime], short: bool = False) -> str:
    """
    Format date string for display.
    
    Args:
        date_str: Date string in various formats or datetime object
        short: If True, return MM/DD format; if False, return full date
    
    Returns:
        Formatted date string
    """
    if isinstance(date_str, datetime):
        if short:
            return date_str.strftime("%m/%d")
        return date_str.strftime("%Y-%m-%d")
    
    if not date_str or not isinstance(date_str, str):
        return "Unknown"
    
    # Try to parse common date formats
    date_formats = [
        "%Y-%m-%d",           # 2019-06-25
        "%Y-%m-%dT%H:%M:%S",  # 2019-06-25T03:45:14
        "%m/%d/%Y",           # 06/25/2019
        "%d/%m/%Y",           # 25/06/2019
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str.split('T')[0], fmt.split('T')[0])
            if short:
                return dt.strftime("%m/%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # If parsing fails, return original string truncated
    return date_str[:10] if len(date_str) > 10 else date_str


def sentiment_color(score: Union[int, float]) -> str:
    """
    Get color name for sentiment score.
    
    Args:
        score: Sentiment score (typically -1 to 1)
    
    Returns:
        Color name compatible with Rich/Textual
    """
    try:
        x = float(score)
        if x >= 0.3:
            return "green"
        elif x >= 0.1:
            return "bright_green"
        elif x >= -0.1:
            return "yellow"
        elif x >= -0.3:
            return "bright_yellow"
        else:
            return "red"
    except (ValueError, TypeError):
        return "white"


def toxicity_color(score: Union[int, float]) -> str:
    """
    Get color name for toxicity score.
    
    Args:
        score: Toxicity score (typically 0 to 1)
    
    Returns:
        Color name compatible with Rich/Textual
    """
    try:
        x = float(score)
        if x >= 0.8:
            return "bright_red"
        elif x >= 0.6:
            return "red"
        elif x >= 0.4:
            return "yellow"
        elif x >= 0.2:
            return "bright_green"
        else:
            return "green"
    except (ValueError, TypeError):
        return "white"


def confidence_color(score: Union[int, float]) -> str:
    """
    Get color name for confidence score.
    
    Args:
        score: Confidence score (typically 0 to 1)
    
    Returns:
        Color name compatible with Rich/Textual
    """
    try:
        x = float(score)
        if x >= 0.8:
            return "green"
        elif x >= 0.6:
            return "bright_green"
        elif x >= 0.4:
            return "yellow"
        elif x >= 0.2:
            return "bright_yellow"
        else:
            return "red"
    except (ValueError, TypeError):
        return "white"


def count_color(count: Union[int, float], max_count: Union[int, float] = None) -> str:
    """
    Get color name for count values (relative to maximum).
    
    Args:
        count: Current count
        max_count: Maximum count for relative scaling
    
    Returns:
        Color name compatible with Rich/Textual
    """
    try:
        c = float(count)
        if max_count is not None and max_count > 0:
            ratio = c / float(max_count)
            if ratio >= 0.8:
                return "bright_blue"
            elif ratio >= 0.6:
                return "blue"
            elif ratio >= 0.4:
                return "cyan"
            elif ratio >= 0.2:
                return "bright_cyan"
            else:
                return "white"
        else:
            # Absolute thresholds if no max provided
            if c >= 100:
                return "bright_blue"
            elif c >= 50:
                return "blue"
            elif c >= 20:
                return "cyan"
            elif c >= 5:
                return "bright_cyan"
            else:
                return "white"
    except (ValueError, TypeError):
        return "white"


def truncate_text(text: str, max_length: int = 50, ellipsis: str = "...") -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including ellipsis
        ellipsis: String to append when truncating
    
    Returns:
        Truncated text
    """
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(ellipsis)] + ellipsis


def format_badge_text(available_files: dict) -> str:
    """
    Format badge text showing which analysis files are available.
    
    Args:
        available_files: Dict mapping filename to availability boolean
    
    Returns:
        Badge string like "STEM" (S=Summary, T=Topics, E=Entities, M=Toxic Messages, Y=Style)
    """
    badge_mapping = {
        "channel_daily_summary.csv": "S",
        "channel_topic_analysis.json": "T", 
        "channel_entity_counts.csv": "E",
        "channel_top_toxic_messages.csv": "M",
        "channel_style_features.json": "Y"
    }
    
    badge = ""
    for filename, available in available_files.items():
        if filename in badge_mapping:
            if available:
                badge += badge_mapping[filename]
            else:
                badge += "-"
    
    return badge if badge else "-----"


def format_run_timestamp(dt: datetime) -> str:
    """
    Format run timestamp for display in run list.
    
    Args:
        dt: Datetime object
    
    Returns:
        Formatted timestamp string
    """
    try:
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 7:
            return dt.strftime("%Y-%m-%d")
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "Just now"
    except (ValueError, TypeError):
        return "Unknown"


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if division by zero
    
    Returns:
        Division result or default
    """
    try:
        n = float(numerator)
        d = float(denominator)
        if d == 0:
            return default
        return n / d
    except (ValueError, TypeError):
        return default