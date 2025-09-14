"""Data loading and normalization for Telegram channel analysis."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


logger = logging.getLogger(__name__)


class DataLoader:
    """Load and normalize Telegram channel data."""

    def __init__(self, max_text_length: int = 8192) -> None:
        """Initialize data loader."""
        self.max_text_length = max_text_length

    def load_data(
        self,
        input_path: Union[str, Path],
        format_type: str = "json",
        text_col: str = "text",
        id_col: str = "msg_id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Load data from file and normalize format."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading data from {input_path} (format: {format_type})")

        # Load data based on format
        if format_type.lower() == "json":
            df = self._load_json(input_path)
        elif format_type.lower() == "jsonl":
            df = self._load_jsonl(input_path)
        elif format_type.lower() == "csv":
            df = self._load_csv(input_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Normalize column names
        df = self._normalize_columns(df, text_col, id_col, date_col)

        # Normalize data types and content
        df = self._normalize_content(df)

        logger.info(f"Loaded {len(df)} messages")
        return df

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Handle nested JSON structures
            if "messages" in data:
                return pd.DataFrame(data["messages"])
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError("JSON must contain a list or dict with messages")

    def _load_jsonl(self, path: Path) -> pd.DataFrame:
        """Load data from JSONL file."""
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading JSONL file: {e}")

        if not records:
            raise ValueError("No valid records found in JSONL file")

        return pd.DataFrame(records)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

    def _normalize_columns(
        self, df: pd.DataFrame, text_col: str, id_col: str, date_col: str
    ) -> pd.DataFrame:
        """Normalize column names to standard format."""
        # Required columns mapping
        column_mapping = {}

        # Map text column
        if text_col in df.columns:
            column_mapping[text_col] = "text"
        elif "message" in df.columns:
            column_mapping["message"] = "text"
        elif "content" in df.columns:
            column_mapping["content"] = "text"
        else:
            raise ValueError(f"Text column '{text_col}' not found in data")

        # Map ID column
        if id_col in df.columns:
            column_mapping[id_col] = "msg_id"
        elif "id" in df.columns:
            column_mapping["id"] = "msg_id"
        elif "message_id" in df.columns:
            column_mapping["message_id"] = "msg_id"
        else:
            raise ValueError(f"ID column '{id_col}' not found in data")

        # Map date column
        if date_col in df.columns:
            column_mapping[date_col] = "date"
        elif "timestamp" in df.columns:
            column_mapping["timestamp"] = "date"
        elif "created_at" in df.columns:
            column_mapping["created_at"] = "date"
        else:
            raise ValueError(f"Date column '{date_col}' not found in data")

        # Optional columns
        optional_columns = {
            "chat_id": "chat_id",
            "channel_id": "chat_id",
            "sender_id": "chat_id",
            "media_type": "media_type",
            "media_url": "media_url",
            "forwarded_from": "forwarded_from",
            "forward_from": "forwarded_from",
        }

        for old_col, new_col in optional_columns.items():
            if old_col in df.columns and new_col not in column_mapping.values():
                column_mapping[old_col] = new_col

        # Apply column mapping
        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ["msg_id", "text", "date"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing after normalization")

        return df

    def _normalize_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize content and data types."""
        df = df.copy()

        # Convert msg_id to string
        df["msg_id"] = df["msg_id"].astype(str)

        # Normalize text content
        df["text"] = df["text"].fillna("").astype(str)
        df["text"] = df["text"].apply(self._normalize_text)

        # Normalize dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            logger.warning("Some dates could not be parsed and will be dropped")
            df = df.dropna(subset=["date"])

        # Fill optional columns with defaults
        if "chat_id" not in df.columns:
            df["chat_id"] = "unknown"
        df["chat_id"] = df["chat_id"].fillna("unknown").astype(str)

        if "media_type" not in df.columns:
            df["media_type"] = None
        if "media_url" not in df.columns:
            df["media_url"] = None
        if "forwarded_from" not in df.columns:
            df["forwarded_from"] = None

        # Remove duplicates based on msg_id
        initial_count = len(df)
        df = df.drop_duplicates(subset=["msg_id"], keep="first")
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate messages")

        # Filter out empty texts
        df = df[df["text"].str.strip() != ""]

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Normalized data: {len(df)} messages after filtering")
        return df

    def _normalize_text(self, text: str) -> str:
        """Normalize text content."""
        if not isinstance(text, str):
            text = str(text)

        # Handle Telegram span objects (convert to plain text)
        if text.startswith("[{") and "text" in text:
            try:
                # Simple heuristic to extract text from span objects
                import re
                text_matches = re.findall(r'"text":\s*"([^"]*)"', text)
                if text_matches:
                    text = " ".join(text_matches)
            except Exception:
                # If parsing fails, keep original text
                pass

        # Basic text cleanup
        text = text.replace("\\n", "\n").replace("\\t", "\t")
        text = text.strip()

        # Truncate if too long
        if len(text) > self.max_text_length:
            logger.debug(f"Truncating text from {len(text)} to {self.max_text_length} chars")
            text = text[: self.max_text_length]

        return text

    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Save processed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".parquet":
            df.to_parquet(output_path, index=False)
        elif output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == ".json":
            df.to_json(output_path, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        logger.info(f"Saved {len(df)} messages to {output_path}")

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        return {
            "total_messages": len(df),
            "unique_channels": df["chat_id"].nunique(),
            "date_range": {
                "start": df["date"].min().isoformat(),
                "end": df["date"].max().isoformat(),
            },
            "avg_text_length": df["text"].str.len().mean(),
            "has_forwarded": df["forwarded_from"].notna().sum(),
            "has_media": df["media_type"].notna().sum(),
        }