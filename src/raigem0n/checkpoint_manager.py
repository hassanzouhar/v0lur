"""
Pipeline checkpoint manager for saving and resuming processing state.

This module provides memory-efficient checkpointing to prevent data loss
during pipeline failures and enable resuming from failed steps.
"""

import json
import logging
import pickle
import gc
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage pipeline checkpoints for fault tolerance and memory efficiency."""
    
    def __init__(self, output_dir: Path, enable_resume: bool = True):
        """Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to store checkpoints and outputs
            enable_resume: Whether to enable resuming from checkpoints
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.enable_resume = enable_resume
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline step definitions
        self.pipeline_steps = [
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
        
        # Track completed steps
        self.completed_steps = set()
        self.current_step = None

        # Memory tracking
        self.process = psutil.Process()
        self.memory_checkpoints = {}

        # Track file formats so we can gracefully fallback when parquet
        # dependencies aren't available.  We default to parquet but store
        # per-step overrides when we need to use an alternative format.
        self.default_checkpoint_format = "parquet"
        self._checkpoint_formats: Dict[str, str] = {}
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        
    def get_checkpoint_path(self, step_name: str, file_type: Optional[str] = None) -> Path:
        """Get checkpoint file path for a step.

        Args:
            step_name: Name of the pipeline step.
            file_type: Optional explicit file type.  When omitted we use the
                last known format for this step or the current default.
        """
        checkpoint_format = file_type or self._checkpoint_formats.get(
            step_name,
            self.default_checkpoint_format,
        )
        return self.checkpoint_dir / f"{step_name}_checkpoint.{checkpoint_format}"
    
    def get_stats_path(self, step_name: str) -> Path:
        """Get stats file path for a step.""" 
        return self.checkpoint_dir / f"{step_name}_stats.json"
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            memory_info = self.process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024  # Resident Set Size
            vms_mb = memory_info.vms / 1024 / 1024  # Virtual Memory Size
            percent = max(self.process.memory_percent(), 0.001)
            return {
                "rss_mb": rss_mb,
                "vms_mb": vms_mb,
                "percent": percent,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def log_memory_usage(self, step_name: str, phase: str = "start") -> None:
        """Log memory usage for a processing step."""
        memory_info = self.get_memory_usage()
        self.memory_checkpoints[f"{step_name}_{phase}"] = memory_info
        
        if "error" not in memory_info:
            logger.info(f"Memory usage [{step_name}_{phase}]: "
                       f"{memory_info['rss_mb']:.1f}MB RSS, "
                       f"{memory_info['percent']:.1f}% of system")
    
    def save_checkpoint(self, 
                       step_name: str, 
                       dataframe: pd.DataFrame, 
                       stats: Dict[str, Any] = None,
                       additional_data: Dict[str, Any] = None) -> None:
        """Save checkpoint data after a processing step.
        
        Args:
            step_name: Name of the processing step
            dataframe: DataFrame with results from this step
            stats: Statistics from the processing step
            additional_data: Any additional data to save
        """
        try:
            self.log_memory_usage(step_name, "before_save")
            
            # Save main dataframe
            checkpoint_path = self.get_checkpoint_path(step_name, "parquet")
            dataframe.to_parquet(checkpoint_path, index=False, compression="snappy")
            self._checkpoint_formats[step_name] = "parquet"

        except (ImportError, ValueError, AttributeError, OSError) as parquet_error:
            logger.info(
                "Parquet support unavailable (%s). Falling back to pickle for %s",
                parquet_error,
                step_name,
            )
            checkpoint_path = self.get_checkpoint_path(step_name, "pkl")
            dataframe.to_pickle(checkpoint_path)
            self._checkpoint_formats[step_name] = "pkl"
            if self.default_checkpoint_format == "parquet":
                # Future checkpoints should use a supported default.
                self.default_checkpoint_format = "pkl"

            # Save statistics
            if stats:
                stats_path = self.get_stats_path(step_name)
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
            
            # Save additional data if provided
            if additional_data:
                for name, data in additional_data.items():
                    if isinstance(data, pd.DataFrame):
                        data_path = self.checkpoint_dir / f"{step_name}_{name}.parquet"
                        data.to_parquet(data_path, index=False, compression="snappy")
                    elif isinstance(data, (dict, list)):
                        data_path = self.checkpoint_dir / f"{step_name}_{name}.json"
                        with open(data_path, 'w') as f:
                            json.dump(data, f, indent=2, default=str)
                    else:
                        # Use pickle for other types
                        data_path = self.checkpoint_dir / f"{step_name}_{name}.pkl"
                        with open(data_path, 'wb') as f:
                            pickle.dump(data, f)
            
            # Mark step as completed
            self.completed_steps.add(step_name)
            
            # Save completion status
            status_path = self.checkpoint_dir / "pipeline_status.json"
            status = {
                "completed_steps": list(self.completed_steps),
                "last_updated": datetime.now().isoformat(),
                "memory_checkpoints": self.memory_checkpoints,
                "checkpoint_formats": self._checkpoint_formats,
            }
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
            
            self.log_memory_usage(step_name, "after_save")
            logger.info(f"Checkpoint saved for step: {step_name}")
            logger.info(f"Checkpoint location: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {step_name}: {e}")
            raise
    
    def load_checkpoint(self, step_name: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Load checkpoint data for a processing step.
        
        Args:
            step_name: Name of the processing step
            
        Returns:
            Tuple of (dataframe, stats) or (None, None) if not found
        """
        try:
            checkpoint_path = self.get_checkpoint_path(step_name)
            stats_path = self.get_stats_path(step_name)

            if not checkpoint_path.exists():
                # Try known alternative formats
                fallback_formats = ["parquet", "pkl", "pickle", "csv"]
                for fmt in fallback_formats:
                    alt_path = self.get_checkpoint_path(step_name, fmt)
                    if alt_path.exists():
                        checkpoint_path = alt_path
                        self._checkpoint_formats[step_name] = fmt
                        break
                else:
                    return None, None

            # Load dataframe
            suffix = checkpoint_path.suffix.lstrip('.')
            if suffix == "parquet":
                df = pd.read_parquet(checkpoint_path)
            elif suffix in {"pkl", "pickle"}:
                df = pd.read_pickle(checkpoint_path)
            elif suffix == "csv":
                df = pd.read_csv(checkpoint_path)
            else:
                raise ValueError(f"Unsupported checkpoint format: {suffix}")
            
            # Load stats if available
            stats = None
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
            
            logger.info(f"Checkpoint loaded for step: {step_name}")
            logger.info(f"Loaded {len(df)} rows from checkpoint")

            return df, stats

        except Exception as e:
            logger.error(f"Failed to load checkpoint for {step_name}: {e}")
            return None, None

    def checkpoint_exists(self, step_name: str) -> bool:
        """Check whether a checkpoint file exists for the specified step."""

        checkpoint_path = self.get_checkpoint_path(step_name)
        if checkpoint_path.exists():
            return True

        for fmt in ("parquet", "pkl", "pickle", "csv"):
            alt_path = self.get_checkpoint_path(step_name, fmt)
            if alt_path.exists():
                self._checkpoint_formats[step_name] = fmt
                return True

        return False

    def get_resume_point(self) -> Optional[str]:
        """Determine which step to resume from.
        
        Returns:
            Name of the step to resume from, or None to start from beginning
        """
        if not self.enable_resume:
            return None
        
        status_path = self.checkpoint_dir / "pipeline_status.json"
        if not status_path.exists():
            return None
        
        try:
            with open(status_path, 'r') as f:
                status = json.load(f)
            
            completed = set(status.get("completed_steps", []))
            self.completed_steps = completed
            self.memory_checkpoints = status.get("memory_checkpoints", {})
            self._checkpoint_formats.update(status.get("checkpoint_formats", {}))
            
            # Find the last completed step
            for step in reversed(self.pipeline_steps):
                if step in completed:
                    next_idx = self.pipeline_steps.index(step) + 1
                    if next_idx < len(self.pipeline_steps):
                        resume_step = self.pipeline_steps[next_idx]
                        logger.info(f"Resuming from step: {resume_step}")
                        logger.info(f"Completed steps: {list(completed)}")
                        return resume_step
                    else:
                        logger.info("All steps completed, nothing to resume")
                        return None
            
            # No completed steps found
            return None
            
        except Exception as e:
            logger.error(f"Failed to determine resume point: {e}")
            return None
    
    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        return step_name in self.completed_steps
    
    def cleanup_old_checkpoints(self, keep_latest: int = 3) -> None:
        """Clean up old checkpoint files to save disk space.
        
        Args:
            keep_latest: Number of latest checkpoints to keep for each step
        """
        try:
            for step_name in self.pipeline_steps:
                pattern = f"{step_name}_checkpoint.*"
                checkpoint_files = list(self.checkpoint_dir.glob(pattern))

                if len(checkpoint_files) > keep_latest:
                    # Sort by modification time and remove oldest
                    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    for old_file in checkpoint_files[keep_latest:]:
                        old_file.unlink()
                        logger.debug(f"Removed old checkpoint: {old_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def force_garbage_collection(self, step_name: str) -> None:
        """Force garbage collection and log memory usage."""
        self.log_memory_usage(step_name, "before_gc")
        
        # Force garbage collection
        collected = gc.collect()
        
        self.log_memory_usage(step_name, "after_gc")
        logger.info(f"Garbage collection: freed {collected} objects")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline progress and memory usage."""
        total_steps = len(self.pipeline_steps)
        completed_count = len(self.completed_steps)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_count,
            "progress_percent": (completed_count / total_steps) * 100,
            "completed_step_names": list(self.completed_steps),
            "remaining_steps": [s for s in self.pipeline_steps if s not in self.completed_steps],
            "memory_checkpoints": self.memory_checkpoints,
            "checkpoint_directory": str(self.checkpoint_dir)
        }