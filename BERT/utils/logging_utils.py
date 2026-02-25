"""
JSON-based experiment logger.

Writes a structured log file that captures:
  - Full configuration (hyperparameters, LoRA settings, paths …)
  - Per-epoch training and validation metrics
  - Checkpoint events (periodic + best-model saves)
  - Final evaluation results
"""

import json
import os
from datetime import datetime
from typing import Any, Dict


class ExperimentLogger:
    """
    Accumulates experiment data and flushes it to a JSON file after every
    update so that partial results are not lost on crash.

    Parameters
    ----------
    log_dir : str
        Directory where ``experiment_log.json`` will be written.
    run_name : str, optional
        Human-readable label embedded in the log.
    """

    def __init__(self, log_dir: str, run_name: str = ""):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "experiment_log.json")
        self._data: Dict[str, Any] = {
            "run_name": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "config": {},
            "lora_config": {},
            "dataset_info": {},
            "training_history": [],   # list of per-epoch dicts
            "checkpoints": [],        # list of checkpoint event dicts
            "best": {
                "epoch": None,
                "val_accuracy": None,
                "checkpoint_path": None,
            },
            "final_eval": {},
        }
        self._flush()

    # ──────────────────────────────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────────────────────────────

    def log_config(self, config: Dict[str, Any]):
        """Store the full run configuration."""
        self._data["config"] = _jsonify(config)
        self._flush()

    def log_lora_config(self, lora_cfg):
        """Store LoRA-specific hyperparameters."""
        if hasattr(lora_cfg, "to_dict"):
            lora_dict = lora_cfg.to_dict()
        elif hasattr(lora_cfg, "__dict__"):
            lora_dict = vars(lora_cfg)
        else:
            lora_dict = dict(lora_cfg)
        self._data["lora_config"] = _jsonify(lora_dict)
        self._flush()

    def log_dataset_info(self, info: Dict[str, Any]):
        """Store dataset metadata (sizes, label names, …)."""
        self._data["dataset_info"] = _jsonify(info)
        self._flush()

    # ──────────────────────────────────────────────────────────────────────
    # Training progress
    # ──────────────────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """
        Append one epoch's metrics to the training history.

        Example ``metrics`` dict::

            {
                "train_loss": 0.312,
                "val_loss":   0.289,
                "val_accuracy": 0.924,
                "lr": 2e-4,
                "epoch_time_s": 143.2,
            }
        """
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        entry.update(_jsonify(metrics))
        self._data["training_history"].append(entry)
        self._flush()

    def log_checkpoint(self, epoch: int, checkpoint_path: str, reason: str = "periodic"):
        """Record a checkpoint save event."""
        self._data["checkpoints"].append({
            "epoch": epoch,
            "path": checkpoint_path,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        self._flush()

    def log_best(self, epoch: int, val_accuracy: float, checkpoint_path: str):
        """Update the best-model record."""
        self._data["best"] = {
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
        }
        self._flush()

    # ──────────────────────────────────────────────────────────────────────
    # Final evaluation
    # ──────────────────────────────────────────────────────────────────────

    def log_final_eval(self, results: Dict[str, Any]):
        """Store final evaluation results (val / test splits)."""
        self._data["final_eval"] = _jsonify(results)
        self._flush()

    def finish(self):
        """Mark the run as complete and flush."""
        self._data["finished_at"] = datetime.now().isoformat()
        self._flush()
        print(f"[logger] Experiment log saved → {self.log_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _flush(self):
        with open(self.log_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    @property
    def log_file(self):
        return self.log_path


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _jsonify(obj):
    """Recursively convert non-serialisable types to strings."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    # numpy / torch scalars
    try:
        return obj.item()
    except (AttributeError, ValueError):
        pass
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)
