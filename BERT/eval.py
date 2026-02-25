"""
Evaluation script for a saved BERT + LoRA checkpoint.

Computes accuracy (and per-class breakdown) on the validation and/or
test splits and appends the results to the experiment JSON log.
"""

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import load_and_tokenize_dataset, build_dataloaders, DATASET_CONFIG
from utils.model_utils import load_model_and_tokenizer, load_lora_checkpoint
from utils.logging_utils import ExperimentLogger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_eval(model, loader: DataLoader, device: torch.device, label_names: List[str]):
    """
    Return a dict with loss, accuracy, and per-class accuracy.
    """
    model.eval()
    total_loss  = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    n = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n

    # Per-class accuracy
    per_class: Dict[str, float] = {}
    for idx, name in enumerate(label_names):
        class_total   = sum(1 for l in all_labels if l == idx)
        class_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == idx and p == idx)
        per_class[name] = round(class_correct / class_total, 6) if class_total > 0 else None

    return {
        "loss":          round(total_loss / len(loader), 6),
        "accuracy":      round(accuracy, 6),
        "per_class_acc": per_class,
        "n_samples":     n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(config: Dict, checkpoint_dir: Optional[str] = None):
    """
    Load a trained LoRA checkpoint and evaluate on validation + test splits.

    Parameters
    ----------
    config : dict
        The same config dict used during training (loaded from JSON log or
        passed directly from main.py).
    checkpoint_dir : str, optional
        Path to the LoRA adapter directory.  Defaults to the ``best``
        checkpoint inside ``config["output_dir"]``.

    Returns
    -------
    results : dict
        ``{val: {...}, test: {...}}``
    """
    # ── Device ──────────────────────────────────────────────────────────────
    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    print(f"[eval] Using device: {device}")

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config["output_dir"], "checkpoints", "best")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: '{checkpoint_dir}'. "
            "Run training first or pass --checkpoint_dir explicitly."
        )

    # ── Dataset info ─────────────────────────────────────────────────────────
    dataset_name = config["dataset"]
    ds_cfg       = DATASET_CONFIG[dataset_name]
    num_labels   = ds_cfg["num_labels"]
    label_names  = ds_cfg["label_names"]

    # ── Model ────────────────────────────────────────────────────────────────
    base_model, tokenizer = load_model_and_tokenizer(
        model_name=config["model_name"],
        num_labels=num_labels,
    )
    model = load_lora_checkpoint(base_model, checkpoint_dir, device=device)

    # ── Data ─────────────────────────────────────────────────────────────────
    _, val_ds, test_ds, _ = load_and_tokenize_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )
    # Dummy train_ds placeholder (not used here)
    _, val_loader, test_loader = build_dataloaders(
        val_ds, val_ds, test_ds,
        batch_size=config.get("batch_size", 64),
        num_workers=config.get("num_workers", 4),
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[eval] Evaluating on validation split …")
    val_results = _run_eval(model, val_loader, device, label_names)
    print(f"  val_loss={val_results['loss']:.4f}  val_acc={val_results['accuracy']:.4f}")

    print("[eval] Evaluating on test split …")
    test_results = _run_eval(model, test_loader, device, label_names)
    print(f"  test_loss={test_results['loss']:.4f}  test_acc={test_results['accuracy']:.4f}")

    # MNLI reports matched & mismatched separately
    split_label = "validation_matched" if dataset_name == "mnli" else "validation"
    results = {
        split_label:         val_results,
        "test": test_results,
        "checkpoint_dir":    checkpoint_dir,
    }

    # Per-class breakdown
    print("\n[eval] Per-class validation accuracy:")
    for cls_name, cls_acc in val_results["per_class_acc"].items():
        print(f"  {cls_name}: {cls_acc:.4f}")

    # ── Log ──────────────────────────────────────────────────────────────────
    log_path = os.path.join(config["output_dir"], "experiment_log.json")
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log_data = json.load(f)
        log_data["final_eval"] = results
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"\n[eval] Results appended to experiment log → {log_path}")
    else:
        # Stand-alone evaluation without a prior training log
        logger = ExperimentLogger(log_dir=config["output_dir"], run_name="eval_only")
        logger.log_config(config)
        logger.log_final_eval(results)
        logger.finish()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import build_config, parse_args

    parser = argparse.ArgumentParser(parents=[], add_help=False)
    # Allow overriding the checkpoint location at eval time
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to LoRA adapter checkpoint (overrides best checkpoint)")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # pass leftover args to parse_args

    from main import parse_args as _pa
    main_args = _pa()
    cfg = build_config(main_args)

    evaluate(cfg, checkpoint_dir=args.checkpoint_dir)
