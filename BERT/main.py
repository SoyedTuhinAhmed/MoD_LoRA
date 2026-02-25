"""
Main entry point for BERT + LoRA fine-tuning.

Usage examples
--------------
# Train on SST-2 with defaults:
python main.py --dataset sst2 --mode train

# Train on MNLI, then evaluate:
python main.py --dataset mnli --mode both --num_epochs 3 --batch_size 16

# Evaluate a previously trained checkpoint:
python main.py --dataset sst2 --mode eval --output_dir runs/sst2_lora_001

# Custom LoRA settings:
python main.py --dataset sst2 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
               --lora_target_modules query value key
"""

import argparse
import json
import os
import sys
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT + LoRA fine-tuning for SST-2 / MNLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Task ────────────────────────────────────────────────────────────────
    task = parser.add_argument_group("Task")
    task.add_argument(
        "--dataset",
        type=str,
        choices=["sst2", "mnli"],
        required=True,
        help="Classification dataset to fine-tune on.",
    )
    task.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="'train' only, 'eval' only (requires a checkpoint), or 'both'.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="HuggingFace model ID for the BERT backbone.",
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_grp = parser.add_argument_group("LoRA")
    lora_grp.add_argument("--lora_r",              type=int,   default=16,
                          help="LoRA rank.")
    lora_grp.add_argument("--lora_alpha",          type=int,   default=32,
                          help="LoRA scaling factor.")
    lora_grp.add_argument("--lora_dropout",        type=float, default=0.1,
                          help="Dropout inside LoRA layers.")
    lora_grp.add_argument("--lora_target_modules", type=str,   nargs="+",
                          default=["query", "value"],
                          help="BERT sub-module names to adapt with LoRA.")
    lora_grp.add_argument("--lora_bias",           type=str,   default="none",
                          choices=["none", "all", "lora_only"],
                          help="Whether to train bias parameters.")

    # ── Training ─────────────────────────────────────────────────────────────
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--num_epochs",          type=int,   default=5)
    train_grp.add_argument("--batch_size",          type=int,   default=32)
    train_grp.add_argument("--learning_rate",       type=float, default=2e-4)
    train_grp.add_argument("--weight_decay",        type=float, default=0.01)
    train_grp.add_argument("--warmup_ratio",        type=float, default=0.06,
                           help="Fraction of total steps used for LR warm-up.")
    train_grp.add_argument("--max_length",          type=int,   default=128,
                           help="Maximum token sequence length.")
    train_grp.add_argument("--save_every_n_epochs", type=int,   default=1,
                           help="Save a periodic checkpoint every N epochs (0 = disabled).")
    train_grp.add_argument("--num_workers",         type=int,   default=4,
                           help="DataLoader worker processes.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_grp = parser.add_argument_group("Evaluation")
    eval_grp.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a LoRA adapter checkpoint for eval-only mode. "
             "Defaults to the 'best' checkpoint inside --output_dir.",
    )

    # ── Output / misc ────────────────────────────────────────────────────────
    misc_grp = parser.add_argument_group("Misc")
    misc_grp.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory for checkpoints and logs. "
             "Defaults to 'runs/<dataset>_lora_<timestamp>'.",
    )
    misc_grp.add_argument("--seed",   type=int, default=42)
    misc_grp.add_argument("--device", type=str, default="auto",
                          help="'auto', 'cpu', 'cuda', or 'cuda:N'.")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

NUM_LABELS = {"sst2": 2, "mnli": 3}


def build_config(args) -> dict:
    """Convert parsed args into a flat config dict and resolve defaults."""
    from utils.dataset_utils import DATASET_CONFIG

    ds_cfg = DATASET_CONFIG[args.dataset]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{args.dataset}_lora_r{args.lora_r}_{timestamp}"

    output_dir = args.output_dir or os.path.join("runs", run_name)

    config = {
        # Identity
        "run_name":   run_name,
        "timestamp":  timestamp,

        # Task
        "dataset":    args.dataset,
        "mode":       args.mode,
        "num_labels": ds_cfg["num_labels"],

        # Model
        "model_name": args.model_name,

        # LoRA
        "lora_r":              args.lora_r,
        "lora_alpha":          args.lora_alpha,
        "lora_dropout":        args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "lora_bias":           args.lora_bias,

        # Training
        "num_epochs":          args.num_epochs,
        "batch_size":          args.batch_size,
        "learning_rate":       args.learning_rate,
        "weight_decay":        args.weight_decay,
        "warmup_ratio":        args.warmup_ratio,
        "max_length":          args.max_length,
        "save_every_n_epochs": args.save_every_n_epochs,
        "num_workers":         args.num_workers,

        # Eval
        "checkpoint_dir": args.checkpoint_dir,

        # Misc
        "output_dir": output_dir,
        "seed":       args.seed,
        "device":     args.device,
    }

    os.makedirs(output_dir, exist_ok=True)

    # Persist config immediately so it is available even before training starts
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[main] Config saved → {config_path}")

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = build_config(args)

    print("\n" + "=" * 60)
    print(f"  Dataset  : {config['dataset'].upper()}")
    print(f"  Mode     : {config['mode']}")
    print(f"  Model    : {config['model_name']}")
    print(f"  LoRA r   : {config['lora_r']}  alpha={config['lora_alpha']}")
    print(f"  Output   : {config['output_dir']}")
    print("=" * 60 + "\n")

    if config["mode"] in ("train", "both"):
        from train import train
        best_ckpt, best_acc = train(config)
        print(f"\n[main] Training finished — best val_acc={best_acc:.4f}")

    if config["mode"] in ("eval", "both"):
        from eval import evaluate
        results = evaluate(
            config,
            checkpoint_dir=config.get("checkpoint_dir"),
        )
        # Print a compact summary
        print("\n[main] ── Evaluation Summary ──")
        for split, metrics in results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                print(f"  {split}: acc={metrics['accuracy']:.4f}  loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    # Ensure imports resolve relative to BERT/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
