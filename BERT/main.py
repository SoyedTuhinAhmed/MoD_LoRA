"""
Main entry point for BERT + LoRA fine-tuning with PHANTOM noise injection.

Model build pipeline (training mode)
-------------------------------------
  1. Load google-bert/bert-base-uncased (base model + tokenizer)
  2. Apply temperature-aware ReRAM drift via noise_injection.py
       → all base weights are perturbed and frozen
  3. Attach PEFT LoRA adapters to the noisy backbone
       → only LoRA params are trainable
  4. Train with train.py (checkpoints save LoRA params only)

Usage examples
--------------
# Train on SST-2 with defaults (T_tile=360 K, no extra noise):
python main.py --dataset sst2 --mode train

# Train on MNLI at 400 K with stochastic noise:
python main.py --dataset mnli --mode both --T_tile 400 --sigma_rel 0.05

# Evaluate a previously trained checkpoint:
python main.py --dataset sst2 --mode eval --output_dir runs/sst2_lora_001

# Custom LoRA + injection settings:
python main.py --dataset sst2 --lora_r 8 --lora_alpha 16 \
               --T_tile 360 --sigma_rel 0.02 --inject_bias
"""

import argparse
import json
import os
import sys
from datetime import datetime

from noise_injection import make_noisy_model_for_temperature, InjectionConfig
from utils.model_utils import load_model_and_tokenizer, apply_lora


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

    # ── Noise injection ──────────────────────────────────────────────────────
    noise_grp = parser.add_argument_group("Noise Injection (PHANTOM ReRAM drift)")
    noise_grp.add_argument(
        "--T_tile",
        type=float,
        default=360.0,
        help="Tile temperature in Kelvin for PHANTOM drift model (300–400 K range).",
    )
    noise_grp.add_argument(
        "--sigma_rel",
        type=float,
        default=0.0,
        help="Relative std for optional Gaussian noise on top of drift "
             "(0.0 = pure multiplicative drift only).",
    )
    noise_grp.add_argument(
        "--inject_bias",
        action="store_true",
        default=False,
        help="Also perturb bias parameters (default: keep biases digital/clean).",
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

        # Noise injection
        "T_tile":       args.T_tile,
        "sigma_rel":    args.sigma_rel,
        "inject_bias":  args.inject_bias,

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

def build_model(config: dict):
    """
    Build the full model pipeline:
      1. Load BERT base model + tokenizer
      2. Apply PHANTOM temperature-aware noise injection (freezes base weights)
      3. Attach LoRA adapters (only these will be trained)

    Returns
    -------
    model : PeftModel
    tokenizer
    lora_cfg : LoraConfig  (for logging)
    injection_meta : dict  (beta, T_tile, … for logging)
    """
    from utils.dataset_utils import DATASET_CONFIG

    num_labels = DATASET_CONFIG[config["dataset"]]["num_labels"]

    # Step 1 – base model
    print(f"[main] Loading base model '{config['model_name']}' …")
    base_model, tokenizer = load_model_and_tokenizer(
        model_name=config["model_name"],
        num_labels=num_labels,
    )

    # Step 2 – noise injection
    T_tile = config["T_tile"]
    injection_cfg = InjectionConfig(
        sigma_rel=config["sigma_rel"],
        inject_bias=config["inject_bias"],
    )
    print(f"[main] Applying PHANTOM noise injection at T_tile={T_tile} K …")
    noisy_model = make_noisy_model_for_temperature(
        base_model,
        T_tile,
        config=injection_cfg,
    )
    beta = getattr(noisy_model, "_injection_beta", None)
    print(f"[main] beta(T={T_tile} K) = {beta:.4f}  "
          f"sigma_rel={injection_cfg.sigma_rel}  inject_bias={injection_cfg.inject_bias}")

    injection_meta = {
        "T_tile_K":      T_tile,
        "beta":          beta,
        "sigma_rel":     injection_cfg.sigma_rel,
        "inject_bias":   injection_cfg.inject_bias,
        "inject_linear": injection_cfg.inject_linear,
        "inject_conv2d": injection_cfg.inject_conv2d,
    }

    # Step 3 – attach LoRA to noisy_model
    # Base params are already frozen; apply_lora re-enables grads on adapters only.
    print("[main] Attaching LoRA adapters to noisy model …")
    model, lora_cfg = apply_lora(
        noisy_model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias=config.get("lora_bias", "none"),
    )

    return model, tokenizer, lora_cfg, injection_meta


def main():
    args   = parse_args()
    config = build_config(args)

    print("\n" + "=" * 60)
    print(f"  Dataset  : {config['dataset'].upper()}")
    print(f"  Mode     : {config['mode']}")
    print(f"  Model    : {config['model_name']}")
    print(f"  T_tile   : {config['T_tile']} K  sigma_rel={config['sigma_rel']}")
    print(f"  LoRA r   : {config['lora_r']}  alpha={config['lora_alpha']}")
    print(f"  Output   : {config['output_dir']}")
    print("=" * 60 + "\n")

    if config["mode"] in ("train", "both"):
        from train import train

        model, tokenizer, lora_cfg, injection_meta = build_model(config)

        # Embed injection metadata into config so it lands in experiment_log.json
        config["noise_injection"] = injection_meta

        best_ckpt, best_acc = train(config, model, tokenizer, lora_cfg)
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
