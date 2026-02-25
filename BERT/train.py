"""
Training script for BERT + LoRA fine-tuning on SST-2 or MNLI.

Called by main.py; can also be run directly (see argument definitions at
the bottom of this file, mirrored in main.py for consistency).
"""

import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils.dataset_utils import load_and_tokenize_dataset, build_dataloaders
from utils.model_utils import save_lora_checkpoint
from utils.logging_utils import ExperimentLogger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        total_acc  += _accuracy(outputs.logits, batch["labels"])
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_acc  += _accuracy(outputs.logits, batch["labels"])
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(config: Dict, model, tokenizer, lora_cfg=None):
    """
    Full training pipeline.

    Parameters
    ----------
    config : dict
        Flat dictionary produced by ``main.py`` containing all
        hyperparameters and path settings.
    model :
        A PeftModel (base + LoRA adapters) already built and ready to train.
        Base parameters must already be frozen; only LoRA params need grads.
    tokenizer :
        HuggingFace tokenizer matching the model.
    lora_cfg : LoraConfig, optional
        LoRA configuration object; logged to the experiment log if supplied.
    """
    # ── Device ──────────────────────────────────────────────────────────────
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    print(f"[train] Using device: {device}")

    model = model.to(device)

    # ── Reproducibility ─────────────────────────────────────────────────────
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = ExperimentLogger(
        log_dir=config["output_dir"],
        run_name=config.get("run_name", ""),
    )
    logger.log_config(config)
    if lora_cfg is not None:
        logger.log_lora_config(lora_cfg)

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset_name = config["dataset"]
    num_labels   = config["num_labels"]

    train_ds, val_ds, test_ds, ds_cfg = load_and_tokenize_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )
    logger.log_dataset_info({
        "dataset": dataset_name,
        "num_labels": num_labels,
        "label_names": ds_cfg["label_names"],
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "max_length": config["max_length"],
    })

    train_loader, val_loader, _ = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
    )

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP scaler (only if CUDA is available)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_acc  = -1.0
    best_ckpt_path = None
    save_every    = config["save_every_n_epochs"]
    ckpt_base     = os.path.join(config["output_dir"], "checkpoints")

    print(f"\n[train] Starting training for {config['num_epochs']} epoch(s) …\n")

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        metrics = {
            "train_loss":    round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss":      round(val_loss, 6),
            "val_accuracy":  round(val_acc, 6),
            "lr":            current_lr,
            "epoch_time_s":  round(elapsed, 2),
        }
        logger.log_epoch(epoch, metrics)

        print(
            f"Epoch {epoch:>3}/{config['num_epochs']} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        # ── Best model checkpoint ────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = save_lora_checkpoint(model, ckpt_base, tag="best")
            logger.log_checkpoint(epoch, best_ckpt_path, reason="best_val_accuracy")
            logger.log_best(epoch, val_acc, best_ckpt_path)
            print(f"  ↑ New best val_acc={val_acc:.4f} — adapter saved to '{best_ckpt_path}'")

        # ── Periodic checkpoint ──────────────────────────────────────────
        if save_every > 0 and epoch % save_every == 0:
            periodic_path = save_lora_checkpoint(model, ckpt_base, tag=f"epoch_{epoch:03d}")
            logger.log_checkpoint(epoch, periodic_path, reason="periodic")

    print(f"\n[train] Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"[train] Best adapter checkpoint: {best_ckpt_path}")

    logger.finish()
    return best_ckpt_path, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import build_config, build_model, parse_args

    args             = parse_args()
    cfg              = build_config(args)
    model, tok, lora_cfg, inj_meta = build_model(cfg)
    cfg["noise_injection"] = inj_meta
    train(cfg, model, tok, lora_cfg)
