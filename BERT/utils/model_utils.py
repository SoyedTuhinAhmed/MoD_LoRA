"""
Model loading and LoRA setup utilities.

Only the lightweight PEFT adapter weights are saved during training; the
frozen BERT backbone is never written to disk.
"""

import os
from typing import List

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, BertForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
# Load base model + tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str = "google-bert/bert-base-uncased",
    num_labels: int = 2,
):
    """
    Download and return a BertForSequenceClassification model and its tokenizer.

    The base model weights are kept frozen; LoRA adapters are added separately
    via :func:`apply_lora`.
    """
    print(f"[model] Loading tokenizer from '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[model] Loading model from '{model_name}' (num_labels={num_labels}) …")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# LoRA setup
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_MODULES = ["query", "value"]


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
    bias: str = "none",
):
    """
    Wrap *model* with PEFT LoRA adapters and return the PeftModel.

    Parameters
    ----------
    r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor (effective scale = lora_alpha / r).
    lora_dropout : float
        Dropout applied inside LoRA layers.
    target_modules : list[str]
        Names of sub-modules to adapt. Defaults to ``["query", "value"]``
        (BERT self-attention projections).
    bias : str
        Whether to train bias parameters (``"none"``, ``"all"``,
        ``"lora_only"``).

    Returns
    -------
    peft_model : PeftModel
        Model with LoRA adapters; all base weights are frozen.
    lora_cfg : LoraConfig
        The configuration object (serialisable to dict for logging).
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        inference_mode=False,
    )

    peft_model = get_peft_model(model, lora_cfg)

    trainable, total = peft_model.get_nb_trainable_parameters()
    print(
        f"[LoRA] Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f} %)"
    )
    return peft_model, lora_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (LoRA-only)
# ─────────────────────────────────────────────────────────────────────────────

def save_lora_checkpoint(model, output_dir: str, tag: str = ""):
    """
    Save *only* the LoRA adapter weights to ``output_dir/tag``.

    The frozen BERT backbone is not written to disk.
    """
    save_path = os.path.join(output_dir, tag) if tag else output_dir
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"[checkpoint] LoRA adapter saved → {save_path}")
    return save_path


def load_lora_checkpoint(base_model, checkpoint_dir: str, device="cpu"):
    """
    Reload LoRA adapter weights into a freshly loaded base model.

    Parameters
    ----------
    base_model :
        A BertForSequenceClassification instance (base weights, no adapters).
    checkpoint_dir : str
        Directory containing the saved adapter (``adapter_model.bin`` /
        ``adapter_model.safetensors`` + ``adapter_config.json``).
    device : str or torch.device

    Returns
    -------
    PeftModel
    """
    print(f"[checkpoint] Loading LoRA adapter from '{checkpoint_dir}' …")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device)
    return model
