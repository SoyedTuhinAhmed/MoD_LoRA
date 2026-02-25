"""
Model loading and LoRA setup utilities.

Works with any HuggingFace sequence-classification model (BERT, Qwen2,
LLaMA, etc.).  Only the lightweight PEFT adapter weights are saved during
training; the frozen backbone is never written to disk.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
# Load base model + tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str = "google-bert/bert-base-uncased",
    num_labels: int = 2,
):
    """
    Download and return an AutoModelForSequenceClassification model and
    its tokenizer.  Works for BERT, Qwen2, LLaMA, and any other model
    supported by HuggingFace Transformers.

    The base model weights are kept frozen; LoRA adapters are added
    separately via :func:`apply_lora`.
    """
    print(f"[model] Loading tokenizer from '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some decoder-only tokenizers (Qwen, LLaMA …) have no pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] Loading model from '{model_name}' (num_labels={num_labels}) …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # Align model's pad_token_id if it was just set above
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Target-module helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_linear_module_names(model: nn.Module) -> List[str]:
    """
    Return the sorted list of *unique last-component names* of every
    ``nn.Linear`` layer in *model*.

    PEFT matches ``target_modules`` against the last component of each
    module's dotted path (e.g. ``"encoder.layer.0.attention.self.query"``
    matches target ``"query"``).  This helper lets you discover the right
    names for a new architecture before running training.

    Example output for BERT:  ['dense', 'key', 'query', 'value']
    Example output for Qwen2: ['down_proj', 'gate_proj', 'k_proj',
                                'o_proj', 'q_proj', 'up_proj', 'v_proj']
    """
    names = {name.split(".")[-1]
             for name, module in model.named_modules()
             if isinstance(module, nn.Linear)}
    return sorted(names)


def _validate_target_modules(model: nn.Module, target_modules: List[str]) -> None:
    """
    Raise a clear ``ValueError`` if none of *target_modules* match any
    ``nn.Linear`` layer name in *model*, listing the valid choices.
    """
    available = get_linear_module_names(model)
    matched = [t for t in target_modules if t in available]
    if not matched:
        raise ValueError(
            f"None of the requested target_modules {target_modules} were found "
            f"in the model's Linear layers.\n"
            f"Available names for '{type(model).__name__}': {available}\n"
            f"Pass the correct names via --lora_target_modules."
        )


# ─────────────────────────────────────────────────────────────────────────────
# LoRA setup
# ─────────────────────────────────────────────────────────────────────────────

# Sensible defaults per model family (last-component names)
_ARCH_DEFAULT_TARGETS = {
    "bert":   ["query", "value"],
    "qwen2":  ["q_proj", "v_proj"],
    "llama":  ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "falcon": ["query_key_value"],
    "gpt2":   ["c_attn"],
}


def _default_target_modules(model: nn.Module) -> List[str]:
    """Pick sensible LoRA targets based on the model's config model_type."""
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    for key, targets in _ARCH_DEFAULT_TARGETS.items():
        if key in model_type:
            available = get_linear_module_names(model)
            matched = [t for t in targets if t in available]
            if matched:
                return matched
    # Ultimate fallback: target all attention-like projection linears
    available = get_linear_module_names(model)
    fallback = [n for n in available if any(k in n for k in
                ("proj", "query", "key", "value", "attn", "qkv"))]
    return fallback if fallback else available


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
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
    target_modules : list[str] or None
        Names of sub-modules to adapt.  If ``None``, auto-detected from
        the model architecture.  Pass ``["all-linear"]`` to adapt every
        Linear layer.
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
        target_modules = _default_target_modules(model)
        print(f"[LoRA] Auto-detected target_modules: {target_modules}")
    elif target_modules != ["all-linear"]:
        _validate_target_modules(model, target_modules)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        inference_mode=False,
    )

    # PEFT ≥0.10 checks `torch.distributed.tensor.DTensor` inline at adapter
    # injection time.  On PyTorch 2.x the submodule exists but is not bound as
    # an attribute of `torch.distributed` until explicitly imported, which
    # triggers an AttributeError inside PEFT.  Force the binding here.
    try:
        import torch.distributed.tensor  # noqa: F401
    except ImportError:
        pass

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

    The frozen backbone is not written to disk.
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
        An AutoModelForSequenceClassification instance (no adapters).
    checkpoint_dir : str
        Directory containing the saved adapter
        (``adapter_model.safetensors`` + ``adapter_config.json``).
    device : str or torch.device

    Returns
    -------
    PeftModel
    """
    print(f"[checkpoint] Loading LoRA adapter from '{checkpoint_dir}' …")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device)
    return model

