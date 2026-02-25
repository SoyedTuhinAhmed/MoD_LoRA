"""
Dataset loading and preprocessing utilities for SST-2 and MNLI.
Both datasets are loaded from the GLUE benchmark via HuggingFace datasets.
"""

from datasets import load_dataset
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset metadata
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "sst2": {
        "hf_path": "glue",
        "hf_name": "sst2",
        "num_labels": 2,
        "label_names": ["negative", "positive"],
        "text_columns": ["sentence"],      # single-sentence task
        "label_column": "label",
        "splits": {
            "train": "train",
            "validation": "validation",
            # GLUE SST-2 test split has no labels; use validation as test proxy
            "test": "validation",
        },
        "metric": "accuracy",
        "description": "Stanford Sentiment Treebank – binary sentiment classification",
    },
    "mnli": {
        "hf_path": "glue",
        "hf_name": "mnli",
        "num_labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "text_columns": ["premise", "hypothesis"],   # sentence-pair task
        "label_column": "label",
        "splits": {
            "train": "train",
            "validation": "validation_matched",
            "test": "validation_mismatched",          # report on both splits
        },
        "metric": "accuracy",
        "description": "Multi-Genre Natural Language Inference – 3-class NLI",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_tokenize_fn(dataset_name: str, tokenizer, max_length: int):
    """Return a batched tokenisation function appropriate for the dataset."""

    if dataset_name == "sst2":
        def tokenize_fn(batch):
            return tokenizer(
                batch["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
    elif dataset_name == "mnli":
        def tokenize_fn(batch):
            return tokenizer(
                batch["premise"],
                batch["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG)}")

    return tokenize_fn


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_tokenize_dataset(dataset_name: str, tokenizer, max_length: int = 128):
    """
    Download, tokenise, and return train / validation / test splits.

    Parameters
    ----------
    dataset_name : str
        One of ``"sst2"`` or ``"mnli"``.
    tokenizer :
        A HuggingFace tokenizer compatible with BERT.
    max_length : int
        Maximum token sequence length (sequences are padded / truncated).

    Returns
    -------
    train_dataset, val_dataset, test_dataset : datasets.Dataset
        PyTorch-format datasets with columns:
        ``input_ids``, ``attention_mask``, ``token_type_ids``, ``labels``.
    config : dict
        The dataset metadata dict from :data:`DATASET_CONFIG`.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid choices: {list(DATASET_CONFIG)}")

    cfg = DATASET_CONFIG[dataset_name]

    print(f"[dataset] Loading '{cfg['hf_name']}' from '{cfg['hf_path']}' …")
    raw = load_dataset(cfg["hf_path"], cfg["hf_name"])

    tokenize_fn = _build_tokenize_fn(dataset_name, tokenizer, max_length)

    print("[dataset] Tokenising splits …")
    tokenized = raw.map(tokenize_fn, batched=True, desc="Tokenising")

    # Rename label column so all datasets share the same ``labels`` key
    tokenized = tokenized.rename_column(cfg["label_column"], "labels")

    # Keep only the tensor columns that the model needs
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    # token_type_ids may not be present for every tokenizer; keep only what exists
    available = set(tokenized[cfg["splits"]["train"]].column_names)
    keep_cols = [c for c in keep_cols if c in available]

    tokenized.set_format("torch", columns=keep_cols)

    splits = cfg["splits"]
    train_dataset = tokenized[splits["train"]]
    val_dataset   = tokenized[splits["validation"]]
    test_dataset  = tokenized[splits["test"]]

    print(
        f"[dataset] Sizes — train: {len(train_dataset):,} | "
        f"val: {len(val_dataset):,} | test: {len(test_dataset):,}"
    )
    return train_dataset, val_dataset, test_dataset, cfg


def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Wrap datasets in DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
