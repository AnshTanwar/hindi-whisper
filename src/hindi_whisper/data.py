"""Public Hindi speech loading and Whisper feature prep."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from datasets import Audio, Dataset, load_dataset

from hindi_whisper.config import ExperimentConfig


@dataclass
class SpeechBatch:
    train: Dataset
    eval: Dataset
    eval_audio: Dataset
    text_key: str


def load_raw(cfg: ExperimentConfig):
    if cfg.dataset == "fleurs":
        train_raw = load_dataset("google/fleurs", "hi_in", split="train")
        eval_raw = load_dataset("google/fleurs", "hi_in", split="validation")
        text_key = "transcription"
    elif cfg.dataset == "common_voice":
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        train_raw = load_dataset(
            "mozilla-foundation/common_voice_11_0", "hi", split="train", token=token
        )
        eval_raw = load_dataset(
            "mozilla-foundation/common_voice_11_0", "hi", split="validation", token=token
        )
        text_key = "sentence"
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}. Use fleurs or common_voice.")

    train_raw = train_raw.cast_column("audio", Audio(sampling_rate=16000))
    eval_raw = eval_raw.cast_column("audio", Audio(sampling_rate=16000))
    return train_raw, eval_raw, text_key


def _usable(example: dict, text_key: str) -> bool:
    text = (example.get(text_key) or "").strip()
    samples = example["audio"]["array"].shape[-1]
    duration = samples / 16000
    return bool(text) and 0.6 <= duration <= 30.0


def prepare_datasets(cfg: ExperimentConfig, processor) -> SpeechBatch:
    train_raw, eval_raw, text_key = load_raw(cfg)

    def prepare(batch: dict) -> dict:
        audio = batch["audio"]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )
        batch["input_features"] = feats.input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_key]).input_ids
        return batch

    train_ds = train_raw.select(range(min(len(train_raw), cfg.max_train * 2)))
    eval_ds = eval_raw.select(range(min(len(eval_raw), cfg.max_eval * 2)))
    train_ds = train_ds.filter(lambda ex: _usable(ex, text_key))
    eval_ds = eval_ds.filter(lambda ex: _usable(ex, text_key))
    train_ds = train_ds.select(range(min(cfg.max_train, len(train_ds))))
    eval_audio = eval_ds.select(range(min(cfg.max_eval, len(eval_ds))))
    eval_ds = eval_audio.map(prepare, remove_columns=eval_audio.column_names)
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names)
    return SpeechBatch(train=train_ds, eval=eval_ds, eval_audio=eval_audio, text_key=text_key)


class SpeechCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        import torch

        feats = [{"input_features": row["input_features"]} for row in features]
        batch = self.processor.feature_extractor.pad(feats, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": row["labels"]} for row in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
