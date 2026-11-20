"""Experiment config loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    run_name: str = "fleurs-small"
    dataset: str = "fleurs"
    model_name: str = "openai/whisper-small"
    language: str = "hindi"
    max_train: int = 600
    max_eval: int = 80
    lora_steps: int = 150
    distill_steps: int = 150
    lora_batch: int = 4
    distill_batch: int = 2
    grad_accum: int = 4
    learning_rate: float = 1e-4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    kl_weight: float = 0.8
    temperature: float = 2.0
    seed: int = 7
    output_dir: str = "outputs"
    eval_n: int = 40
    stream_clips: int = 4
    stream_chunk_s: float = 1.0
    warmup_steps: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
