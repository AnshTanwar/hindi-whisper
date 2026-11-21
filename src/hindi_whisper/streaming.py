"""Chunked streaming decode with LocalAgreement-2.

Policy: Liu et al. 2020, used by the IWSLT 2022 winning CUNI-KIT system,
applied to Whisper by Macháček, Dabre, and Bojar, 2023.
https://arxiv.org/abs/2307.14743
"""

from __future__ import annotations

import time

import numpy as np

from hindi_whisper.agreement import longest_common_prefix
from hindi_whisper.model import ensure_generation


def local_agreement_stream(model, processor, audio, language: str, sr: int = 16000, chunk_s: float = 1.0):
    """Commit a word prefix only when two consecutive chunk decodes agree."""
    import torch

    param = next(model.parameters())
    device = param.device
    model.eval()
    ensure_generation(model, language)
    audio = np.asarray(audio, dtype="float32")
    committed: list[str] = []
    previous: list[str] = []
    latencies: list[float] = []
    hop = int(chunk_s * sr)
    max_samples = 30 * sr
    started = time.perf_counter()
    end = hop
    while True:
        buf = audio[max(0, end - max_samples) : min(end, len(audio))]
        feats = processor.feature_extractor(buf, sampling_rate=sr, return_tensors="pt").input_features.to(
            device=device, dtype=param.dtype
        )
        with torch.no_grad():
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=128)
        words = processor.batch_decode(ids, skip_special_tokens=True)[0].split()
        prefix = longest_common_prefix(previous, words)
        if len(prefix) > len(committed):
            latencies.append(time.perf_counter() - started)
            committed = prefix
        previous = words
        if end >= len(audio):
            if len(words) > len(committed):
                committed = words
            break
        end += hop
    hypothesis = " ".join(committed)
    mean_latency = float(sum(latencies) / len(latencies)) if latencies else None
    return hypothesis, mean_latency, len(latencies)


def concatenate_clips(eval_audio, n: int):
    pieces = [eval_audio[i]["audio"]["array"] for i in range(min(n, len(eval_audio)))]
    if not pieces:
        raise RuntimeError("No audio clips to stream.")
    return np.concatenate(pieces).astype("float32")
