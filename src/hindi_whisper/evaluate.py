"""Held-out WER."""

from __future__ import annotations

import torch
from jiwer import wer

from hindi_whisper.model import ensure_generation


def _as_features(model, values):
    param = next(model.parameters())
    feats = torch.as_tensor(values, device=param.device, dtype=param.dtype)
    if feats.ndim == 2:
        feats = feats.unsqueeze(0)
    return feats


def word_error_rate(model, processor, eval_ds, n: int, language: str) -> float:
    model.eval()
    ensure_generation(model, language)
    rows = eval_ds.select(range(min(n, len(eval_ds))))
    hyps: list[str] = []
    refs: list[str] = []
    for index in range(len(rows)):
        feats = _as_features(model, rows[index]["input_features"])
        with torch.no_grad():
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=128)
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        label_ids = [token for token in rows[index]["labels"] if token != -100]
        refs.append(processor.decode(label_ids, skip_special_tokens=True))
        if index % 10 == 0:
            print(f"wer {index + 1}/{len(rows)}", flush=True)
    return float(wer(refs, hyps))
