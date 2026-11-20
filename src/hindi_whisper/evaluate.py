"""Held-out WER."""

from __future__ import annotations

import torch
from jiwer import wer


def word_error_rate(model, processor, eval_ds, n: int, language: str) -> float:
    device = next(model.parameters()).device
    model.eval()
    rows = eval_ds.select(range(min(n, len(eval_ds))))
    hyps: list[str] = []
    refs: list[str] = []
    for index in range(len(rows)):
        feats = torch.tensor(rows[index]["input_features"]).unsqueeze(0).to(device)
        with torch.no_grad():
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=128)
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        label_ids = [token for token in rows[index]["labels"] if token != -100]
        refs.append(processor.decode(label_ids, skip_special_tokens=True))
    return float(wer(refs, hyps))
