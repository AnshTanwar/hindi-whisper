"""Decoder distillation against a frozen Hindi-adapted Whisper teacher."""

from __future__ import annotations

import os

from hindi_whisper.config import ExperimentConfig
from hindi_whisper.data import SpeechCollator
from hindi_whisper.model import build_distilled_student, distillation_loss


def distill(cfg: ExperimentConfig, merged_teacher, processor, train_ds, output_dir: str):
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    merged_teacher.eval()
    for param in merged_teacher.parameters():
        param.requires_grad = False

    student = build_distilled_student(merged_teacher)
    # Student stays fp32. The previous T4 run cast both models to fp16 and the KL term went NaN.
    # The frozen teacher can score in fp16; its logits are cast back to fp32 in the loss.
    if device == "cuda":
        merged_teacher.half()
    merged_teacher.to(device)
    student.to(device)
    print(
        f"distill device {device} teacher {next(merged_teacher.parameters()).dtype} "
        f"student {next(student.parameters()).dtype}",
        flush=True,
    )

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.distill_batch,
        shuffle=True,
        collate_fn=SpeechCollator(processor),
    )
    iterator = iter(loader)
    student.train()
    losses: list[float] = []
    for step in range(cfg.distill_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        teacher_dtype = next(merged_teacher.parameters()).dtype
        with torch.no_grad():
            teacher_logits = merged_teacher(
                input_features=batch["input_features"].to(dtype=teacher_dtype),
                labels=batch["labels"],
            ).logits
        student_out = student(
            input_features=batch["input_features"].float(),
            labels=batch["labels"],
        )
        loss = distillation_loss(
            student_out.logits,
            teacher_logits,
            batch["labels"],
            cfg.kl_weight,
            cfg.temperature,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite distillation loss at step {step}: {float(loss.detach())}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0:
            print(f"distill step {step} loss {losses[-1]:.3f}", flush=True)

    student.config.use_cache = True
    merged_teacher.float()
    merged_teacher.config.use_cache = True
    student_dir = os.path.join(output_dir, "student")
    student.save_pretrained(student_dir)
    processor.save_pretrained(student_dir)
    return student, (sum(losses) / len(losses) if losses else None)
