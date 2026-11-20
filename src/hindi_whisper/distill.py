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
    if device == "cuda":
        merged_teacher.half()
        student.half()
    merged_teacher.to(device)
    student.to(device)

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
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            teacher_logits = merged_teacher(
                input_features=batch["input_features"], labels=batch["labels"]
            ).logits
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            student_out = student(
                input_features=batch["input_features"], labels=batch["labels"]
            )
        loss = distillation_loss(
            student_out.logits,
            teacher_logits,
            batch["labels"],
            student_out.loss,
            cfg.kl_weight,
            cfg.temperature,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0:
            print(f"distill step {step} loss {losses[-1]:.3f}")

    student_dir = os.path.join(output_dir, "student")
    student.save_pretrained(student_dir)
    processor.save_pretrained(student_dir)
    return student, (sum(losses) / len(losses) if losses else None)
