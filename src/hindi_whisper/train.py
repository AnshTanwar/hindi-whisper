"""LoRA fine-tune of Whisper on the prepared Hindi set."""

from __future__ import annotations

import os

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from hindi_whisper.config import ExperimentConfig
from hindi_whisper.data import SpeechCollator


def train_lora(cfg: ExperimentConfig, teacher, processor, train_ds, output_dir: str):
    use_cuda = _cuda()
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, "lora"),
        per_device_train_batch_size=cfg.lora_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.lora_steps,
        fp16=use_cuda,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        remove_unused_columns=False,
        predict_with_generate=False,
        gradient_checkpointing=True,
        report_to=[],
        seed=cfg.seed,
    )
    kwargs = dict(
        model=teacher,
        args=args,
        train_dataset=train_ds,
        data_collator=SpeechCollator(processor),
    )
    try:
        trainer = Seq2SeqTrainer(**kwargs, processing_class=processor)
    except TypeError:
        trainer = Seq2SeqTrainer(**kwargs, tokenizer=processor)
    trainer.train()
    adapter_dir = os.path.join(output_dir, "lora-adapter")
    teacher.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    return teacher


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()
