"""Teacher LoRA and the 2-layer Distil-Whisper student."""

from __future__ import annotations

import copy

from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from hindi_whisper.config import ExperimentConfig


def load_processor(cfg: ExperimentConfig) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")


def load_lora_teacher(cfg: ExperimentConfig):
    teacher = WhisperForConditionalGeneration.from_pretrained(cfg.model_name)
    teacher.config.forced_decoder_ids = None
    teacher.config.suppress_tokens = []
    teacher.config.use_cache = False
    teacher.generation_config.language = cfg.language
    teacher.generation_config.task = "transcribe"
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    teacher = get_peft_model(teacher, lora)
    teacher.enable_input_require_grads()
    return teacher


def build_distilled_student(merged_teacher):
    """Distil-Whisper layout: frozen copied encoder, first and last decoder layers.

    Gandhi, von Platen, Rush, 2023. https://arxiv.org/abs/2311.00430
    """
    if not hasattr(merged_teacher.config, "decoder_layers"):
        raise RuntimeError("This transformers build has no WhisperConfig.decoder_layers.")

    src_layers = merged_teacher.model.decoder.layers
    cfg = copy.deepcopy(merged_teacher.config)
    cfg.decoder_layers = 2
    student = WhisperForConditionalGeneration(cfg)
    student.model.encoder.load_state_dict(merged_teacher.model.encoder.state_dict())
    student.model.decoder.layers[0].load_state_dict(src_layers[0].state_dict())
    student.model.decoder.layers[1].load_state_dict(src_layers[-1].state_dict())
    student.model.decoder.embed_tokens.load_state_dict(merged_teacher.model.decoder.embed_tokens.state_dict())
    student.model.decoder.embed_positions.load_state_dict(
        merged_teacher.model.decoder.embed_positions.state_dict()
    )
    student.model.decoder.layer_norm.load_state_dict(merged_teacher.model.decoder.layer_norm.state_dict())
    student.proj_out.load_state_dict(merged_teacher.proj_out.state_dict())
    for param in student.model.encoder.parameters():
        param.requires_grad = False
    student.config.use_cache = False
    student.generation_config.language = merged_teacher.generation_config.language
    student.generation_config.task = "transcribe"
    return student


def distillation_loss(student_logits, teacher_logits, labels, ce_loss, kl_weight: float, temperature: float):
    """CE on labels plus temperature-scaled KL on non-padding positions."""
    import torch
    import torch.nn.functional as F

    lab = labels[:, 1:]
    mask = lab.ne(-100).unsqueeze(-1)
    student_log = student_logits[:, :-1].float()
    teacher_log = teacher_logits[:, :-1].float()
    student_flat = student_log.masked_select(mask.expand_as(student_log)).view(-1, student_log.size(-1))
    teacher_flat = teacher_log.masked_select(mask.expand_as(teacher_log)).view(-1, teacher_log.size(-1))
    if student_flat.numel() == 0:
        return ce_loss
    kl = F.kl_div(
        F.log_softmax(student_flat / temperature, dim=-1),
        F.softmax(teacher_flat / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return ce_loss.float() + kl_weight * kl
