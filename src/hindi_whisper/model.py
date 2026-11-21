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
    ensure_generation(student, merged_teacher.generation_config.language or "hindi")
    return student


def ensure_generation(model, language: str, base_model: str = "openai/whisper-small"):
    """Whisper models built from a config lack lang_to_id and reject generate(language=...)."""
    from transformers import GenerationConfig

    lang_to_id = getattr(model.generation_config, "lang_to_id", None)
    task_to_id = getattr(model.generation_config, "task_to_id", None)
    if not lang_to_id or not task_to_id:
        model.generation_config = GenerationConfig.from_pretrained(base_model)
    model.config.use_cache = True
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = language
    model.generation_config.task = "transcribe"


def load_merged_adapter(cfg: ExperimentConfig, adapter_dir: str):
    """Load a finished LoRA adapter and merge it into the base Whisper weights."""
    from peft import PeftModel

    base = WhisperForConditionalGeneration.from_pretrained(cfg.model_name)
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    base.config.use_cache = False
    base.generation_config.language = cfg.language
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.config.forced_decoder_ids = None
    merged.config.use_cache = False
    merged.generation_config.language = cfg.language
    merged.generation_config.task = "transcribe"
    merged.generation_config.forced_decoder_ids = None
    return merged


def distillation_loss(student_logits, teacher_logits, labels, kl_weight: float, temperature: float):
    """Float32 CE on labels plus temperature-scaled KL on non-padding positions."""
    import torch.nn.functional as F

    student_log = student_logits[:, :-1].float()
    teacher_log = teacher_logits[:, :-1].float()
    lab = labels[:, 1:]
    steps = min(student_log.size(1), teacher_log.size(1), lab.size(1))
    student_log = student_log[:, :steps]
    teacher_log = teacher_log[:, :steps]
    lab = lab[:, :steps]
    mask = lab.ne(-100)
    if int(mask.sum()) == 0:
        return student_log.sum() * 0.0
    ce = F.cross_entropy(student_log[mask], lab[mask])
    token_kl = F.kl_div(
        F.log_softmax(student_log / temperature, dim=-1),
        F.softmax(teacher_log / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    kl = (token_kl * mask).sum() / mask.sum().clamp(min=1)
    return ce + kl_weight * kl * (temperature ** 2)
