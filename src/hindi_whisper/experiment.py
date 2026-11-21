"""Run one experiment config end to end and write metrics.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hindi_whisper.config import ExperimentConfig
from hindi_whisper.data import prepare_datasets
from hindi_whisper.distill import distill
from hindi_whisper.evaluate import word_error_rate
from hindi_whisper.model import load_lora_teacher, load_merged_adapter, load_processor
from hindi_whisper.streaming import concatenate_clips, local_agreement_stream
from hindi_whisper.train import train_lora


def run(cfg: ExperimentConfig, adapter_dir: str | None = None, student_dir: str | None = None) -> dict:
    import torch

    torch.manual_seed(cfg.seed)
    if os.path.isdir("/kaggle/working") and cfg.output_dir == "outputs":
        cfg.output_dir = "/kaggle/working/outputs"
    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    processor = load_processor(cfg)
    batch = prepare_datasets(cfg, processor)
    print(f"train clips {len(batch.train)} eval clips {len(batch.eval)}")

    if student_dir:
        if not adapter_dir:
            raise ValueError("--from-student requires --from-adapter so the teacher can be scored too.")
        from transformers import WhisperForConditionalGeneration

        print(f"loading saved LoRA adapter {adapter_dir}", flush=True)
        merged = load_merged_adapter(cfg, adapter_dir)
        print(f"loading saved student {student_dir}", flush=True)
        student = WhisperForConditionalGeneration.from_pretrained(student_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        merged.to(device)
        student.to(device)
        mean_loss = None
    elif adapter_dir:
        print(f"loading saved LoRA adapter {adapter_dir}", flush=True)
        merged = load_merged_adapter(cfg, adapter_dir)
        student, mean_loss = distill(cfg, merged, processor, batch.train, run_dir)
    else:
        teacher = load_lora_teacher(cfg)
        teacher = train_lora(cfg, teacher, processor, batch.train, run_dir)
        merged = teacher.merge_and_unload()
        student, mean_loss = distill(cfg, merged, processor, batch.train, run_dir)

    wer_teacher = word_error_rate(merged, processor, batch.eval, cfg.eval_n, cfg.language)
    wer_student = word_error_rate(student, processor, batch.eval, cfg.eval_n, cfg.language)
    audio = concatenate_clips(batch.eval_audio, cfg.stream_clips)
    hypothesis, mean_latency, commits = local_agreement_stream(
        student, processor, audio, cfg.language, chunk_s=cfg.stream_chunk_s
    )

    dataset_name = (
        "google/fleurs hi_in" if cfg.dataset == "fleurs" else "mozilla-foundation/common_voice_11_0 hi"
    )
    metrics = {
        "run_name": cfg.run_name,
        "dataset": dataset_name,
        "speakers": "Hindi",
        "teacher": f"{cfg.model_name} + LoRA r={cfg.lora_r} on q_proj,v_proj",
        "student": "Distil-Whisper layout: frozen copied encoder, decoder = teacher first+last layers",
        "papers": [
            "Hu et al. 2021 LoRA",
            "Gandhi, von Platen, Rush 2023 Distil-Whisper arXiv:2311.00430",
            "Machacek, Dabre, Bojar 2023 LocalAgreement-2 on Whisper arXiv:2307.14743",
        ],
        "train_clips": len(batch.train),
        "eval_clips": len(batch.eval),
        "lora_steps": cfg.lora_steps,
        "lora_source": adapter_dir or "trained_this_run",
        "distill_steps": cfg.distill_steps,
        "distill_mean_loss": mean_loss,
        "distill_logged_every_10_steps": _logged_distill_steps(run_dir),
        "wer_teacher_lora": wer_teacher,
        "wer_student": wer_student,
        "stream_chunk_s": cfg.stream_chunk_s,
        "stream_commits": commits,
        "stream_mean_commit_latency_s": mean_latency,
        "stream_hypothesis_preview": hypothesis[:300],
        "config": cfg.to_dict(),
        "note": "Numbers are from this run only. FLEURS is read speech, not 8 kHz telephony.",
    }
    path = os.path.join(run_dir, "metrics.json")
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("wer_teacher_lora", "wer_student", "stream_mean_commit_latency_s")}, indent=2))
    print("wrote", path)
    return metrics


def _logged_distill_steps(run_dir: str):
    """Losses printed every 10 steps. This is not the mean over every step."""
    path = os.path.join(run_dir, "distill_trace.json")
    if not os.path.exists(path):
        return None
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    return trace.get("logged_every_10_steps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and distill Whisper on Hindi speech.")
    parser.add_argument("--config", required=True, help="Path to an experiments/*.yaml file.")
    parser.add_argument(
        "--from-adapter",
        default="",
        help="Skip LoRA training and load this saved adapter directory.",
    )
    parser.add_argument(
        "--from-student",
        default="",
        help="Skip distillation and load this saved student directory.",
    )
    args = parser.parse_args()
    run(
        ExperimentConfig.from_yaml(args.config),
        adapter_dir=args.from_adapter or None,
        student_dir=args.from_student or None,
    )


if __name__ == "__main__":
    main()
