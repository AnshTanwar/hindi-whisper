# Hindi Whisper

LoRA fine-tune of Whisper-small on public Hindi speech, a 2-layer distilled student, and LocalAgreement-2 streaming. The measured WER and latency land in `outputs/<run>/metrics.json` after a run. This repo does not ship results from someone else's paper.

## Methods

- **Fine-tune.** LoRA on `q_proj` and `v_proj` (Hu et al., 2021).
- **Distil.** Encoder copied from the teacher and frozen. Decoder keeps the teacher's first and last layers. Loss is cross-entropy plus KL. Layout from Distil-Whisper (Gandhi, von Platen, Rush, 2023), [arXiv:2311.00430](https://arxiv.org/abs/2311.00430).
- **Stream.** A prefix is committed only when two consecutive chunks agree. LocalAgreement-2 as used by the IWSLT 2022 winning CUNI-KIT system and applied to Whisper by Macháček, Dabre, and Bojar (2023), [arXiv:2307.14743](https://arxiv.org/abs/2307.14743).

## Data

| Config | Corpus | Notes |
|---|---|---|
| `experiments/fleurs_small.yaml` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) `hi_in` | Public, no token. Read speech, not phone audio. Short T4 run. |
| `experiments/fleurs_t4.yaml` | same | Longer schedule after the small run works. |
| `experiments/common_voice_hindi.yaml` | Common Voice 11 Hindi | Gated. Set `HF_TOKEN`. |

## Run

```bash
pip install -e .
hindi-whisper --config experiments/fleurs_small.yaml
```

GPU is expected (Kaggle T4 or a local CUDA card). CPU will run and be slow.

On Kaggle: GPU T4, Internet on, then run `notebooks/kaggle_run.ipynb`.

## Layout

- `src/hindi_whisper/data.py` — load and filter clips
- `src/hindi_whisper/train.py` — LoRA training
- `src/hindi_whisper/distill.py` — student training
- `src/hindi_whisper/evaluate.py` — held-out WER
- `src/hindi_whisper/streaming.py` — chunked decode
- `experiments/` — one YAML per run

## Tests

```bash
pip install pytest
pytest
```

The tests cover prefix agreement and config loading. They do not download Whisper or FLEURS.
