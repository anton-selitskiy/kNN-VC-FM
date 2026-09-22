# One-Step Voice Conversion by Learning kNN Transport in WavLM Space

This repository contains the research code, evaluation utilities, saved metric
outputs, and project page for kNN-FM-VC. The method learns a target-conditioned
flow field in WavLM embedding space to approximate the frame-level transport
induced by kNN-VC. It replaces explicit nearest-neighbor matching at inference
time with a single learned network and supports few-step, including one-step,
conversion.

## Method overview

Training examples are constructed from source and target-speaker WavLM-Large
embeddings. For every source frame, the dataset finds the four target frames
with the highest cosine similarity and averages them to obtain a pseudo-target.
The model combines:

- cross-attention from the evolving source sequence to a target-speaker
  utterance;
- a masked temporal mean of the target sequence for global speaker
  conditioning;
- sinusoidal time embeddings and FiLM-conditioned residual blocks; and
- a flow-matching objective with an auxiliary speaker-similarity term.

The project evaluates Schrödinger Bridge, Gaussian Bridge, and Gaussian Tube
paths at 1, 5, and 50 integration steps, alongside kNN, kDOT, MKL, FreeVC, and
the expanded Phoneme Hallucinator baseline.

## Repository layout

```text
training_models/                  Flow-matching dataset, model, and training loop
preprocessing.py                  LibriSpeech-to-WavLM feature extraction
evaluation/                       WER, FAD, speaker-similarity, and UTMOS utilities
evaluation/results/               Tracked metric summaries and per-pair outputs
outputs/                           Tracked run manifests and selected raw results
```

## Environment

The recorded experiment environment used Python 3.10 with PyTorch 2.0.0,
TorchAudio 2.0.1, and CUDA 11.8. `requirements.txt` is an exported environment
inventory rather than a minimal pip requirements file. Create an isolated
environment and install PyTorch/TorchAudio versions appropriate for the target
CUDA runtime before installing the remaining packages.

Feature extraction expects the kNN-VC implementation to be available as
`knn-vc/` at the repository root because `preprocessing.py` imports its
`hubconf`, matcher, and vocoder utilities directly.

## Data preprocessing

`preprocessing.py` recursively reads LibriSpeech FLAC files, converts them to
mono 16 kHz audio when necessary, extracts WavLM-Large features through kNN-VC,
and saves one half-precision tensor per utterance while preserving the dataset
directory structure.

The input and output roots are currently defined near the top of the script:

```python
audio_root = Path("LibriSpeech/train-clean-100")
output_root = Path("LibriSpeech_wavlm/train-clean-100")
```

After setting those paths and preparing the kNN-VC dependency, run:

```bash
python preprocessing.py
```

## Training

The current training entry point is:

```bash
python training_models/FM_knn_wavlm_emb_attn.py
```

Before running it, update the embedding root in its `__main__` block and choose
whether training should resume. The checked-in entry point currently assumes a
CUDA device, uses a machine-specific `/scratch/aselitsk/...` path, and calls
`train_fm(..., contin=True)`, which requires
`best_fm_knn_wavlm_emb_model_attn.pt` to exist. These values should be changed
for a fresh run.

The script currently uses a batch size of 64, 28 data-loader workers, AdamW,
mixed precision, gradient clipping, and best-loss checkpointing. It is the
current single training implementation in this checkout; the reported path
variants and 1/5/50-step inference settings are not yet exposed through one
reproducible configuration or command-line interface.

## Evaluation

The repository includes the original WER, FAD, and speaker-similarity scripts,
plus two reproducible evaluators added for the paper results:

- `evaluation/eval_SIM_SEED.py` computes target- or source-speaker cosine
  similarity with the official SEED-TTS WavLM speaker verifier. It batches only
  equal-length waveforms, caches embeddings, and saves per-pair and summary
  CSVs.
- `evaluation/eval_UTMOSv2.py` computes UTMOSv2 scores, supports resumable
  method-level evaluation, and saves the input manifest, per-pair scores, and
  summary statistics.

Both evaluators report the sample standard deviation and a two-sided 95%
Student-t confidence interval across utterances. Installation requirements,
method-directory mappings, and complete example commands are documented in
[`evaluation/README_metrics.md`](evaluation/README_metrics.md).

### Tracked evaluation outputs

- `evaluation/results/seed_tts_ssim/` contains target- and source-reference
  SEED-TTS similarity scores, method summaries, and paired comparisons for
  1,310 conversion pairs per evaluated method.
- `outputs/utmosv2_ci_full/` contains the tracked input manifest and available
  per-pair UTMOSv2 scores.
- `outputs/seed_tts_ssim/run_metadata_source.json` records the source-reference
  similarity run configuration.

Large reusable embedding caches and local smoke-test outputs are intentionally
not part of the tracked result set.

## Reproducibility status

The evaluation scripts and their tracked outputs preserve substantially more
provenance than the original repository. Full end-to-end reproduction still
requires external model checkpoints, the complete converted-audio tree, and
the exact inference code/configuration used for every probability path and
solver-step setting. Do not treat the website audio bundle as a replacement for
the full evaluation corpus.
