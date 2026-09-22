# SEED-TTS similarity and UTMOSv2 evaluation

These evaluators reconstruct converted-audio paths from the LibriSpeech pair
CSV and save both utterance-level scores and method-level summaries. Summary
files contain the sample standard deviation and a two-sided 95% Student-t
confidence interval across utterances.

## SEED-TTS WavLM speaker similarity

`eval_SIM_SEED.py` uses the official SEED-TTS WavLM-Large
speaker-verification checkpoint and cosine similarity between each converted
utterance and its paired target reference. Audio must be 16 kHz. Multi-channel
audio is averaged to mono. Exact-length batching prevents padded frames from
affecting the verifier's statistics pooling.

Install the s3prl revision pinned by SEED-TTS evaluation and provide a local
clone of <https://github.com/BytedanceSpeech/seed-tts-eval> plus its released
`wavlm_large_finetune.pth` checkpoint.

```bash
python evaluation/eval_SIM_SEED.py \
  --eval-root /path/to/eval \
  --pairs-csv evaluation/librispeech_test_pairs_text.csv \
  --seed-tts-eval-root /path/to/seed-tts-eval \
  --checkpoint /path/to/wavlm_large_finetune.pth \
  --output-dir outputs/seed_tts_ssim \
  --methods our_SB_1 our_SB_5 our_SB our_1 our_5 our \
    our_no_reg_1 our_no_reg_5 our_no_reg kNN kDOT MKL FreeVC PhH_exp
```

The embedding cache is saved every 25 batches and reused when the command is
restarted.

To test residual source-speaker identity with the same verifier and cached
converted embeddings, rerun in the same output directory with
`--reference source`. Source-reference outputs are written under distinct
`seed_tts_source_ssim_*` names and do not overwrite target-similarity results.

The reported evaluation outputs are tracked in
`evaluation/results/seed_tts_ssim/`. This directory contains utterance-level
target and source scores, their method-level summaries, and the paired
target--source comparison. The reusable embedding cache is intentionally not
tracked.

## UTMOSv2

`eval_UTMOSv2.py` uses the official UTMOSv2 `fusion_stage3` predictor, fold 0,
and seed 42. The default is one prediction repetition without silence removal.
The random state is reset for each method so paired systems share the same crop
randomness stream. Completed methods are saved immediately and skipped when a
run is resumed.

Install <https://github.com/sarulab-speech/UTMOSv2> and download the released
fold-0, seed-42 checkpoint.

```bash
python evaluation/eval_UTMOSv2.py \
  --eval-root /path/to/eval \
  --pairs-csv evaluation/librispeech_test_pairs_text.csv \
  --checkpoint /path/to/utmosv2_fold0_s42_best_model.pth \
  --output-dir outputs/utmosv2_ci \
  --methods our_SB_1 our_SB_5 our_SB our_1 our_5 our \
    our_no_reg_1 our_no_reg_5 our_no_reg kNN kDOT MKL FreeVC PhH_exp
```

Use `--num-repetitions` to average additional stochastic crops or
`--remove-silence` only when intentionally defining a different protocol.
