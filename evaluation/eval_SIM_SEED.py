"""Evaluate speaker similarity with the SEED-TTS WavLM verifier."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from scipy.stats import t as student_t
from tqdm import tqdm


METHOD_DIRS = {
    "our_SB_1": "our_SB_1",
    "our_SB_5": "our_SB_5",
    "our_SB": "our_SB",
    "our_1": "our_1",
    "our_5": "our_5",
    "our": "our",
    "our_no_reg_1": "our_no_reg_1",
    "our_no_reg_5": "our_no_reg_5",
    "our_no_reg": "our_no_reg",
    "kNN": "kNN",
    "kDOT": "kDOT",
    "MKL": "MKL",
    "MeanVC": "MeanVC",
    "FreeVC": "freevc",
    "PhH_exp": "Phoneme_Hallucinator/expanded",
    "PhH_unexp": "Phoneme_Hallucinator/unexpanded",
}

DOUBLE_UNDERSCORE_METHODS = {
    "our_SB_1",
    "our_SB_5",
    "our_SB",
    "our_1",
    "our_5",
    "our",
    "our_no_reg_1",
    "our_no_reg_5",
    "our_no_reg",
    "kNN",
    "kDOT",
    "MKL",
}


def import_seed_model(seed_tts_eval_root: Path) -> type[torch.nn.Module]:
    """Import the verifier without requiring SEED-TTS to be installed."""
    verification_dir = (
        seed_tts_eval_root
        / "thirdparty/UniSpeech/downstreams/speaker_verification"
    )
    if not (verification_dir / "models/ecapa_tdnn.py").is_file():
        raise FileNotFoundError(
            f"Invalid SEED-TTS evaluator root: {seed_tts_eval_root}"
        )
    sys.path.insert(0, str(verification_dir))
    from models.ecapa_tdnn import ECAPA_TDNN_SMALL

    return ECAPA_TDNN_SMALL


def import_wavlm() -> tuple[type[torch.nn.Module], type]:
    """Import only the pinned s3prl WavLM implementation."""
    spec = importlib.util.find_spec("s3prl")
    if spec is None or spec.origin is None:
        raise ImportError(
            "Install the SEED-TTS-pinned s3prl revision before evaluation"
        )
    s3prl_dir = Path(spec.origin).parent
    for package, path in (
        ("s3prl", s3prl_dir),
        ("s3prl.upstream", s3prl_dir / "upstream"),
        ("s3prl.upstream.wavlm", s3prl_dir / "upstream/wavlm"),
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    wavlm_module = importlib.import_module("s3prl.upstream.wavlm.WavLM")
    return wavlm_module.WavLM, wavlm_module.WavLMConfig


class LocalWavLMLarge(torch.nn.Module):
    """s3prl-compatible WavLM-Large constructed without a network download."""

    def __init__(self, wavlm_class: type, config_class: type) -> None:
        super().__init__()
        self.cfg = config_class(
            {
                "extractor_mode": "layer_norm",
                "encoder_layers": 24,
                "encoder_embed_dim": 1024,
                "encoder_ffn_embed_dim": 4096,
                "encoder_attention_heads": 16,
                "layer_norm_first": True,
                "normalize": True,
                "encoder_layerdrop": 0.1,
                "relative_position_embedding": True,
                "num_buckets": 320,
                "max_distance": 800,
                "gru_rel_pos": True,
            }
        )
        self.model = wavlm_class(self.cfg)

    def forward(self, wavs: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        if self.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
        device = wavs[0].device
        lengths = torch.tensor([len(wav) for wav in wavs], device=device)
        padding_mask = ~torch.lt(
            torch.arange(int(lengths.max()), device=device).unsqueeze(0),
            lengths.unsqueeze(1),
        )
        padded = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        final, layer_results = self.model.extract_features(
            padded,
            padding_mask=padding_mask,
            mask=False,
            output_layer=self.cfg.encoder_layers,
            ret_layer_results=True,
        )[0]
        hidden_states = [state.transpose(0, 1) for state, _ in layer_results[:-1]]
        hidden_states.append(final)
        return {"hidden_states": hidden_states}


def load_model(
    seed_tts_eval_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    ecapa_class = import_seed_model(seed_tts_eval_root)
    wavlm_class, config_class = import_wavlm()
    original_load = torch.hub.load
    torch.hub.load = lambda *args, **kwargs: LocalWavLMLarge(
        wavlm_class, config_class
    )
    try:
        model = ecapa_class(feat_dim=1024, feat_type="wavlm_large")
    finally:
        torch.hub.load = original_load

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_unexpected = {"loss_calculator.projection.weight"}
    if (
        incompatible.missing_keys
        or set(incompatible.unexpected_keys) != expected_unexpected
    ):
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def converted_path(eval_root: Path, method: str, row: pd.Series) -> Path:
    source = Path(row["source_file"]).stem
    target = Path(row["target_file"]).stem
    separator = "__to__" if method in DOUBLE_UNDERSCORE_METHODS else "_to_"
    return eval_root / METHOD_DIRS[method] / f"{source}{separator}{target}.wav"


def reference_path(eval_root: Path, row: pd.Series, reference: str) -> Path:
    source = Path(row["source_file"]).stem
    target = Path(row["target_file"]).stem
    return (
        eval_root
        / f"freevc/{reference}"
        / f"{source}_to_{target}_{reference}.flac"
    )


def build_records(
    eval_root: Path,
    pairs: pd.DataFrame,
    methods: list[str],
    reference: str,
) -> pd.DataFrame:
    records = []
    for row_index, row in pairs.iterrows():
        reference_file = reference_path(eval_root, row, reference)
        for method in methods:
            converted = converted_path(eval_root, method, row)
            records.append(
                {
                    "row_index": int(row_index),
                    "method": method,
                    "source_file": row["source_file"],
                    "target_file": row["target_file"],
                    "converted_path": str(converted.resolve()),
                    "reference": reference,
                    "reference_path": str(reference_file.resolve()),
                    "available": converted.is_file() and reference_file.is_file(),
                }
            )
    return pd.DataFrame(records)


def load_mono_16k(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz, found {sample_rate} Hz: {path}")
    waveform = waveform.mean(dim=0)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Non-finite samples: {path}")
    return waveform


def embed_paths(
    model: torch.nn.Module,
    paths: list[Path],
    batch_size: int,
    cache_path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    embeddings = (
        torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_path.is_file()
        else {}
    )
    pending = [path for path in paths if str(path) not in embeddings]
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in tqdm(pending, desc="Inspecting audio", unit="file"):
        info = sf.info(path)
        if info.samplerate != 16_000:
            raise ValueError(
                f"Expected 16 kHz, found {info.samplerate} Hz: {path}"
            )
        grouped[info.frames].append(path)

    batches = []
    for frames in sorted(grouped):
        group = grouped[frames]
        batches.extend(
            group[index : index + batch_size]
            for index in range(0, len(group), batch_size)
        )
    for batch_index, batch in enumerate(
        tqdm(batches, desc="SEED SSIM embeddings", unit="batch"), 1
    ):
        waveforms = [load_mono_16k(path) for path in batch]
        if len({wav.numel() for wav in waveforms}) != 1:
            raise RuntimeError("Exact-length batching invariant violated")
        with torch.inference_mode():
            encoded = model(torch.stack(waveforms).to(device))
        encoded = F.normalize(encoded, dim=-1).cpu()
        embeddings.update(
            {
                str(path): embedding
                for path, embedding in zip(batch, encoded, strict=True)
            }
        )
        if batch_index % 25 == 0:
            torch.save(embeddings, cache_path)
    torch.save(embeddings, cache_path)
    return embeddings


def summarize(records: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows = []
    for method, group in records.groupby("method", sort=False):
        values = group[score_column]
        n = int(len(values))
        mean = float(values.mean())
        sample_std = float(values.std(ddof=1))
        ci_half = float(student_t.ppf(0.975, n - 1) * sample_std / n**0.5)
        rows.append(
            {
                "method": method,
                "n": n,
                "mean": mean,
                "sample_std": sample_std,
                "ci95_half_width": ci_half,
                "ci95_lower": mean - ci_half,
                "ci95_upper": mean + ci_half,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--seed-tts-eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--reference",
        choices=("target", "source"),
        default="target",
        help="Reference side used for cosine similarity.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_DIRS),
        default=list(METHOD_DIRS),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(
        args.eval_root,
        pd.read_csv(args.pairs_csv),
        args.methods,
        args.reference,
    )
    missing = records.loc[~records["available"], "converted_path"]
    if not missing.empty:
        raise FileNotFoundError(
            f"Missing {len(missing)} pairs; first: {missing.iloc[0]}"
        )

    paths = sorted(
        {
            Path(path)
            for column in ("converted_path", "reference_path")
            for path in records[column]
        },
        key=str,
    )
    model = load_model(args.seed_tts_eval_root, args.checkpoint, device)
    embeddings = embed_paths(
        model,
        paths,
        args.batch_size,
        args.output_dir / "embedding_cache.pt",
        device,
    )
    score_column = (
        "seed_tts_wavlm_ssim"
        if args.reference == "target"
        else "seed_tts_wavlm_source_ssim"
    )
    output_prefix = (
        "seed_tts_ssim"
        if args.reference == "target"
        else "seed_tts_source_ssim"
    )
    records[score_column] = [
        float(
            F.cosine_similarity(
                embeddings[converted], embeddings[reference], dim=0
            )
        )
        for converted, reference in zip(
            records["converted_path"], records["reference_path"], strict=True
        )
    ]
    records.to_csv(args.output_dir / f"{output_prefix}_per_pair.csv", index=False)

    summary = summarize(records, score_column)
    summary.to_csv(args.output_dir / f"{output_prefix}_summary.csv", index=False)
    metadata_name = (
        "run_metadata.json"
        if args.reference == "target"
        else "run_metadata_source.json"
    )
    (args.output_dir / metadata_name).write_text(
        json.dumps(
            {
                "metric": "SEED-TTS WavLM-Large speaker-verification cosine",
                "checkpoint": str(args.checkpoint.resolve()),
                "reference": f"paired {args.reference} utterance",
                "methods": args.methods,
                "sample_rate_hz": 16000,
                "batching": "only exactly equal-length waveforms",
                "batch_size": args.batch_size,
                "device": str(device),
                "ci": "two-sided Student-t 95% CI across utterance scores",
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
