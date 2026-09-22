"""Evaluate converted speech with the official pretrained UTMOSv2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import utmosv2
from scipy.stats import t as student_t


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


def converted_path(eval_root: Path, method: str, row: pd.Series) -> Path:
    source = Path(row["source_file"]).stem
    target = Path(row["target_file"]).stem
    separator = "__to__" if method in DOUBLE_UNDERSCORE_METHODS else "_to_"
    return eval_root / METHOD_DIRS[method] / f"{source}{separator}{target}.wav"


def build_records(
    eval_root: Path,
    pairs: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    records = []
    for row_index, row in pairs.iterrows():
        for method in methods:
            path = converted_path(eval_root, method, row)
            records.append(
                {
                    "row_index": int(row_index),
                    "method": method,
                    "source_file": row["source_file"],
                    "target_file": row["target_file"],
                    "converted_path": str(path.resolve()),
                    "available": path.is_file(),
                }
            )
    return pd.DataFrame(records)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in scored.groupby("method", sort=False):
        values = group["utmosv2"]
        n = int(len(values))
        mean = float(values.mean())
        sample_std = float(values.std(ddof=1))
        ci_half = float(student_t.ppf(0.975, n - 1) * sample_std / np.sqrt(n))
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-repetitions", type=int, default=1)
    parser.add_argument("--remove-silence", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_DIRS),
        default=list(METHOD_DIRS),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_repetitions < 1:
        raise ValueError("--num-repetitions must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs_csv)
    records = build_records(args.eval_root, pairs, args.methods)
    missing = records.loc[~records["available"], "converted_path"]
    if not missing.empty:
        raise FileNotFoundError(
            f"Missing {len(missing)} converted files; first: {missing.iloc[0]}"
        )

    model = utmosv2.create_model(
        pretrained=True,
        fold=0,
        checkpoint_path=args.checkpoint,
        seed=42,
        device=args.device,
    )

    per_pair_path = args.output_dir / "utmosv2_per_pair.csv"
    completed = (
        pd.read_csv(per_pair_path) if per_pair_path.is_file() else pd.DataFrame()
    )
    if not completed.empty:
        unexpected = set(completed["method"]) - set(args.methods)
        if unexpected:
            raise ValueError(
                "Existing per-pair output contains methods not requested in this run: "
                f"{sorted(unexpected)}"
            )
    completed_methods = set(completed["method"]) if not completed.empty else set()
    parts = [completed] if not completed.empty else []

    for method, group in records.groupby("method", sort=False):
        if method in completed_methods:
            expected_rows = len(group)
            actual_rows = int(completed["method"].eq(method).sum())
            if actual_rows != expected_rows:
                raise ValueError(
                    f"Incomplete saved method {method}: "
                    f"expected {expected_rows}, found {actual_rows}"
                )
            print(f"Skipping completed method: {method}")
            continue

        paths = [Path(path) for path in group["converted_path"]]
        directories = {path.parent for path in paths}
        if len(directories) != 1:
            raise ValueError(f"Expected one directory for {method}: {directories}")
        if len(set(paths)) != len(paths):
            raise ValueError(f"Duplicate converted paths for {method}")

        # Reset per method so paired systems share the crop-randomness stream.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        predictions = model.predict(
            input_dir=directories.pop(),
            val_list=[path.name for path in paths],
            predict_dataset="sarulab",
            device=args.device,
            num_workers=0,
            batch_size=args.batch_size,
            num_repetitions=args.num_repetitions,
            remove_silent_section=args.remove_silence,
            verbose=True,
        )
        score_by_path = {
            str(Path(item["file_path"]).resolve()): float(item["predicted_mos"])
            for item in predictions
        }
        missing_predictions = {
            str(path.resolve()) for path in paths
        } - score_by_path.keys()
        if missing_predictions:
            first = sorted(missing_predictions)[0]
            raise RuntimeError(
                f"UTMOSv2 returned no score for {len(missing_predictions)} files; "
                f"first: {first}"
            )

        part = group.copy()
        part["utmosv2"] = [score_by_path[str(path.resolve())] for path in paths]
        parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(per_pair_path, index=False)

    if not parts:
        raise RuntimeError("No UTMOSv2 predictions were produced")
    scored = pd.concat(parts, ignore_index=True)
    summary = summarize(scored)
    summary.to_csv(args.output_dir / "utmosv2_summary.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "metric": "UTMOSv2 fusion_stage3 quick prediction",
                "checkpoint": str(args.checkpoint.resolve()),
                "fold": 0,
                "seed": args.seed,
                "num_repetitions": args.num_repetitions,
                "remove_silent_section": args.remove_silence,
                "predict_dataset": "sarulab",
                "methods": args.methods,
                "device": args.device,
                "batch_size": args.batch_size,
                "ci": "two-sided Student-t 95% CI across utterance scores",
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
