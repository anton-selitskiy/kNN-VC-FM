#!/usr/bin/env python3
"""Curate a small, reproducible listening set for the project page."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


EXAMPLE_INDICES = (2, 8, 13, 15, 19)
METHODS = {
    "sb-1": ("our_SB_1", "__to__"),
    "sb-5": ("our_SB_5", "__to__"),
    "sb-50": ("our_SB", "__to__"),
    "gb-1": ("our_1", "__to__"),
    "gb-5": ("our_5", "__to__"),
    "gb-50": ("our", "__to__"),
    "tube-1": ("our_no_reg_1", "__to__"),
    "tube-5": ("our_no_reg_5", "__to__"),
    "tube-50": ("our_no_reg", "__to__"),
    "knn": ("kNN", "__to__"),
    "kdot": ("kDOT", "__to__"),
    "mkl": ("MKL", "__to__"),
    "freevc": ("freevc", "_to_"),
    "ph-expanded": ("Phoneme_Hallucinator/expanded", "_to_"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Directory containing the per-method evaluation audio folders.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("evaluation/librispeech_test_pairs_text_w.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/audio"),
    )
    return parser.parse_args()


def copy_audio(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    with args.pairs_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for existing_example in args.output_dir.glob("example-*"):
        if existing_example.is_dir():
            shutil.rmtree(existing_example)

    manifest = []
    for example_number, row_index in enumerate(EXAMPLE_INDICES, start=1):
        row = rows[row_index]
        source_id = Path(row["source_file"]).stem
        target_id = Path(row["target_file"]).stem
        stem = f"{source_id}_to_{target_id}"
        example_dir = args.output_dir / f"example-{example_number}"

        copy_audio(
            args.eval_root / "freevc" / "source" / f"{stem}_source.flac",
            example_dir / "source.flac",
        )
        copy_audio(
            args.eval_root / "freevc" / "target" / f"{stem}_target.flac",
            example_dir / "target.flac",
        )

        files = {"source": "source.flac", "target": "target.flac"}
        for method_id, (directory, separator) in METHODS.items():
            filename = f"{source_id}{separator}{target_id}.wav"
            copy_audio(
                args.eval_root / directory / filename,
                example_dir / f"{method_id}.wav",
            )
            files[method_id] = f"{method_id}.wav"

        manifest.append(
            {
                "id": f"example-{example_number}",
                "row_index": row_index,
                "source_id": source_id,
                "target_id": target_id,
                "transcript": row["text"].capitalize(),
                "files": files,
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(manifest)} examples in {args.output_dir}")


if __name__ == "__main__":
    main()
