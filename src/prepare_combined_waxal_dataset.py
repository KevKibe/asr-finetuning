#!/usr/bin/env python3
"""Build a combined multi-language Waxal dataset with all splits merged into train."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq


if len(sys.argv) != 5:
    print(
        "Usage: python prepare_combined_waxal_dataset.py "
        "<waxal_sna_root> <waxal_lin_root> <waxal_lug_root> <combined_root>"
    )
    sys.exit(1)

waxal_roots = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])]
combined_root = Path(sys.argv[4])


def copy_partition(
    source_dir: Path,
    destination_dir: Path,
    name_prefix: str,
) -> int:
    parquet_files = sorted(source_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in expected partition: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    for parquet_file in parquet_files:
        destination_file = destination_dir / f"{name_prefix}-{parquet_file.name}"
        table = pq.read_table(parquet_file)
        language_columns = [column for column in ("language", "lang") if column in table.column_names]

        # Use Hive-style language partition as single source of truth.
        if language_columns:
            table = table.drop_columns(language_columns)
            pq.write_table(table, destination_file)
        else:
            try:
                destination_file.hardlink_to(parquet_file)
            except OSError:
                shutil.copy2(parquet_file, destination_file)

    return len(parquet_files)


for root in waxal_roots:
    if not root.is_dir():
        raise FileNotFoundError(f"Waxal dataset directory does not exist: {root}")

shutil.rmtree(combined_root, ignore_errors=True)
combined_root.mkdir(parents=True)

manifest = {
    "waxal_roots": [str(root) for root in waxal_roots],
    "partitions": [],
}

for source_root in waxal_roots:
    source_name = source_root.name
    language_dirs = sorted(source_root.glob("corpus=*/split=*/language=*"))
    if not language_dirs:
        raise FileNotFoundError(
            "No language partitions found under expected layout in "
            f"{source_root}: corpus=<name>/split=<name>/language=<name>"
        )

    for language_dir in language_dirs:
        source_split = language_dir.parents[0].name.removeprefix("split=")
        corpus = language_dir.parents[1].name.removeprefix("corpus=")
        language = language_dir.name.removeprefix("language=")

        destination_dir = (
            combined_root
            / f"corpus={corpus}"
            / "split=train"
            / f"language={language}"
        )
        name_prefix = f"source-{source_name}-{source_split}"
        num_files = copy_partition(language_dir, destination_dir, name_prefix)

        manifest["partitions"].append(
            {
                "source_dataset": source_name,
                "corpus": corpus,
                "source_split": source_split,
                "destination_split": "train",
                "language": language,
                "parquet_files": num_files,
            }
        )

(combined_root / "_composition.json").write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Combined Waxal dataset ready at: {combined_root}")
for partition in manifest["partitions"]:
    print(
        f"  {partition['source_dataset']}:{partition['source_split']} -> "
        f"train ({partition['language']}, {partition['parquet_files']} parquet file(s))"
    )
