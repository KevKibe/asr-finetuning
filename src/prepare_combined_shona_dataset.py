"""Build a combined FLEURS and Waxal dataset from parquet partitions."""

import json
import shutil
import sys
from pathlib import Path


if len(sys.argv) != 4:
    print(
        "Usage: python prepare_combined_shona_dataset.py "
        "<fleurs_root> <waxal_root> <combined_root>"
    )
    sys.exit(1)

fleurs_root = Path(sys.argv[1])
waxal_root = Path(sys.argv[2])
combined_root = Path(sys.argv[3])

# Source corpus, source split, destination split.
# FLEURS has no validation role in a combined dataset: all of its examples
# contribute to training. Waxal validation remains isolated for evaluation.
partition_mapping = (
    (fleurs_root, "fleurs", "train", "train"),
    (fleurs_root, "fleurs", "dev", "train"),
    (fleurs_root, "fleurs", "test", "train"),
    (waxal_root, "waxal", "train", "train"),
    (waxal_root, "waxal", "test", "train"),
    (waxal_root, "waxal", "validation", "validation"),
)


def copy_partition(source_dir: Path, destination_dir: Path, name_prefix: str) -> int:
    parquet_files = sorted(source_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in expected partition: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    for parquet_file in parquet_files:
        destination_file = destination_dir / f"{name_prefix}-{parquet_file.name}"
        try:
            destination_file.hardlink_to(parquet_file)
        except OSError:
            shutil.copy2(parquet_file, destination_file)

    return len(parquet_files)


if not fleurs_root.is_dir():
    raise FileNotFoundError(f"FLEURS dataset directory does not exist: {fleurs_root}")
if not waxal_root.is_dir():
    raise FileNotFoundError(f"Waxal dataset directory does not exist: {waxal_root}")

shutil.rmtree(combined_root, ignore_errors=True)
combined_root.mkdir(parents=True)

manifest = {
    "fleurs_root": str(fleurs_root),
    "waxal_root": str(waxal_root),
    "partitions": [],
}

for source_root, corpus, source_split, destination_split in partition_mapping:
    language_dirs = sorted(
        (source_root / f"corpus={corpus}" / f"split={source_split}").glob("language=*")
    )
    if not language_dirs:
        raise FileNotFoundError(
            "No language partitions found in expected source split: "
            f"{source_root}/corpus={corpus}/split={source_split}"
        )

    for language_dir in language_dirs:
        destination_dir = (
            combined_root
            / f"corpus={corpus}"
            / f"split={destination_split}"
            / language_dir.name
        )
        name_prefix = f"source-{source_split}"
        num_files = copy_partition(language_dir, destination_dir, name_prefix)
        manifest["partitions"].append(
            {
                "corpus": corpus,
                "source_split": source_split,
                "destination_split": destination_split,
                "language": language_dir.name.removeprefix("language="),
                "parquet_files": num_files,
            }
        )

# PyArrow recursively scans the dataset root. Files prefixed with an underscore
# are ignored by the dataset scanner, unlike a root-level composition.json.
(combined_root / "_composition.json").write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Combined dataset ready at: {combined_root}")
for partition in manifest["partitions"]:
    print(
        f"  {partition['corpus']}/{partition['source_split']} -> "
        f"{partition['destination_split']} ({partition['language']}, "
        f"{partition['parquet_files']} parquet file(s))"
    )
