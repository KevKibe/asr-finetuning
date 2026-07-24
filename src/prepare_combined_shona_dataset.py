"""Build a combined FLEURS and Waxal dataset from parquet partitions."""

import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


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


def language_base(language: str) -> str:
    """Return the language portion of a BCP-47-like language identifier."""
    return language.split("_", 1)[0]


def copy_partition(
    source_dir: Path,
    destination_dir: Path,
    name_prefix: str,
    canonical_language: str,
) -> int:
    parquet_files = sorted(source_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in expected partition: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    for parquet_file in parquet_files:
        destination_file = destination_dir / f"{name_prefix}-{parquet_file.name}"
        table = pq.read_table(parquet_file)
        language_columns = [
            column for column in ("language", "lang") if column in table.column_names
        ]

        if not language_columns:
            # Some datasets store language only in their Hive-style directory
            # name. The destination language=<canonical_language> directory is
            # sufficient for fairseq2 to materialize the correct language field.
            try:
                destination_file.hardlink_to(parquet_file)
            except OSError:
                shutil.copy2(parquet_file, destination_file)
            continue

        # Normalize language columns to plain string arrays in every file to
        # prevent mixed string/dictionary schema merges on some environments.
        normalized_language = pa.array([canonical_language] * table.num_rows, type=pa.string())
        for language_column in language_columns:
            language_index = table.schema.get_field_index(language_column)
            table = table.set_column(language_index, language_column, normalized_language)

        pq.write_table(table, destination_file)

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

# Models identify FLEURS languages with script-qualified labels (for example,
# lug_Latn). Map Waxal's short labels (lug) to their matching FLEURS labels.
fleurs_language_labels = {
    language_dir.name.removeprefix("language=")
    for language_dir in fleurs_root.glob("corpus=fleurs/split=*/language=*")
}
fleurs_language_by_base = {
    language_base(language): language for language in fleurs_language_labels
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
        source_language = language_dir.name.removeprefix("language=")
        canonical_language = source_language
        if source_root == waxal_root:
            canonical_language = fleurs_language_by_base.get(
                language_base(source_language), source_language
            )

        destination_dir = (
            combined_root
            / f"corpus={corpus}"
            / f"split={destination_split}"
            / f"language={canonical_language}"
        )
        name_prefix = f"source-{source_split}"
        num_files = copy_partition(
            language_dir, destination_dir, name_prefix, canonical_language
        )
        manifest["partitions"].append(
            {
                "corpus": corpus,
                "source_split": source_split,
                "destination_split": destination_split,
                "source_language": source_language,
                "language": canonical_language,
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
