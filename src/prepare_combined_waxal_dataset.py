#!/usr/bin/env python3
"""Build a combined multi-language Waxal dataset.

Train and test are merged into split=train, while validation/dev/valid are
merged into split=validation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


if len(sys.argv) != 5:
    print(
        "Usage: python prepare_combined_waxal_dataset.py "
        "<waxal_sna_root> <waxal_lin_root> <waxal_lug_root> <combined_root>"
    )
    sys.exit(1)

waxal_roots = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])]
combined_root = Path(sys.argv[4])

# Omnilingual language IDs are script-qualified.
LANGUAGE_CANONICAL_MAP = {
    "sna": "sna_Latn",
    "lug": "lug_Latn",
    "lin": "lin_Latn",
}

TEXT_COLUMNS = ("text", "transcript", "sentence", "normalized_text", "target_text")
DURATION_COLUMNS = ("duration", "audio_duration", "duration_s", "duration_ms")
FRAME_COLUMNS = (
    "num_frames",
    "n_frames",
    "input_frames",
    "num_samples",
    "n_samples",
    "audio_num_samples",
)


def canonical_language(language: str) -> str:
    return LANGUAGE_CANONICAL_MAP.get(language, language)


def destination_split(source_split: str) -> str:
    if source_split in {"validation", "valid", "dev"}:
        return "validation"
    return "train"


def _and_mask(current: pa.Array | None, condition: pa.Array | None) -> pa.Array | None:
    if condition is None:
        return current
    if current is None:
        return condition

    return pc.and_(current, condition)


def _non_empty_text_condition(table: pa.Table) -> pa.Array | None:
    for column_name in TEXT_COLUMNS:
        if column_name not in table.column_names:
            continue

        column = table[column_name]
        text = pc.cast(column, pa.string(), safe=False)
        text = pc.fill_null(text, "")
        trimmed = pc.utf8_trim_whitespace(text)

        return pc.greater(pc.utf8_length(trimmed), 0)

    return None


def _positive_numeric_condition(table: pa.Table, candidates: tuple[str, ...]) -> pa.Array | None:
    condition = None

    for column_name in candidates:
        if column_name not in table.column_names:
            continue

        numeric = pc.cast(table[column_name], pa.float64(), safe=False)
        valid = pc.invert(pc.is_null(numeric))
        positive = pc.greater(numeric, 0.0)
        condition = _and_mask(condition, pc.and_(valid, positive))

    return condition


def sanitize_rows(table: pa.Table) -> tuple[pa.Table, int]:
    keep_condition = None
    keep_condition = _and_mask(keep_condition, _non_empty_text_condition(table))
    keep_condition = _and_mask(keep_condition, _positive_numeric_condition(table, DURATION_COLUMNS))
    keep_condition = _and_mask(keep_condition, _positive_numeric_condition(table, FRAME_COLUMNS))

    if keep_condition is None:
        return table, 0

    filtered = table.filter(keep_condition)
    dropped_rows = table.num_rows - filtered.num_rows

    return filtered, dropped_rows


def copy_partition(
    source_dir: Path,
    destination_dir: Path,
    name_prefix: str,
) -> dict[str, int]:
    parquet_files = sorted(source_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in expected partition: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    dropped_files = 0
    rows_before = 0
    rows_after = 0
    dropped_rows = 0

    for parquet_file in parquet_files:
        destination_file = destination_dir / f"{name_prefix}-{parquet_file.name}"
        table = pq.read_table(parquet_file)
        rows_before += table.num_rows

        table, table_dropped_rows = sanitize_rows(table)
        dropped_rows += table_dropped_rows

        if table.num_rows == 0:
            dropped_files += 1
            continue

        rows_after += table.num_rows
        language_columns = [column for column in ("language", "lang") if column in table.column_names]

        # Use Hive-style language partition as single source of truth.
        if language_columns:
            table = table.drop_columns(language_columns)
            pq.write_table(table, destination_file)
        else:
            # If rows were filtered, rewrite the file to persist the filtered table.
            if table_dropped_rows > 0:
                pq.write_table(table, destination_file)
            else:
                try:
                    destination_file.hardlink_to(parquet_file)
                except OSError:
                    shutil.copy2(parquet_file, destination_file)

        copied_files += 1

    if copied_files == 0:
        raise ValueError(
            "All rows were filtered out while sanitizing partition "
            f"{source_dir}. Check source parquet contents for invalid rows."
        )

    return {
        "parquet_files": copied_files,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "dropped_rows": dropped_rows,
        "dropped_files": dropped_files,
    }


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
        target_split = destination_split(source_split)
        corpus = language_dir.parents[1].name.removeprefix("corpus=")
        source_language = language_dir.name.removeprefix("language=")
        language = canonical_language(source_language)

        destination_dir = (
            combined_root
            / f"corpus={corpus}"
            / f"split={target_split}"
            / f"language={language}"
        )
        name_prefix = f"source-{source_name}-{source_split}"
        partition_stats = copy_partition(language_dir, destination_dir, name_prefix)

        manifest["partitions"].append(
            {
                "source_dataset": source_name,
                "corpus": corpus,
                "source_split": source_split,
                "destination_split": target_split,
                "source_language": source_language,
                "language": language,
                "parquet_files": partition_stats["parquet_files"],
                "rows_before": partition_stats["rows_before"],
                "rows_after": partition_stats["rows_after"],
                "dropped_rows": partition_stats["dropped_rows"],
                "dropped_files": partition_stats["dropped_files"],
            }
        )

(combined_root / "_composition.json").write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Combined Waxal dataset ready at: {combined_root}")
for partition in manifest["partitions"]:
    print(
        f"  {partition['source_dataset']}:{partition['source_split']} -> "
        f"{partition['destination_split']} ({partition['language']}, {partition['parquet_files']} parquet file(s), "
        f"dropped_rows={partition['dropped_rows']}, dropped_files={partition['dropped_files']})"
    )
