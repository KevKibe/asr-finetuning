"""Build a combined FLEURS and Waxal dataset from parquet partitions."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
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

# Extremely short clips can survive >0 checks but still collapse to zero
# time steps after feature extraction/subsampling in validation.
NUMERIC_MINIMUMS = {
    "duration": 0.03,
    "audio_duration": 0.03,
    "duration_s": 0.03,
    "duration_ms": 30.0,
    "num_frames": 1.0,
    "n_frames": 1.0,
    "input_frames": 1.0,
    "num_samples": 400.0,
    "n_samples": 400.0,
    "audio_num_samples": 400.0,
}


def language_base(language: str) -> str:
    """Return the language portion of a BCP-47-like language identifier."""
    return language.split("_", 1)[0]


def _and_mask(current: pa.Array | None, condition: pa.Array | None) -> pa.Array | None:
    if condition is None:
        return current
    if current is None:
        return condition

    return pc.and_(current, condition)


def _non_empty_text_condition(table: pa.Table) -> pa.Array | None:
    condition = None

    for column_name in TEXT_COLUMNS:
        if column_name not in table.column_names:
            continue

        text = pc.cast(table[column_name], pa.string(), safe=False)
        text = pc.fill_null(text, "")
        trimmed = pc.utf8_trim_whitespace(text)
        non_empty = pc.greater(pc.utf8_length(trimmed), 0)
        condition = _and_mask(condition, non_empty)

    return condition


def _minimum_numeric_condition(table: pa.Table, candidates: tuple[str, ...]) -> pa.Array | None:
    condition = None

    for column_name in candidates:
        if column_name not in table.column_names:
            continue

        numeric = pc.cast(table[column_name], pa.float64(), safe=False)
        valid = pc.invert(pc.is_null(numeric))
        minimum = NUMERIC_MINIMUMS.get(column_name, 1.0)
        meets_minimum = pc.greater_equal(numeric, minimum)
        condition = _and_mask(condition, pc.and_(valid, meets_minimum))

    return condition


def sanitize_rows(table: pa.Table) -> tuple[pa.Table, int]:
    keep_condition = None
    keep_condition = _and_mask(keep_condition, _non_empty_text_condition(table))
    keep_condition = _and_mask(keep_condition, _minimum_numeric_condition(table, DURATION_COLUMNS))
    keep_condition = _and_mask(keep_condition, _minimum_numeric_condition(table, FRAME_COLUMNS))

    if keep_condition is None:
        return table, 0

    filtered = table.filter(keep_condition)
    dropped_rows = table.num_rows - filtered.num_rows

    return filtered, dropped_rows


def _candidate_path_bases(source_dir: Path, source_dataset_root: Path) -> list[Path]:
    bases = [source_dir]

    for i in range(1, 4):
        if len(source_dir.parents) >= i:
            bases.append(source_dir.parents[i - 1])

    bases.append(source_dataset_root)

    deduped: list[Path] = []
    seen = set()
    for base in bases:
        resolved = base.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)

    return deduped


def _normalize_audio_path(value: Any, path_bases: list[Path]) -> Any:
    if value is None:
        return value

    if not isinstance(value, dict):
        return value

    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        return value

    path_obj = Path(path)
    if path_obj.is_absolute():
        return value

    for base in path_bases:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            updated = dict(value)
            updated["path"] = str(candidate)
            return updated

    return value


def normalize_audio_paths(table: pa.Table, source_dir: Path, source_dataset_root: Path) -> tuple[pa.Table, int]:
    path_bases = _candidate_path_bases(source_dir, source_dataset_root)
    normalized_rows = 0

    for column_name in ("audio", "speech"):
        if column_name not in table.column_names:
            continue

        column_idx = table.column_names.index(column_name)
        field_type = table.schema.field(column_name).type
        rows = table[column_name].to_pylist()
        updated_rows = []

        for row in rows:
            normalized = _normalize_audio_path(row, path_bases)
            if normalized is not row:
                normalized_rows += 1
            updated_rows.append(normalized)

        table = table.set_column(column_idx, column_name, pa.array(updated_rows, type=field_type))

    return table, normalized_rows


def copy_partition(
    source_dir: Path,
    destination_dir: Path,
    name_prefix: str,
    canonical_language: str,
    source_dataset_root: Path,
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
    normalized_audio_paths = 0

    for parquet_file in parquet_files:
        destination_file = destination_dir / f"{name_prefix}-{parquet_file.name}"
        table = pq.read_table(parquet_file)
        rows_before += table.num_rows

        table, path_updates = normalize_audio_paths(table, source_dir, source_dataset_root)
        normalized_audio_paths += path_updates

        table, table_dropped_rows = sanitize_rows(table)
        dropped_rows += table_dropped_rows

        if table.num_rows == 0:
            dropped_files += 1
            continue

        rows_after += table.num_rows
        language_columns = [column for column in ("language", "lang") if column in table.column_names]

        if not language_columns:
            # Some datasets store language only in their Hive-style directory
            # name. The destination language=<canonical_language> directory is
            # sufficient for fairseq2 to materialize the correct language field.
            if table_dropped_rows > 0:
                pq.write_table(table, destination_file)
            else:
                try:
                    destination_file.hardlink_to(parquet_file)
                except OSError:
                    shutil.copy2(parquet_file, destination_file)
            copied_files += 1
            continue

        # Rely on hive partition paths (language=<canonical_language>) as the
        # single source of truth, and drop in-file language columns to prevent
        # schema merge conflicts (string vs dictionary) across environments.
        table = table.drop_columns(language_columns)

        pq.write_table(table, destination_file)

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
        "normalized_audio_paths": normalized_audio_paths,
    }


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
        partition_stats = copy_partition(
            language_dir,
            destination_dir,
            name_prefix,
            canonical_language,
            source_root,
        )
        manifest["partitions"].append(
            {
                "corpus": corpus,
                "source_split": source_split,
                "destination_split": destination_split,
                "source_language": source_language,
                "language": canonical_language,
                "parquet_files": partition_stats["parquet_files"],
                "rows_before": partition_stats["rows_before"],
                "rows_after": partition_stats["rows_after"],
                "dropped_rows": partition_stats["dropped_rows"],
                "dropped_files": partition_stats["dropped_files"],
                "normalized_audio_paths": partition_stats["normalized_audio_paths"],
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
        f"{partition['parquet_files']} parquet file(s), dropped_rows={partition['dropped_rows']}, "
        f"dropped_files={partition['dropped_files']}, normalized_audio_paths={partition['normalized_audio_paths']})"
    )
