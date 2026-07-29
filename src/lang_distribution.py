import sys
import re
from pathlib import Path

import pyarrow.parquet as pq

if len(sys.argv) < 2:
    print("Usage: python lang_distribution.py <dataset_root>")
    sys.exit(1)

root = Path(sys.argv[1])

lines = ["dataset\tsplit\tlanguage\tcorpus\thours\tpath"]
num_rows = 0


WAXAL_FILENAME_RE = re.compile(
    r"^(?P<language>[A-Za-z0-9_\-]+)-(?P<split>train|test|validation)-(?P<shard>\d+)\.parquet$"
)


def _iter_legacy_partitions(dataset_root: Path):
    for lang_dir in sorted(dataset_root.glob("corpus=*/split=*/language=*")):
        corpus = lang_dir.parents[1].name.replace("corpus=", "", 1)
        split = lang_dir.parents[0].name.replace("split=", "", 1)
        language = lang_dir.name.replace("language=", "", 1)
        parquet_files = sorted(p for p in lang_dir.rglob("*.parquet") if p.is_file())
        if not parquet_files:
            continue
        yield corpus, split, language, parquet_files


def _infer_waxal_partition(dataset_root: Path, parquet_file: Path):
    rel_parts = parquet_file.relative_to(dataset_root).parts

    corpus = None
    language = None
    split = None

    for idx, part in enumerate(rel_parts[:-1]):
        if part.lower() == "asr" and idx + 1 < len(rel_parts) - 1:
            corpus = part
            language = rel_parts[idx + 1]
            break

    filename = parquet_file.name
    match = WAXAL_FILENAME_RE.match(filename)
    if match:
        split = match.group("split")
        language_from_name = match.group("language")
        if language is None:
            language = language_from_name
    else:
        # Fall back to a parent directory language name plus filename suffix.
        if language is None and len(rel_parts) >= 2:
            language = rel_parts[-2]
        filename_stem = Path(filename).stem
        if "-" in filename_stem:
            split = filename_stem.split("-")[-1]

    if corpus is None or language is None or split not in {"train", "test", "validation"}:
        return None

    return corpus, split, language


def _iter_waxal_partitions(dataset_root: Path):
    grouped: dict[tuple[str, str, str], list[Path]] = {}

    for parquet_file in sorted(dataset_root.rglob("*.parquet")):
        if not parquet_file.is_file():
            continue

        inferred = _infer_waxal_partition(dataset_root, parquet_file)
        if inferred is None:
            continue

        corpus, split, language = inferred
        grouped.setdefault((corpus, split, language), []).append(parquet_file)

    for (corpus, split, language), parquet_files in sorted(grouped.items()):
        yield corpus, split, language, parquet_files

legacy_partitions = list(_iter_legacy_partitions(root))
if legacy_partitions:
    partition_iter = legacy_partitions
else:
    partition_iter = list(_iter_waxal_partitions(root))

# Discover all partitions either in Omnilingual-ASR layout or WaxalNLP layout.
for corpus, split, language, parquet_files in partition_iter:

    total_audio_size = 0
    for parquet_file in parquet_files:
        parquet_reader = pq.ParquetFile(parquet_file)
        for batch in parquet_reader.iter_batches(columns=["audio_size"]):
            total_audio_size += batch.column(0).to_numpy(zero_copy_only=False).sum()

    hours = total_audio_size / 16000 / 3600
    lines.append(f"{root.name}\t{split}\t{language}\t{corpus}\t{hours:.6f}\t{root}")
    num_rows += 1

if num_rows == 0:
    print(f"No parquet partitions found under: {root}")
    print(
        "Expected paths like corpus=<name>/split=<name>/language=<name>/*.parquet "
        "or data/ASR/<language>/<language>-<split>-*.parquet"
    )
    sys.exit(2)

output = root / "language_distribution_0.tsv"
output.write_text("\n".join(lines))

print(output.read_text())