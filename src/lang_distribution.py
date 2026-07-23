import sys
from pathlib import Path
import pyarrow.parquet as pq

if len(sys.argv) < 2:
    print("Usage: python lang_distribution.py <dataset_root>")
    sys.exit(1)

root = Path(sys.argv[1])

lines = ["dataset\tsplit\tlanguage\tcorpus\thours\tpath"]
num_rows = 0

# Discover all Omnilingual-ASR partition directories:
# corpus=<name>/split=<name>/language=<name>
for lang_dir in sorted(root.glob("corpus=*/split=*/language=*")):
    corpus = lang_dir.parents[1].name.replace("corpus=", "", 1)
    split = lang_dir.parents[0].name.replace("split=", "", 1)
    language = lang_dir.name.replace("language=", "", 1)

    parquet_files = sorted(lang_dir.glob("*.parquet"))
    if not parquet_files:
        continue

    total_audio_size = 0
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["audio_size"])
        total_audio_size += table["audio_size"].to_numpy().sum()

    hours = total_audio_size / 16000 / 3600
    lines.append(
        f"{root.name}\t{split}\t{language}\t{corpus}\t{hours:.6f}\t{root}"
    )
    num_rows += 1

if num_rows == 0:
    print(f"No parquet partitions found under: {root}")
    print("Expected paths like corpus=<name>/split=<name>/language=<name>/*.parquet")
    sys.exit(2)

output = root / "language_distribution_0.tsv"
output.write_text("\n".join(lines))

print(output.read_text())