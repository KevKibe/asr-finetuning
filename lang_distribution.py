import sys
from pathlib import Path
import pyarrow.parquet as pq

if len(sys.argv) < 2:
    print("Usage: python lang_distribution.py <dataset_root>")
    sys.exit(1)

root = Path(sys.argv[1])

# Detect language
lang_dir = next(root.glob("corpus=*/split=*/language=*"))
language = lang_dir.name.replace("language=", "")

lines = ["dataset\tsplit\tlanguage\tcorpus\thours\tpath"]

for split in ["train", "dev", "test"]:
    parquet_file = root / "corpus=fleurs" / f"split={split}" / f"language={language}" / "part-0.parquet"

    if parquet_file.exists():
        table = pq.read_table(parquet_file, columns=["audio_size"])
        hours = table["audio_size"].to_numpy().sum() / 16000 / 3600

        lines.append(
            f"{root.name}\t{split}\t{language}\tfleurs\t{hours:.6f}\t{root}"
        )

output = root / "language_distribution_0.tsv"
output.write_text("\n".join(lines))

print(output.read_text())