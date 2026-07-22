import sys
from pathlib import Path
from huggingface_hub import snapshot_download

if len(sys.argv) < 2:
    print("Usage: python dataset_download.py <dataset_repo_id> [local_dir]")
    print("Example: python dataset_download.py KevinKibe/fleurs-shona-omni")
    sys.exit(1)

dataset_repo = sys.argv[1]
local_dir = sys.argv[2] if len(sys.argv) > 2 else Path.cwd() / Path(dataset_repo).name

print(f"Downloading dataset: {dataset_repo}")
print(f"Destination: {local_dir}\n")

dataset_path = snapshot_download(
    repo_id=dataset_repo,
    repo_type="dataset",
    local_dir=local_dir
)

print(f"\nDataset ready at: {dataset_path}")