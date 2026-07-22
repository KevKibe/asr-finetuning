import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

if len(sys.argv) < 2:
    print("Usage: python upload_model.py <repo_id> [export_dir] [hf_token]")
    print("Example: python upload_model.py KevinKibe/omniASR-fleurs-shona-finetuned")
    print("")
    print("If HF_TOKEN environment variable is not set, pass it as third argument.")
    sys.exit(1)

repo_id = sys.argv[1]
folder_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "hf_export"
HF_TOKEN = sys.argv[3] if len(sys.argv) > 3 else os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN not provided and HF_TOKEN environment variable not set")
    sys.exit(1)


print(f"Uploading to: {repo_id}")
print(f"From directory: {folder_path}\n")

if not folder_path.exists():
    print(f"Error: Export directory does not exist: {folder_path}")
    sys.exit(1)

# Create private repo (safe if it already exists)
print("Creating repository...")
create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=True,
    token=HF_TOKEN,
    exist_ok=True,
)

# Upload folder
print("Uploading files...")
upload_folder(
    folder_path=str(folder_path),
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Upload fine-tuned OmniASR model"
)

print(f"\n✓ Upload complete!")
print(f"Model available at: https://huggingface.co/{repo_id}")