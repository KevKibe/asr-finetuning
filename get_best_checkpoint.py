import sys
from pathlib import Path
import json
import shutil

import torch
from safetensors.torch import save_file


# Get paths from command-line arguments or use defaults
ROOT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "finetuning_output"
EXPORT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "hf_export"


def fmt(value, decimals=2):
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def load_jsonl(path: Path):
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def find_run(root_dir: Path) -> Path:
    runs = [p for p in root_dir.glob("ws_*") if p.is_dir()]
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly one run, found: {runs}")
    run_dir = runs[0]
    print(f"Using run: {run_dir}")
    return run_dir


def find_best_checkpoint(run_dir: Path):
    valid_file = run_dir / "metrics" / "valid.jsonl"
    if not valid_file.exists():
        raise RuntimeError(f"Missing validation metrics: {valid_file}")

    candidates = []
    for m in load_jsonl(valid_file):
        try:
            step = int(m["Step"])
            eval_loss = float(m["CTC Loss"])
        except Exception:
            continue

        ckpt = run_dir / "checkpoints" / f"step_{step}"
        weight_file = ckpt / "model/pp_00/tp_00/sdp_00.pt"
        if not weight_file.exists():
            continue

        candidates.append(
            {
                "step": step,
                "eval_loss": eval_loss,
                "wer": m.get("Word Error Rate (WER)"),
                "uer": m.get("Unit Error Rate (UER)"),
                "path": ckpt,
            }
        )

    if not candidates:
        raise RuntimeError("No checkpoints found matching validation metrics")

    best = min(candidates, key=lambda x: x["eval_loss"])

    print("\nBest checkpoint:")
    print(f"  Step: {best['step']}")
    print(f"  CTC Loss: {best['eval_loss']}")
    print(f"  WER: {best['wer']}")
    print(f"  UER: {best['uer']}")
    print(f"  Path: {best['path']}\n")

    return best, candidates


def generate_readme(run_dir: Path, best: dict, export_dir: Path):
    valid_metrics = load_jsonl(run_dir / "metrics" / "valid.jsonl")
    train_metrics = load_jsonl(run_dir / "metrics" / "train.jsonl")

    lines = []
    lines.append("# OmniASR Fine-tuned Model\n\n")
    lines.append("## Training Summary\n\n")
    lines.append(f"Best checkpoint: step_{best['step']}\n\n")
    lines.append(f"Best validation CTC Loss: {fmt(best['eval_loss'], 4)}\n\n")
    lines.append(f"Best validation WER: {fmt(best['wer'])}\n\n")
    lines.append(f"Best validation UER: {fmt(best['uer'])}\n\n")

    lines.append("## Validation Metrics\n\n")
    lines.append("| Step | CTC Loss | WER | UER |\n")
    lines.append("|---|---|---|---|\n")
    for m in valid_metrics:
        lines.append(
            f"| {m.get('Step', '')} "
            f"| {fmt(m.get('CTC Loss'), 4)} "
            f"| {fmt(m.get('Word Error Rate (WER)'))} "
            f"| {fmt(m.get('Unit Error Rate (UER)'))} |\n"
        )

    lines.append("\n## Training Metrics\n\n")
    lines.append("| Step | CTC Loss | UER | WER | Learning Rate |\n")
    lines.append("|---|---|---|---|---|\n")
    for m in train_metrics:
        lines.append(
            f"| {m.get('Step', '')} "
            f"| {fmt(m.get('CTC Loss'), 4)} "
            f"| {fmt(m.get('Unit Error Rate (UER)'))} "
            f"| {fmt(m.get('Word Error Rate (WER)'))} "
            f"| {fmt(m.get('Learning Rate'), 8)} |\n"
        )

    (export_dir / "README.md").write_text("".join(lines), encoding="utf-8")
    print("Generated README.md")


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy(src, dst)


def copytree_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    print(f"Root dir: {ROOT_DIR}")
    print(f"Export dir: {EXPORT_DIR}\n")

    run_dir = find_run(ROOT_DIR)
    best, _ = find_best_checkpoint(run_dir)
    best_ckpt = best["path"]

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert weights to safetensors
    weight_path = best_ckpt / "model/pp_00/tp_00/sdp_00.pt"
    print(f"Loading: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu")
    save_file(state_dict, EXPORT_DIR / "model.safetensors")
    print("Saved model.safetensors")

    # Copy config files
    copy_if_exists(run_dir / "config.yaml", EXPORT_DIR / "config.yaml")
    copy_if_exists(run_dir / "checkpoints/model.yaml", EXPORT_DIR / "model.yaml")

    # Copy useful artifacts
    copytree_if_exists(run_dir / "metrics", EXPORT_DIR / "metrics")
    copytree_if_exists(run_dir / "transcriptions", EXPORT_DIR / "transcriptions")

    # Remove unwanted artifacts
    for name in ("tb", "tensorboard"):
        shutil.rmtree(EXPORT_DIR / name, ignore_errors=True)

    # Generate model card README
    generate_readme(run_dir, best, EXPORT_DIR)

    print(f"Export complete: {EXPORT_DIR}")


if __name__ == "__main__":
    main()