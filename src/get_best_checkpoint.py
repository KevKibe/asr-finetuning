import sys
from pathlib import Path
import json
import shutil
from typing import Optional


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


def _parse_step(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _checkpoint_weight_exists(ckpt_dir: Path) -> bool:
    # Keep compatibility with fairseq2 checkpoint layout.
    return (ckpt_dir / "model/pp_00/tp_00/sdp_00.pt").exists()


def _checkpoint_steps(run_dir: Path) -> list[int]:
    checkpoints_root = run_dir / "checkpoints"
    steps = []
    for ckpt_dir in checkpoints_root.glob("step_*"):
        if not ckpt_dir.is_dir():
            continue
        step = _parse_step(ckpt_dir.name.removeprefix("step_"))
        if step is None:
            continue
        if _checkpoint_weight_exists(ckpt_dir):
            steps.append(step)
    return sorted(set(steps))


def _train_loss_by_step(run_dir: Path) -> dict[int, float]:
    by_step: dict[int, float] = {}
    for m in load_jsonl(run_dir / "metrics" / "train.jsonl"):
        step = _parse_step(m.get("Step"))
        if step is None:
            continue
        try:
            by_step[step] = float(m["CTC Loss"])
        except Exception:
            continue
    return by_step


def find_run(root_dir: Path) -> Path:
    runs = [p for p in root_dir.glob("ws_*") if p.is_dir()]
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly one run, found: {runs}")
    run_dir = runs[0]
    print(f"Using run: {run_dir}")
    return run_dir


def find_best_checkpoint(run_dir: Path):
    valid_file = run_dir / "metrics" / "valid.jsonl"
    candidates = []
    selection_source = "validation_ctc_loss"

    if valid_file.exists():
        for m in load_jsonl(valid_file):
            step = _parse_step(m.get("Step"))
            if step is None:
                continue
            try:
                eval_loss = float(m["CTC Loss"])
            except Exception:
                continue

            ckpt = run_dir / "checkpoints" / f"step_{step}"
            if not _checkpoint_weight_exists(ckpt):
                continue

            candidates.append(
                {
                    "step": step,
                    "eval_loss": eval_loss,
                    "train_loss": None,
                    "wer": m.get("Word Error Rate (WER)"),
                    "uer": m.get("Unit Error Rate (UER)"),
                    "path": ckpt,
                }
            )

    if candidates:
        best = min(candidates, key=lambda x: x["eval_loss"])
    else:
        checkpoint_steps = _checkpoint_steps(run_dir)
        if not checkpoint_steps:
            raise RuntimeError("No checkpoints with model weights were found under run/checkpoints")

        train_loss_map = _train_loss_by_step(run_dir)
        candidates = [
            {
                "step": step,
                "eval_loss": None,
                "train_loss": train_loss_map.get(step),
                "wer": None,
                "uer": None,
                "path": run_dir / "checkpoints" / f"step_{step}",
            }
            for step in checkpoint_steps
        ]

        train_candidates = [c for c in candidates if c["train_loss"] is not None]
        if train_candidates:
            selection_source = "train_ctc_loss"
            best = min(train_candidates, key=lambda x: x["train_loss"])
        else:
            selection_source = "latest_step"
            best = max(candidates, key=lambda x: x["step"])

    best["selection_source"] = selection_source

    print("\nBest checkpoint:")
    print(f"  Step: {best['step']}")
    if best["eval_loss"] is not None:
        print(f"  Validation CTC Loss: {best['eval_loss']}")
    elif best["train_loss"] is not None:
        print(f"  Train CTC Loss: {best['train_loss']}")
    else:
        print("  CTC Loss: n/a")
    print(f"  WER: {best['wer']}")
    print(f"  UER: {best['uer']}")
    print(f"  Selection source: {best['selection_source']}")
    print(f"  Path: {best['path']}\n")

    return best, candidates


def generate_readme(run_dir: Path, best: dict, export_dir: Path):
    valid_metrics = load_jsonl(run_dir / "metrics" / "valid.jsonl")
    train_metrics = load_jsonl(run_dir / "metrics" / "train.jsonl")

    lines = []
    lines.append("# OmniASR Fine-tuned Model\n\n")
    lines.append("## Training Summary\n\n")
    lines.append(f"Best checkpoint: step_{best['step']}\n\n")
    lines.append(f"Checkpoint selection source: {best.get('selection_source', 'unknown')}\n\n")
    if best.get("eval_loss") is not None:
        lines.append(f"Best validation CTC Loss: {fmt(best['eval_loss'], 4)}\n\n")
        lines.append(f"Best validation WER: {fmt(best['wer'])}\n\n")
        lines.append(f"Best validation UER: {fmt(best['uer'])}\n\n")
    elif best.get("train_loss") is not None:
        lines.append(f"Best train CTC Loss: {fmt(best['train_loss'], 4)}\n\n")
        lines.append("Validation metrics: n/a (validation disabled)\n\n")
    else:
        lines.append("CTC Loss: n/a\n\n")
        lines.append("Validation metrics: n/a (validation disabled)\n\n")

    lines.append("## Validation Metrics\n\n")
    lines.append("| Step | CTC Loss | WER | UER |\n")
    lines.append("|---|---|---|---|\n")
    if valid_metrics:
        for m in valid_metrics:
            lines.append(
                f"| {m.get('Step', '')} "
                f"| {fmt(m.get('CTC Loss'), 4)} "
                f"| {fmt(m.get('Word Error Rate (WER)'))} "
                f"| {fmt(m.get('Unit Error Rate (UER)'))} |\n"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a |\n")

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
    best, candidates = find_best_checkpoint(run_dir)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all checkpoints
    checkpoints_dir = EXPORT_DIR / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying {len(candidates)} checkpoints...")
    for candidate in candidates:
        step = candidate["step"]
        src_ckpt = candidate["path"]
        dst_ckpt = checkpoints_dir / f"step_{step}"
        
        if not dst_ckpt.exists():
            print(f"  Copying step_{step}...")
            copytree_if_exists(src_ckpt / "model", dst_ckpt / "model")
    
    print(f"✓ All {len(candidates)} checkpoints saved")

    # Copy config files once (shared across all checkpoints)
    print("\nCopying config files...")
    copy_if_exists(run_dir / "config.yaml", EXPORT_DIR / "config.yaml")
    copy_if_exists(run_dir / "checkpoints/model.yaml", EXPORT_DIR / "model.yaml")

    # Copy useful artifacts
    copytree_if_exists(run_dir / "metrics", EXPORT_DIR / "metrics")
    copytree_if_exists(run_dir / "transcriptions", EXPORT_DIR / "transcriptions")

    # Remove unwanted artifacts
    for name in ("tb", "tensorboard"):
        shutil.rmtree(EXPORT_DIR / name, ignore_errors=True)

    # Generate model card README with best checkpoint info
    generate_readme(run_dir, best, EXPORT_DIR)

    print(f"\nExport complete: {EXPORT_DIR}")
    print(f"Best checkpoint: step_{best['step']}")
    print(f"All checkpoints location: {checkpoints_dir}")


if __name__ == "__main__":
    main()