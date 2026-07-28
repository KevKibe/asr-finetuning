#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from infer_waxalnlp import _add_omnilingual_to_path, _decode_audio_to_waveform


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_audio_dir = project_root / "audio"
    default_csv = project_root / "src" / "Test_phase2.csv"

    parser = argparse.ArgumentParser(
        description="Transcribe wav files with Omnilingual and write predictions into Test_phase2.csv."
    )
    parser.add_argument(
        "--model-card",
        required=True,
        help="Model card name registered in Omnilingual cards (e.g. luganda-omni).",
    )
    parser.add_argument(
        "--omni-lang",
        default=None,
        help="Optional Omnilingual language id (e.g. lug_Latn).",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(default_audio_dir),
        help=f"Directory containing wav files (default: {default_audio_dir}).",
    )
    parser.add_argument(
        "--csv-path",
        default=str(default_csv),
        help=f"Input CSV with ID and Target columns (default: {default_csv}).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to overwriting --csv-path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Omnilingual device (default: cuda).",
    )
    return parser.parse_args()


def _build_audio_index(audio_dir: Path) -> dict[str, Path]:
    audio_index: dict[str, Path] = {}
    for wav_path in sorted(audio_dir.glob("*.wav")):
        stem = wav_path.stem
        if stem not in audio_index:
            audio_index[stem] = wav_path
    return audio_index


def _transcribe_batch(
    *,
    pipeline: object,
    audio_inputs: list[object],
    sample_ids: list[str],
    omni_lang: str | None,
    batch_size: int,
) -> dict[str, str]:
    if not audio_inputs:
        return {}

    lang_list = [omni_lang] * len(audio_inputs) if omni_lang else None

    try:
        hyps = pipeline.transcribe(audio_inputs, lang=lang_list, batch_size=batch_size)
        return {sid: str(hyp) for sid, hyp in zip(sample_ids, hyps)}
    except Exception as exc:
        print(f"Batch failed ({len(audio_inputs)} samples), falling back to single-item mode: {exc}")

    out: dict[str, str] = {}
    for sid, audio in zip(sample_ids, audio_inputs):
        try:
            hyp = pipeline.transcribe(
                [audio],
                lang=[omni_lang] if omni_lang else None,
                batch_size=1,
            )[0]
            out[sid] = str(hyp)
        except Exception as single_exc:
            print(f"Failed to transcribe {sid}: {single_exc}")
    return out


def main() -> int:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _add_omnilingual_to_path(project_root)

    audio_dir = Path(args.audio_dir)
    csv_path = Path(args.csv_path)
    output_csv = Path(args.output_csv) if args.output_csv else csv_path

    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"Audio directory does not exist: {audio_dir}", file=sys.stderr)
        return 1
    if not csv_path.exists() or not csv_path.is_file():
        print(f"CSV file does not exist: {csv_path}", file=sys.stderr)
        return 1

    try:
        sf_module = importlib.import_module("soundfile")
    except ImportError:
        print("Missing package: soundfile", file=sys.stderr)
        print("Install: pip install soundfile", file=sys.stderr)
        return 1

    try:
        ASRInferencePipeline = importlib.import_module(
            "omnilingual_asr.models.inference.pipeline"
        ).ASRInferencePipeline
    except ImportError:
        print("Missing package/module: omnilingual_asr", file=sys.stderr)
        print("Ensure omnilingual-asr is cloned and installed", file=sys.stderr)
        return 1

    print(f"Initializing Omnilingual pipeline with model card: {args.model_card}")
    pipeline = ASRInferencePipeline(model_card=args.model_card, device=args.device)

    df = pd.read_csv(csv_path)
    if "ID" not in df.columns:
        print("Input CSV must include an 'ID' column.", file=sys.stderr)
        return 1
    if "Target" not in df.columns:
        df["Target"] = ""
    # Keep Target writable for text outputs even when CSV inference picks float.
    df["Target"] = df["Target"].astype("object").where(df["Target"].notna(), "")

    audio_index = _build_audio_index(audio_dir)
    if not audio_index:
        print(f"No .wav files found in: {audio_dir}", file=sys.stderr)
        return 1

    id_order: list[str] = []
    id_rows: dict[str, list[int]] = {}
    for idx, value in df["ID"].items():
        sample_id = str(value).strip()
        if not sample_id:
            continue
        if sample_id not in id_rows:
            id_rows[sample_id] = []
            id_order.append(sample_id)
        id_rows[sample_id].append(int(idx))

    pred_by_id: dict[str, str] = {}
    missing = 0
    skipped_decode = 0
    processed_unique = 0

    batch_audio: list[object] = []
    batch_ids: list[str] = []

    for sample_id in id_order:
        wav_path = audio_index.get(sample_id)
        if wav_path is None:
            missing += 1
            continue

        decoded = _decode_audio_to_waveform({"path": str(wav_path)}, sf_module)
        if decoded is None:
            skipped_decode += 1
            continue

        batch_audio.append(decoded)
        batch_ids.append(sample_id)

        if len(batch_audio) >= args.batch_size:
            batch_pred = _transcribe_batch(
                pipeline=pipeline,
                audio_inputs=batch_audio,
                sample_ids=batch_ids,
                omni_lang=args.omni_lang,
                batch_size=args.batch_size,
            )
            pred_by_id.update(batch_pred)
            processed_unique += len(batch_ids)
            print(f"Processed {processed_unique}/{len(id_order)} unique IDs")
            batch_audio, batch_ids = [], []

    if batch_audio:
        batch_pred = _transcribe_batch(
            pipeline=pipeline,
            audio_inputs=batch_audio,
            sample_ids=batch_ids,
            omni_lang=args.omni_lang,
            batch_size=args.batch_size,
        )
        pred_by_id.update(batch_pred)
        processed_unique += len(batch_ids)
        print(f"Processed {processed_unique}/{len(id_order)} unique IDs")

    filled_rows = 0
    for sample_id, rows in id_rows.items():
        text = pred_by_id.get(sample_id)
        if text is None:
            continue
        for row_idx in rows:
            df.at[row_idx, "Target"] = text
            filled_rows += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Done. Wrote: {output_csv}")
    print(f"Filled rows: {filled_rows}/{len(df)}")
    if missing:
        print(f"Missing audio for IDs: {missing}")
    if skipped_decode:
        print(f"Skipped due to audio decode issues: {skipped_decode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
