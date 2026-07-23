#!/usr/bin/env python3
"""Run Omnilingual-ASR inference on WaxalNLP ASR splits for a selected language.

This script loads a WaxalNLP ASR subset (e.g. lug_asr), runs inference with an
Omnilingual model card, and optionally computes WER/CER against the dataset's
`transcription` field.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable


def _add_omnilingual_to_path(project_root: Path) -> None:
    """Allow importing local omnilingual-asr checkout without manual PYTHONPATH."""
    omni_src = project_root / "omnilingual-asr" / "src"
    if omni_src.exists():
        sys.path.insert(0, str(omni_src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Omnilingual-ASR inference on google/WaxalNLP ASR data."
    )
    parser.add_argument("--model-card", required=True, help="Model card name registered in Omnilingual cards.")
    parser.add_argument(
        "--language",
        required=True,
        help="Waxal language code (e.g. lug, lin, sna).",
    )
    parser.add_argument(
        "--omni-lang",
        default=None,
        help="Optional Omnilingual language id (e.g. lug_Latn). Recommended for LLM models.",
    )
    parser.add_argument("--dataset", default="google/WaxalNLP", help="HF dataset repo.")
    parser.add_argument(
        "--config",
        default=None,
        help="HF config name. Defaults to '<language>_asr'.",
    )
    parser.add_argument("--split", default="test", help="Dataset split: test, validation, train.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for incremental transcription.",
    )
    parser.add_argument("--device", default=None, help="Force device, e.g. cuda or cpu.")
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable WER/CER computation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: outputs/inference/<model>_<config>_<split>.jsonl",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset mode to avoid downloading full split files up front.",
    )
    return parser.parse_args()


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i]
        for j, y in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if x == y else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _compute_wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    total_words = 0
    total_errs = 0

    for ref, hyp in zip(refs, hyps):
        ref_words = _normalize_text(ref).split()
        hyp_words = _normalize_text(hyp).split()
        total_words += len(ref_words)
        total_errs += _levenshtein(ref_words, hyp_words)

    if total_words == 0:
        return math.nan
    return total_errs / total_words


def _compute_cer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    total_chars = 0
    total_errs = 0

    for ref, hyp in zip(refs, hyps):
        ref_chars = list(_normalize_text(ref).replace(" ", ""))
        hyp_chars = list(_normalize_text(hyp).replace(" ", ""))
        total_chars += len(ref_chars)
        total_errs += _levenshtein(ref_chars, hyp_chars)

    if total_chars == 0:
        return math.nan
    return total_errs / total_chars


def _print_env_hints() -> None:
    print("If imports fail, ensure dependencies are installed in your active env:")
    print("  pip install 'datasets[audio]' omnilingual-asr")


def _run_inference_batch(
    *,
    pipeline: object,
    args: argparse.Namespace,
    audio_inputs: list[object],
    batch_refs: list[str],
    batch_ids: list[str],
    batch_langs: list[str],
    refs: list[str],
    hyps: list[str],
    out_f: object,
) -> int:
    if not audio_inputs:
        return 0

    lang_list = [args.omni_lang] * len(audio_inputs) if args.omni_lang else None
    batch_hyps = pipeline.transcribe(audio_inputs, lang=lang_list, batch_size=args.batch_size)

    for sid, sl, ref, hyp in zip(batch_ids, batch_langs, batch_refs, batch_hyps):
        row = {
            "id": sid,
            "language": sl,
            "config": args.config_name,
            "split": args.split,
            "reference": ref,
            "prediction": hyp,
            "model_card": args.model_card,
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    refs.extend(batch_refs)
    hyps.extend(batch_hyps)
    return len(batch_refs)


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _add_omnilingual_to_path(project_root)

    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = datasets_module.load_dataset
        Audio = datasets_module.Audio
    except ImportError:
        _print_env_hints()
        print("Missing package: datasets", file=sys.stderr)
        return 1

    try:
        ASRInferencePipeline = importlib.import_module(
            "omnilingual_asr.models.inference.pipeline"
        ).ASRInferencePipeline
    except ImportError:
        _print_env_hints()
        print("Missing package/module: omnilingual_asr", file=sys.stderr)
        return 1

    lang = args.language.lower().strip()
    config_name = args.config or f"{lang}_asr"
    args.config_name = config_name

    if args.output:
        output_path = Path(args.output)
    else:
        output_name = f"{_safe_slug(args.model_card)}_{_safe_slug(config_name)}_{_safe_slug(args.split)}.jsonl"
        output_path = project_root / "outputs" / "inference" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} / {config_name} / split={args.split}")
    try:
        dataset = load_dataset(
            args.dataset,
            config_name,
            split=args.split,
            streaming=args.streaming,
        )
        # Bypass datasets' torchcodec decode path and hand raw audio bytes/path
        # to Omnilingual, which performs its own decoding internally.
        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    n_samples = None if args.streaming else len(dataset)
    if n_samples is not None and args.max_samples is not None:
        n_samples = min(n_samples, args.max_samples)

    if n_samples == 0:
        print("No samples found for the selected split/config.", file=sys.stderr)
        return 1

    print(f"Initializing pipeline with model card: {args.model_card}")
    pipeline = ASRInferencePipeline(model_card=args.model_card, device=args.device)

    has_id = "id" in dataset.column_names
    has_language = "language" in dataset.column_names
    has_transcription = "transcription" in dataset.column_names

    if not has_transcription:
        print("Dataset split has no 'transcription' column. Cannot run ASR eval.", file=sys.stderr)
        return 1

    refs: list[str] = []
    hyps: list[str] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        if args.streaming:
            stream = dataset
            if args.max_samples is not None:
                stream = itertools.islice(stream, args.max_samples)

            audio_inputs: list[object] = []
            batch_refs: list[str] = []
            batch_ids: list[str] = []
            batch_langs: list[str] = []

            for idx, sample in enumerate(stream):
                audio = sample.get("audio")
                if not isinstance(audio, dict):
                    continue

                audio_bytes = audio.get("bytes")
                audio_path = audio.get("path")

                if audio_bytes is not None:
                    audio_inputs.append(audio_bytes)
                elif audio_path:
                    audio_inputs.append(audio_path)
                else:
                    # Fallback if decode=True slips through in a future datasets change.
                    waveform = audio.get("array")
                    sample_rate = audio.get("sampling_rate", audio.get("sample_rate"))
                    if waveform is None or sample_rate is None:
                        continue
                    audio_inputs.append({"waveform": waveform, "sample_rate": int(sample_rate)})

                batch_refs.append(str(sample.get("transcription", "")))
                batch_ids.append(str(sample.get("id", idx)) if has_id else str(idx))
                batch_langs.append(str(sample.get("language", lang)) if has_language else lang)

                if len(audio_inputs) >= args.chunk_size:
                    processed = _run_inference_batch(
                        pipeline=pipeline,
                        args=args,
                        audio_inputs=audio_inputs,
                        batch_refs=batch_refs,
                        batch_ids=batch_ids,
                        batch_langs=batch_langs,
                        refs=refs,
                        hyps=hyps,
                        out_f=out_f,
                    )
                    print(f"Processed {len(refs)} samples (+{processed})")
                    audio_inputs, batch_refs, batch_ids, batch_langs = [], [], [], []

            processed = _run_inference_batch(
                pipeline=pipeline,
                args=args,
                audio_inputs=audio_inputs,
                batch_refs=batch_refs,
                batch_ids=batch_ids,
                batch_langs=batch_langs,
                refs=refs,
                hyps=hyps,
                out_f=out_f,
            )
            if processed:
                print(f"Processed {len(refs)} samples (+{processed})")
        else:
            assert n_samples is not None
            for start in range(0, n_samples, args.chunk_size):
                end = min(start + args.chunk_size, n_samples)
                batch = dataset.select(range(start, end))

                audio_inputs = []
                batch_refs = []
                batch_ids = []
                batch_langs = []

                for idx, sample in enumerate(batch):
                    audio = sample.get("audio")
                    if not isinstance(audio, dict):
                        continue

                    audio_bytes = audio.get("bytes")
                    audio_path = audio.get("path")

                    if audio_bytes is not None:
                        audio_inputs.append(audio_bytes)
                    elif audio_path:
                        audio_inputs.append(audio_path)
                    else:
                        # Fallback if decode=True slips through in a future datasets change.
                        waveform = audio.get("array")
                        sample_rate = audio.get("sampling_rate", audio.get("sample_rate"))
                        if waveform is None or sample_rate is None:
                            continue
                        audio_inputs.append({"waveform": waveform, "sample_rate": int(sample_rate)})

                    batch_refs.append(str(sample.get("transcription", "")))
                    batch_ids.append(str(sample.get("id", start + idx)) if has_id else str(start + idx))
                    batch_langs.append(str(sample.get("language", lang)) if has_language else lang)

                processed = _run_inference_batch(
                    pipeline=pipeline,
                    args=args,
                    audio_inputs=audio_inputs,
                    batch_refs=batch_refs,
                    batch_ids=batch_ids,
                    batch_langs=batch_langs,
                    refs=refs,
                    hyps=hyps,
                    out_f=out_f,
                )
                print(f"Processed {len(refs)}/{n_samples} samples (+{processed})")

    print(f"Saved predictions to: {output_path}")

    if not args.no_metrics:
        wer = _compute_wer(refs, hyps)
        cer = _compute_cer(refs, hyps)

        metrics = {
            "dataset": args.dataset,
            "config": config_name,
            "split": args.split,
            "num_samples": len(refs),
            "model_card": args.model_card,
            "omni_lang": args.omni_lang,
            "wer": wer,
            "cer": cer,
        }

        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print("Metrics:")
        print(f"  WER: {wer:.4f}" if not math.isnan(wer) else "  WER: nan")
        print(f"  CER: {cer:.4f}" if not math.isnan(cer) else "  CER: nan")
        print(f"Saved metrics to: {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
