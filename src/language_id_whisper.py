#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper language ID on wav files in a directory.")
    parser.add_argument(
        "--audio_dir",
        default="audio",
        help="Directory containing .wav files (default: audio).",
    )
    parser.add_argument(
        "--output",
        default="language_predictions_whisper.csv",
        help="Output CSV path (default: language_predictions_whisper.csv).",
    )
    return parser.parse_args()


def _normalize_lang_map(raw_map: dict[str, int]) -> tuple[dict[str, int], dict[int, str]]:
    code_to_id: dict[str, int] = {}
    for key, token_id in raw_map.items():
        code = key
        if code.startswith("<|") and code.endswith("|>"):
            code = code[2:-2]
        code_to_id[code] = int(token_id)

    id_to_code = {token_id: code for code, token_id in code_to_id.items()}
    return code_to_id, id_to_code


def _build_lang_maps(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
) -> tuple[dict[str, int], dict[int, str]]:
    tokenizer = processor.tokenizer

    # Newer HF versions expose this directly on tokenizer.
    raw_map = getattr(tokenizer, "lang_to_id", None)
    if isinstance(raw_map, dict) and raw_map:
        return _normalize_lang_map(raw_map)

    # Some versions expose language ids via generation config.
    raw_map = getattr(model.generation_config, "lang_to_id", None)
    if isinstance(raw_map, dict) and raw_map:
        return _normalize_lang_map(raw_map)

    # Fallback: infer language tokens from additional special tokens.
    code_to_id: dict[str, int] = {}
    for token in getattr(tokenizer, "additional_special_tokens", []) or []:
        if not (token.startswith("<|") and token.endswith("|>")):
            continue
        code = token[2:-2]
        if not re.fullmatch(r"[a-z]{2,5}", code):
            continue

        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            continue
        code_to_id[code] = int(token_id)

    if not code_to_id:
        raise RuntimeError(
            "Could not discover Whisper language tokens for this Transformers version. "
            "Upgrade transformers or switch to a multilingual Whisper checkpoint."
        )

    id_to_code = {token_id: code for code, token_id in code_to_id.items()}
    return code_to_id, id_to_code


def _detect_language(
    waveform: torch.Tensor,
    sample_rate: int,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    lang_token_ids: list[int],
    id_to_code: dict[int, str],
) -> tuple[str, float]:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    audio = waveform.squeeze().numpy()
    inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    start_id = model.generation_config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start_id]], device=device)

    with torch.no_grad():
        outputs = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        first_step_logits = outputs.logits[0, 0]

    language_logits = first_step_logits[lang_token_ids]
    language_probs = torch.softmax(language_logits, dim=-1)

    best_idx = int(torch.argmax(language_probs).item())
    best_token_id = lang_token_ids[best_idx]
    best_code = id_to_code[best_token_id]
    confidence = float(language_probs[best_idx].item())

    return best_code, confidence


def main() -> int:
    args = _parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"Audio directory does not exist: {audio_dir}", file=sys.stderr)
        return 1

    wav_files = sorted(glob.glob(os.path.join(str(audio_dir), "*.wav")))
    if not wav_files:
        print(f"No .wav files found in: {audio_dir}", file=sys.stderr)
        return 1

    processor = WhisperProcessor.from_pretrained(DEFAULT_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(DEFAULT_MODEL)
    model.to(DEFAULT_DEVICE)
    model.eval()

    code_to_id, id_to_code = _build_lang_maps(processor, model)
    lang_token_ids = list(code_to_id.values())

    results = []
    for wav_path in wav_files:
        waveform, sample_rate = torchaudio.load(wav_path)
        language_code, confidence = _detect_language(
            waveform=waveform,
            sample_rate=sample_rate,
            processor=processor,
            model=model,
            device=DEFAULT_DEVICE,
            lang_token_ids=lang_token_ids,
            id_to_code=id_to_code,
        )

        file_id = os.path.basename(wav_path)
        results.append(
            {
                "file": file_id,
                "language": language_code,
                "confidence": confidence,
            }
        )

        print(f"{file_id:25s} -> {language_code} ({confidence:.3f})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print("\nDone.")
    print(f"Saved predictions to: {output_path}")
    print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
