#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import torchaudio
import pandas as pd

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

DEFAULT_MODEL = "facebook/mms-lid-256"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMS language ID on wav files in a directory.")
    parser.add_argument("audio_dir", help="Directory containing .wav files.")
    parser.add_argument(
        "--output",
        default="language_predictions.csv",
        help="Output CSV path (default: language_predictions.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"Audio directory does not exist: {audio_dir}", file=sys.stderr)
        return 1

    device = DEFAULT_DEVICE

    wav_files = sorted(glob.glob(os.path.join(str(audio_dir), "*.wav")))
    if not wav_files:
        print(f"No .wav files found in: {audio_dir}", file=sys.stderr)
        return 1

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(DEFAULT_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(DEFAULT_MODEL)
    model.to(device)
    model.eval()
    id2label = model.config.id2label

    results = []
    for wav_path in wav_files:
        waveform, sr = torchaudio.load(wav_path)

        # Convert to mono for stable feature extraction.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        waveform = waveform.squeeze().numpy()
        inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred = logits.argmax(-1).item()
        language = id2label[pred]
        confidence = torch.softmax(logits, dim=-1)[0, pred].item()

        file_id = os.path.basename(wav_path)
        results.append(
            {
                "file": file_id,
                "language": language,
                "confidence": confidence,
            }
        )

        print(f"{file_id:25s} -> {language} ({confidence:.3f})")

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