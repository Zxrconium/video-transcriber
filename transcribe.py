#!/usr/bin/env python3
"""Transcribe spoken words from an MP4 video file to text."""

import argparse
import os
import sys
import tempfile
import subprocess
import warnings
from pathlib import Path

# Suppress HuggingFace Hub warnings that appear when no HF_TOKEN is set and on
# Windows systems where symlinks require Developer Mode or administrator rights.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated.*",
    category=UserWarning,
    module="huggingface_hub",
)


def extract_audio(video_path: str, audio_path: str) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def transcribe(audio_path: str, model_size: str, language: str | None) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Error: faster-whisper is not installed.\n"
            "Run: pip install faster-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading Whisper model '{model_size}'...", file=sys.stderr)
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if language:
        kwargs["language"] = language

    print("Transcribing...", file=sys.stderr)
    try:
        segments, info = model.transcribe(audio_path, beam_size=5, **kwargs)

        print(
            f"Detected language '{info.language}' with probability {info.language_probability:.2f}",
            file=sys.stderr,
        )

        parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
                print(f"  [{segment.start:.1f}s → {segment.end:.1f}s] {text}", file=sys.stderr)

    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)

    if not parts:
        print(
            "Warning: no speech detected. The audio may be silent, too noisy, or in an "
            "unsupported language. Try a larger model (e.g. -m small) or force a language "
            "with -l en.",
            file=sys.stderr,
        )

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe spoken words from an MP4 video to text."
    )
    parser.add_argument("video", help="Path to the MP4 (or any ffmpeg-supported) video file")
    parser.add_argument(
        "-o", "--output",
        help="Write transcript to this file instead of stdout",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Force a language code, e.g. 'en', 'fr'. Auto-detected when omitted.",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        print(f"Extracting audio from '{video_path}'...", file=sys.stderr)
        extract_audio(video_path, audio_path)

        transcript = transcribe(audio_path, args.model, args.language)
    finally:
        os.unlink(audio_path)

    if args.output:
        Path(args.output).write_text(transcript + "\n", encoding="utf-8")
        print(f"Transcript saved to '{args.output}'", file=sys.stderr)
    else:
        if transcript:
            print(transcript)


if __name__ == "__main__":
    main()
