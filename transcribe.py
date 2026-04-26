#!/usr/bin/env python3
"""Transcribe spoken words from an MP4 video file to text."""

import argparse
import os
import sys
import tempfile
import subprocess
from pathlib import Path


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
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}'...", file=sys.stderr)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    kwargs = {}
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0,
        **kwargs,
    )

    print(
        f"Detected language '{info.language}' with probability {info.language_probability:.2f}",
        file=sys.stderr,
    )

    parts = []
    for segment in segments:
        parts.append(segment.text.strip())

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
        print(transcript)


if __name__ == "__main__":
    main()
