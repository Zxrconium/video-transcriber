# Video Transcriber

Transcribe spoken words from an MP4 (or any ffmpeg-supported) video file to text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Requirements

- Python 3.10+
- ffmpeg installed on your system (`apt install ffmpeg` / `brew install ffmpeg`)

```
pip install -r requirements.txt
```

## Usage

```
python3 transcribe.py <video.mp4> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `-l`, `--language` | auto | Force language code, e.g. `en`, `fr`, `de` |
| `-o`, `--output` | stdout | Save transcript to a file |

### Examples

```bash
# Print transcript to terminal
python3 transcribe.py lecture.mp4

# Save to a text file, use a more accurate model
python3 transcribe.py interview.mp4 -m small -o transcript.txt

# Force English, largest model
python3 transcribe.py video.mp4 -m large-v3 -l en -o out.txt
```

## How it works

1. **ffmpeg** extracts the audio track as a 16 kHz mono WAV.
2. **faster-whisper** (CTranslate2-optimised Whisper) transcribes the audio on CPU using int8 quantisation.
3. The transcript is printed to stdout or written to the specified file.

The Whisper model is downloaded automatically on first run and cached in `~/.cache/huggingface/`.
