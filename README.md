# Rick TTS - Voice Cloning CLI

A simple command-line tool for cloning voices and generating speech using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS).

Clone any voice from a short audio sample, then generate unlimited speech in that voice.

---

## Features

- **Voice Cloning** — Create a voice profile from 5-15 seconds of audio
- **Cached Voices** — Clone once, use forever
- **Multiple Profiles** — Save different voices by name
- **Model Options** — Choose between 0.6B (fast) or 1.7B (quality)
- **Cross-Platform** — Works on Mac (MPS), Linux (CUDA), and CPU

---

## Quick Start

### 1. Clone a Voice

```bash
uv run main.py --clone \
  --voice-name datta \
  --ref-audio data/clip_01_10sec.wav \
  --ref-text data/transcript_clip_01.txt \
  --text "Wubba lubba dub dub!"
```

### 2. Use the Voice

```bash
uv run main.py --voice-name datta --text "Hello from the cloned voice!"
```

That's it! The voice is cached and ready to use anytime.

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd rickgemma

# Install dependencies (using uv)
uv sync
```

### Download Models

You need at least one model. The 1.7B model has better quality, 0.6B is faster.

```bash
# 1.7B model (recommended)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base

# 0.6B model (lighter)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

> [!Tip]
> Start with the 0.6B model for testing — it's faster to download and run.

---

## Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `--clone` | Create a new voice profile |
| `--voice-name <n>` | Name of the voice profile |
| `--text "..."` | Text to speak |
| `--list-voices` | Show all saved voices |
| `--clear-cache` | Delete saved voices |

### Clone a New Voice

```bash
uv run main.py --clone \
  --voice-name morty \
  --ref-audio samples/morty_clip.wav \
  --ref-text samples/morty_transcript.txt \
  --text "Oh geez Rick!"
```

> [!Important]
> You need **two files** to clone a voice:
> - A WAV audio file (5-15 seconds of clear speech)
> - A TXT file with the exact words spoken in the audio

### Generate Speech

```bash
uv run main.py --voice-name morty --text "Any text you want"
```

### List Saved Voices

```bash
uv run main.py --list-voices
```

Output:
```
Saved voices:
============================================================
  morty        model:0.6B  created:2026-01-26 18:36  ref:morty_clip.wav
  datta        model:1.7B  created:2026-01-26 18:34  ref:clip_01_10sec.wav
============================================================
```

### Delete Voices

```bash
# Delete specific voice
uv run main.py --clear-cache --voice-name morty

# Delete all voices (asks for confirmation)
uv run main.py --clear-cache
```

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--text` | required | Text to synthesize |
| `--voice-name` | `default` | Voice profile name |
| `--model` | `1.7B` | Model size: `0.6B` or `1.7B` |
| `--output` | `outputs/output.wav` | Output file path |
| `--language` | `English` | Language for synthesis |
| `--ref-audio` | — | Reference audio for cloning |
| `--ref-text` | — | Transcript of reference audio |

### Supported Languages

English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Auto

---

## Directory Structure

```
rickgemma/
├── models/                          # Downloaded models
│   ├── Qwen3-TTS-12Hz-0.6B-Base/
│   └── Qwen3-TTS-12Hz-1.7B-Base/
├── data/
│   ├── voice_clone_cache/           # Saved voice profiles
│   │   ├── datta.pt                 # Voice tensors
│   │   └── datta.json               # Metadata
│   └── *.wav, *.txt                 # Your reference files
├── outputs/                         # Generated audio
│   └── output.wav
├── logs/                            # Log files
│   └── tts.log                      # Last run log (overwritten each run)
├── main.py                          # Main CLI
└── pyproject.toml
```

---

## Tips for Best Results

### Reference Audio

> [!Tip]
> **Ideal reference audio:**
> - Duration: 5-15 seconds
> - Clear speech, minimal background noise
> - Single speaker
> - Natural speaking pace

### Transcript Accuracy

> [!Warning]
> The transcript must **exactly match** what's spoken in the audio. 
> Mismatched transcripts produce poor voice clones.

Keep punctuation — it affects prosody:
```
Good: "Hello, my name is Rick. How are you?"
Bad:  "hello my name is rick how are you"
```

### Model Compatibility

> [!Important]
> Voice profiles are tied to the model used during cloning.
> A voice cloned with `1.7B` won't work with `0.6B` and vice versa.

---

## Device Support

| Device | Status | Notes |
|--------|--------|-------|
| NVIDIA GPU (CUDA) | Yes | Best performance, uses bfloat16 |
| Apple Silicon (MPS) | Yes | Uses float32 (required for voice cloning) |
| CPU | Yes | Slow but works |

The script auto-detects your hardware.

---

## Troubleshooting

### "Model not found"

Download the model first:
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

### "Model mismatch"

You're using a voice cloned with a different model size. Options:
1. Use the same model: `--model 1.7B`
2. Re-clone the voice with your preferred model

---

## License

MIT

---

## Credits

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) by Alibaba
- Built with [uv](https://github.com/astral-sh/uv) for Python packaging
