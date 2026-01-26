# Rick TTS - Voice Cloning CLI

A simple command-line tool for cloning voices and generating speech using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS).

Clone any voice from a short audio sample, then generate unlimited speech in that voice.

Includes both a **CLI** (`main.py`) and a **Web UI** (`app.py`).

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
git clone git@github.com:dattasaurabh82/rickgemma.git
cd rickgemma

# Switch to voice branch (TTS code is here)
git checkout voice

# Install dependencies
uv sync
```

> [!Note]
> This project uses [uv](https://github.com/astral-sh/uv) for Python package management. 
> Install it first if you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Download Models

You need at least one model. The 1.7B model has better quality, 0.6B is faster.

```bash
# 1.7B model (recommended, ~3.5GB)
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base

# 0.6B model (lighter, ~1.2GB)
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

See [Qwen3-TTS on Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS) for more details.

> [!Tip]
> Start with the 0.6B model for testing — it's faster to download and run.

### Prepare voice cloning data

> [!Important]
> May vary how you get the .wav audio data and the respective transcript in txt format, bit one method is listed below - to use any youtube speaker

#### Audio Clips (2 separate clips)

Although 1 clip fo 15 sec is okay

> [!Warning]
> Make sure you have `ffmpeg` installed

```bash
cd data

# Clip 1: First 10 seconds (0:00 - 0:10)
yt-dlp -x --audio-format wav -f bestaudio --download-sections "*00:00:00-00:00:10" -o "clip_01_10sec.%(ext)s" "[A YOUTUBE LINK OF A PERSON SPEAKING WHOSE VOICE YOU WANT TO CLONE]

# Clip 2: Next 15 seconds (0:10 - 0:25)
yt-dlp -x --audio-format wav -f bestaudio --download-sections "*00:00:10-00:00:25" -o "clip_02_15sec.%(ext)s" "[A YOUTUBE LINK OF A PERSON SPEAKING WHOSE VOICE YOU WANT TO CLONE]
```

#### Transcript Clips (extract text by timestamp)

First download the full subtitle, then use awk to extract by time range

```bash
# Step 1: Download full subtitles
yt-dlp --skip-download --write-auto-subs --sub-lang en --convert-subs srt -o "full" "[A YOUTUBE LINK OF A PERSON SPEAKING WHOSE VOICE YOU WANT TO CLONE]"

# Step 2: Extract transcript for 0:00-0:10 (clip 1)
awk '/^00:00:0[0-9],[0-9]+ -->/{p=1; next} /^00:00:[1-9][0-9],[0-9]+ -->/{p=0} /^[0-9]+$/{next} /^$/{next} /-->/{next} p' full_en.srt | awk '!seen[$0]++ && NF' > data/transcript_clip_01.txt

# Step 3: Extract transcript for 0:10-0:25 (clip 2)
awk '/^00:00:(1[0-9]|2[0-4]),[0-9]+ -->/{p=1; next} /^00:00:2[5-9],[0-9]+ -->/{p=0} /^00:00:[3-9][0-9]+ -->/{p=0} /^[0-9]+$/{next} /^$/{next} /-->/{next} p' full_en.srt | awk '!seen[$0]++ && NF' > data/transcript_clip_02.txt

rm full_en.srt
cd ..
```

---

## Usage

### Web Interface (Gradio)

```bash
# Local only
uv run app.py

# Accessible on network
uv run app.py --host 0.0.0.0 --port 7860

# Public share link
uv run app.py --share
```

Opens at `http://127.0.0.1:7860` (or your specified host/port).

### CLI Commands

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
├── main.py                          # CLI
├── app.py                           # Web UI (Gradio)
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
