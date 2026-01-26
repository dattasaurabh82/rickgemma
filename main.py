#!/usr/bin/env python3
"""
rick_tts_headless.py - Qwen3-TTS Voice Cloning CLI

Clone voices and generate speech from the command line.
"""

import os
import sys
import argparse
import warnings
import atexit
import signal
import re
import json
from pathlib import Path
from datetime import datetime

# Suppress warnings before torch import
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*Flash Attention.*")
warnings.filterwarnings("ignore", message=".*SoX.*")

import torch
import soundfile as sf
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# =============================================================================
# PATHS & CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "data" / "voice_clone_cache"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "tts.log"

MODEL_VARIANTS = {
    "0.6B": "Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen3-TTS-12Hz-1.7B-Base",
}

# Global state for cleanup
_model = None
_device = None
_log_file = None


# =============================================================================
# LOGGING (simple: terminal with colors + file)
# =============================================================================
def log(msg, level="info"):
    """Print to terminal (with color) and log file."""
    global _log_file
    
    # Color mapping
    colors = {
        "info": Fore.WHITE,
        "success": Fore.GREEN,
        "warning": Fore.LIGHTRED_EX,
        "error": Fore.RED + Style.BRIGHT,
        "header": Fore.CYAN + Style.BRIGHT,
        "dim": Style.DIM,
    }
    color = colors.get(level, "")
    
    # Print to terminal
    print(f"{color}{msg}{Style.RESET_ALL}")
    
    # Write to log file
    if _log_file:
        _log_file.write(f"{msg}\n")
        _log_file.flush()


def init_logging():
    """Initialize log file (overwrites previous)."""
    global _log_file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_FILE, "w", encoding="utf-8")
    _log_file.write(f"=== Qwen3-TTS Log - {datetime.now().isoformat()} ===\n\n")


def close_logging():
    """Close log file."""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


# =============================================================================
# CLEANUP
# =============================================================================
def cleanup():
    """Release model and clear GPU memory."""
    global _model, _device
    
    if _model is not None:
        log("\n[Cleanup] Releasing model...", "dim")
        del _model
        _model = None
    
    if _device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif _device and "cuda" in _device:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    close_logging()


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    log(f"\n[Signal] Interrupted, cleaning up...", "warning")
    cleanup()
    sys.exit(0)


atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# PATH HELPERS
# =============================================================================
def get_voice_cache_path(voice_name: str) -> Path:
    return CACHE_DIR / f"{voice_name}.pt"


def get_voice_meta_path(voice_name: str) -> Path:
    return CACHE_DIR / f"{voice_name}.json"


def get_model_path(model_size: str) -> Path:
    return MODELS_DIR / MODEL_VARIANTS[model_size]


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Cloning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone a voice (first time)
  python rick_tts_headless.py --clone --voice-name rick \\
    --ref-audio voice.wav --ref-text transcript.txt --text "Hello"
  
  # Use cached voice
  python rick_tts_headless.py --voice-name rick --text "Hello again"
  
  # List saved voices
  python rick_tts_headless.py --list-voices
        """
    )
    
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--list-voices", action="store_true", help="List saved voices")
    parser.add_argument("--clear-cache", action="store_true", help="Delete cached voice(s)")
    parser.add_argument("--clone", action="store_true", help="Clone a new voice")
    parser.add_argument("--voice-name", type=str, default="default", help="Voice profile name")
    parser.add_argument("--ref-audio", type=str, help="Reference audio WAV file")
    parser.add_argument("--ref-text", type=str, help="Reference transcript TXT file")
    parser.add_argument("--output", type=str, help="Output WAV path (default: outputs/output.wav)")
    parser.add_argument("--language", type=str, default="English", help="Language (default: English)")
    parser.add_argument("--model", choices=["0.6B", "1.7B"], default="1.7B", help="Model size")
    
    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations."""
    if args.list_voices or args.clear_cache:
        return
    
    errors = []
    
    if not args.text:
        errors.append("--text is required")
    
    if args.clone:
        if not args.ref_audio:
            errors.append("--clone requires --ref-audio")
        if not args.ref_text:
            errors.append("--clone requires --ref-text")
        if args.ref_audio and not Path(args.ref_audio).exists():
            errors.append(f"File not found: {args.ref_audio}")
        if args.ref_text and not Path(args.ref_text).exists():
            errors.append(f"File not found: {args.ref_text}")
    
    model_path = get_model_path(args.model)
    if not model_path.exists():
        errors.append(f"Model not found: {model_path}")
        errors.append(f"  Download: huggingface-cli download Qwen/{MODEL_VARIANTS[args.model]} --local-dir {model_path}")
    
    if errors:
        for e in errors:
            log(f"ERROR: {e}", "error")
        sys.exit(1)


# =============================================================================
# DEVICE & MODEL
# =============================================================================
def get_device_and_dtype():
    """
    Detect best device. 
    IMPORTANT: MPS/CPU require float32 for voice cloning (float16 causes NaN errors).
    """
    global _device
    
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_cuda:
        device, dtype, flash_attn = "cuda:0", torch.bfloat16, True
        log("[Device] CUDA → bfloat16 + flash_attn", "info")
    elif has_mps:
        device, dtype, flash_attn = "mps", torch.float32, False
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        log("[Device] MPS → float32", "info")
    else:
        device, dtype, flash_attn = "cpu", torch.float32, False
        log("[Device] CPU → float32", "info")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _device = device
    return device, dtype, flash_attn


def load_model(device, dtype, use_flash_attn, model_size):
    """Load Qwen3-TTS model."""
    global _model
    from qwen_tts import Qwen3TTSModel
    
    model_path = get_model_path(model_size)
    log(f"[Model] Loading {model_size} from {model_path.name}...", "info")
    
    kwargs = {"device_map": device, "dtype": dtype}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    
    model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
    _model = model
    log("[Model] Loaded", "success")
    return model


# =============================================================================
# VOICE CACHE
# =============================================================================
def load_transcript(txt_path):
    """Load and clean transcript text."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    text = text.replace('\n', ' ').replace('\r', ' ')
    return re.sub(r' +', ' ', text)


def save_voice_prompt(prompt, voice_name, model_size, ref_audio):
    """Save voice prompt and metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    torch.save(prompt, get_voice_cache_path(voice_name))
    
    meta = {
        "model_size": model_size,
        "created_at": datetime.now().isoformat(),
        "ref_audio": str(ref_audio),
    }
    with open(get_voice_meta_path(voice_name), 'w') as f:
        json.dump(meta, f, indent=2)
    
    log(f"[Cache] Saved voice '{voice_name}'", "success")


def load_voice_metadata(voice_name):
    """Load voice metadata if exists."""
    meta_path = get_voice_meta_path(voice_name)
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def load_voice_prompt(voice_name, device, model_size):
    """Load voice prompt with model validation."""
    meta = load_voice_metadata(voice_name)
    
    if meta and meta.get("model_size") != model_size:
        log(f"\n[ERROR] Model mismatch!", "error")
        log(f"  Voice '{voice_name}' was cloned with: {meta['model_size']}", "error")
        log(f"  You requested: {model_size}", "error")
        log(f"\n  Options:", "warning")
        log(f"    1. Use --model {meta['model_size']}", "info")
        log(f"    2. Re-clone with --clone --model {model_size}", "info")
        sys.exit(1)
    
    prompt = torch.load(get_voice_cache_path(voice_name), map_location=device, weights_only=False)
    log(f"[Cache] Loaded voice '{voice_name}'", "success")
    return prompt


def has_cached_voice(voice_name):
    return get_voice_cache_path(voice_name).exists()


# =============================================================================
# TTS GENERATION
# =============================================================================
def clone_voice_and_generate(model, args, device):
    """Clone voice, cache it, and generate speech."""
    log(f"\n[Clone] Voice: {args.voice_name}", "header")
    log(f"  Audio: {args.ref_audio}", "dim")
    log(f"  Text:  {args.ref_text}", "dim")
    
    ref_text = load_transcript(args.ref_text)
    log(f"  Transcript: \"{ref_text[:60]}{'...' if len(ref_text) > 60 else ''}\"", "dim")
    
    log("\n[Clone] Creating voice prompt...", "info")
    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    
    save_voice_prompt(prompt, args.voice_name, args.model, args.ref_audio)
    generate_speech(model, args, prompt)


def generate_with_cached_voice(model, args, device):
    """Generate speech with cached voice."""
    log(f"\n[Cached] Voice: {args.voice_name}", "header")
    prompt = load_voice_prompt(args.voice_name, device, args.model)
    generate_speech(model, args, prompt)


def generate_speech(model, args, prompt):
    """Generate and save speech."""
    log(f"\n[TTS] Generating...", "info")
    log(f"  Text: \"{args.text}\"", "dim")
    
    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        voice_clone_prompt=prompt,
    )
    
    output_path = Path(args.output) if args.output else OUTPUTS_DIR / "output.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(str(output_path), wavs[0], sr)
    
    duration = len(wavs[0]) / sr
    log(f"\n[Output] {output_path}", "success")
    log(f"  Duration: {duration:.2f}s @ {sr}Hz", "dim")


# =============================================================================
# LIST & CLEAR COMMANDS
# =============================================================================
def list_voices():
    """List all saved voice profiles."""
    log("\nSaved voices:", "header")
    log("=" * 60, "dim")
    
    if not CACHE_DIR.exists():
        log("  (none)", "dim")
        log("\n  Clone with: --clone --voice-name <name> --ref-audio <wav> --ref-text <txt>", "info")
        return
    
    voice_files = list(CACHE_DIR.glob("*.pt"))
    if not voice_files:
        log("  (none)", "dim")
        return
    
    for pt in sorted(voice_files):
        name = pt.stem
        meta = load_voice_metadata(name)
        if meta:
            model = meta.get("model_size", "?")
            created = meta.get("created_at", "?")[:16].replace("T", " ")
            ref = Path(meta.get("ref_audio", "?")).name
            log(f"  {name:<12} {Fore.CYAN}model:{model}  {Fore.WHITE}created:{created}  ref:{ref}", "info")
        else:
            log(f"  {name:<12} (legacy - no metadata)", "dim")
    
    log("=" * 60, "dim")


def clear_cache(voice_name=None):
    """Delete cached voice(s)."""
    if not CACHE_DIR.exists():
        log("\n[Clear] No cache found.", "warning")
        return
    
    # Specific voice
    if voice_name:
        pt = get_voice_cache_path(voice_name)
        if not pt.exists():
            log(f"\n[Clear] Voice '{voice_name}' not found.", "warning")
            return
        
        pt.unlink()
        meta = get_voice_meta_path(voice_name)
        if meta.exists():
            meta.unlink()
        
        log(f"\n[Clear] Deleted '{voice_name}'", "success")
        return
    
    # All voices
    voice_files = list(CACHE_DIR.glob("*.pt"))
    if not voice_files:
        log("\n[Clear] No voices cached.", "warning")
        return
    
    log(f"\n[Clear] Delete {len(voice_files)} voice(s)?", "warning")
    for pt in sorted(voice_files):
        log(f"  - {pt.stem}", "dim")
    
    confirm = input(f"\n{Fore.YELLOW}Type 'yes' to confirm: {Style.RESET_ALL}").strip().lower()
    if confirm != "yes":
        log("[Clear] Cancelled.", "info")
        return
    
    for pt in voice_files:
        pt.unlink()
        meta = CACHE_DIR / f"{pt.stem}.json"
        if meta.exists():
            meta.unlink()
    
    log(f"[Clear] Deleted {len(voice_files)} voice(s).", "success")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    validate_args(args)
    
    # Quick commands (no model needed)
    if args.list_voices:
        list_voices()
        return
    
    if args.clear_cache:
        voice = args.voice_name if args.voice_name != "default" else None
        clear_cache(voice)
        return
    
    # Initialize logging for TTS operations
    init_logging()
    
    # Determine mode
    if args.clone:
        mode = "clone"
    elif has_cached_voice(args.voice_name):
        mode = "cached"
    else:
        log(f"\n[ERROR] Voice '{args.voice_name}' not found!", "error")
        log(f"\n  Clone first with:", "info")
        log(f"    --clone --voice-name {args.voice_name} --ref-audio <wav> --ref-text <txt> --text <text>", "dim")
        sys.exit(1)
    
    # Header
    log(f"\n{'='*60}", "header")
    log(f"  Qwen3-TTS Voice Cloning", "header")
    log(f"{'='*60}", "header")
    log(f"  Mode:  {mode.upper()}", "info")
    log(f"  Voice: {args.voice_name}", "info")
    log(f"  Model: {args.model}", "info")
    log(f"{'='*60}", "header")
    
    # Load model and generate
    device, dtype, flash_attn = get_device_and_dtype()
    model = load_model(device, dtype, flash_attn, args.model)
    
    if mode == "clone":
        clone_voice_and_generate(model, args, device)
    else:
        generate_with_cached_voice(model, args, device)
    
    log(f"\n{'='*60}", "header")
    log("Done!", "success")


if __name__ == "__main__":
    main()
