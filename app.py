#!/usr/bin/env python3
"""
app.py - Gradio interface for Qwen3-TTS Voice Cloning
"""

import os
import sys
import warnings
import json
import re
import atexit
from pathlib import Path
from datetime import datetime

# Suppress warnings before torch import
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*Flash Attention.*")
warnings.filterwarnings("ignore", message=".*SoX.*")

import torch
import soundfile as sf
import gradio as gr

# =============================================================================
# PATHS & CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "data" / "voice_clone_cache"

MODEL_VARIANTS = {
    "0.6B": "Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen3-TTS-12Hz-1.7B-Base",
}

LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", 
    "French", "German", "Spanish", "Portuguese", "Russian", "Auto"
]

# Global model cache
_model = None
_model_size = None
_device = None


# =============================================================================
# CLEANUP
# =============================================================================
def cleanup():
    """Release model and clear GPU memory."""
    global _model, _device
    
    if _model is not None:
        print("[Cleanup] Releasing model...")
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
    
    print("[Cleanup] Done.")


atexit.register(cleanup)


# =============================================================================
# DEVICE & MODEL
# =============================================================================
def get_device_and_dtype():
    """Detect best device. MPS/CPU require float32 for voice cloning."""
    global _device
    
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_cuda:
        device, dtype, flash_attn = "cuda:0", torch.bfloat16, True
    elif has_mps:
        device, dtype, flash_attn = "mps", torch.float32, False
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device, dtype, flash_attn = "cpu", torch.float32, False
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _device = device
    return device, dtype, flash_attn


def get_model_path(model_size: str) -> Path:
    return MODELS_DIR / MODEL_VARIANTS[model_size]


def load_model(model_size: str):
    """Load model (cached - only reloads if size changes)."""
    global _model, _model_size
    
    if _model is not None and _model_size == model_size:
        return _model
    
    # Unload previous model
    if _model is not None:
        del _model
        _model = None
        if _device == "mps":
            torch.mps.empty_cache()
        elif _device and "cuda" in _device:
            torch.cuda.empty_cache()
    
    from qwen_tts import Qwen3TTSModel
    
    model_path = get_model_path(model_size)
    if not model_path.exists():
        raise gr.Error(f"Model not found: {model_path}\nDownload with: hf download Qwen/{MODEL_VARIANTS[model_size]} --local-dir {model_path}")
    
    device, dtype, flash_attn = get_device_and_dtype()
    
    kwargs = {"device_map": device, "dtype": dtype}
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    
    _model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
    _model_size = model_size
    
    return _model


# =============================================================================
# VOICE CACHE HELPERS
# =============================================================================
def get_voice_cache_path(voice_name: str) -> Path:
    return CACHE_DIR / f"{voice_name}.pt"


def get_voice_meta_path(voice_name: str) -> Path:
    return CACHE_DIR / f"{voice_name}.json"


def load_voice_metadata(voice_name: str) -> dict:
    meta_path = get_voice_meta_path(voice_name)
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def get_cached_voices() -> list:
    """Get list of cached voice names."""
    if not CACHE_DIR.exists():
        return []
    return sorted([p.stem for p in CACHE_DIR.glob("*.pt")])


def get_voices_table() -> list:
    """Get voice data for display table."""
    voices = get_cached_voices()
    if not voices:
        return []
    
    data = []
    for name in voices:
        meta = load_voice_metadata(name)
        model = meta.get("model_size", "?")
        created = meta.get("created_at", "?")[:16].replace("T", " ")
        ref = Path(meta.get("ref_audio", "?")).name if meta.get("ref_audio") else "?"
        data.append([name, model, created, ref])
    
    return data


def clean_transcript(text: str) -> str:
    """Clean transcript text."""
    text = text.strip()
    text = text.replace('\n', ' ').replace('\r', ' ')
    return re.sub(r' +', ' ', text)


# =============================================================================
# GRADIO FUNCTIONS
# =============================================================================
def clone_voice(audio_file, transcript_text, transcript_file, voice_name, model_size, language):
    """Clone a voice from reference audio."""
    try:
        if not audio_file:
            raise gr.Error("Please upload a reference audio file")
        
        if not voice_name or not voice_name.strip():
            raise gr.Error("Please enter a voice name")
        
        voice_name = voice_name.strip().lower().replace(" ", "_")
        
        # Get transcript
        if transcript_file is not None:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
        elif transcript_text and transcript_text.strip():
            transcript = transcript_text
        else:
            raise gr.Error("Please provide transcript text or upload a transcript file")
        
        transcript = clean_transcript(transcript)
        if not transcript:
            raise gr.Error("Transcript is empty")
        
        # Load model
        model = load_model(model_size)
        
        # Create voice prompt
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=audio_file,
            ref_text=transcript,
            x_vector_only_mode=False,
        )
        
        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(voice_prompt, get_voice_cache_path(voice_name))
        
        meta = {
            "model_size": model_size,
            "created_at": datetime.now().isoformat(),
            "ref_audio": str(audio_file),
        }
        with open(get_voice_meta_path(voice_name), 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Return updated UI
        voices = get_cached_voices()
        gr.Info(f"Voice '{voice_name}' cloned successfully!", duration=5)
        return (
            gr.update(choices=voices, value=voice_name),
            gr.update(value=get_voices_table()),
            f"Voice '{voice_name}' cloned successfully!",
            gr.update(interactive=False),
            gr.update(interactive=True),
            None,
        )
    except gr.Error:
        raise
    except Exception as e:
        error_msg = f"Clone failed: {str(e)}"
        gr.Warning(error_msg, duration=10)
        return (
            gr.update(),
            gr.update(),
            error_msg,
            gr.update(),
            gr.update(),
            None,
        )


def generate_speech(voice_name, text, model_size, language):
    """Generate speech with cached voice."""
    try:
        if not voice_name:
            raise gr.Error("Please select a voice")
        
        if not text or not text.strip():
            raise gr.Error("Please enter text to synthesize")
        
        cache_path = get_voice_cache_path(voice_name)
        if not cache_path.exists():
            raise gr.Error(f"Voice '{voice_name}' not found")
        
        # Check model compatibility
        meta = load_voice_metadata(voice_name)
        if meta and meta.get("model_size") != model_size:
            raise gr.Error(f"Voice '{voice_name}' was cloned with {meta['model_size']} model. Please select {meta['model_size']} or re-clone the voice.")
        
        # Load model and voice prompt
        model = load_model(model_size)
        device, _, _ = get_device_and_dtype()
        voice_prompt = torch.load(cache_path, map_location=device, weights_only=False)
        
        # Generate
        wavs, sr = model.generate_voice_clone(
            text=text.strip(),
            language=language,
            voice_clone_prompt=voice_prompt,
        )
        
        # Save output
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{voice_name}_{datetime.now().strftime('%H%M%S')}.wav"
        sf.write(str(output_path), wavs[0], sr)
        
        gr.Info(f"Generated {output_path.name}", duration=3)
        return str(output_path)
    
    except gr.Error:
        raise
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        gr.Warning(error_msg, duration=10)
        raise gr.Error(error_msg)


def delete_voice(voice_name):
    """Delete a cached voice."""
    try:
        if not voice_name:
            raise gr.Error("Please select a voice to delete")
        
        pt_file = get_voice_cache_path(voice_name)
        meta_file = get_voice_meta_path(voice_name)
        
        if pt_file.exists():
            pt_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
        
        voices = get_cached_voices()
        gr.Info(f"Voice '{voice_name}' deleted", duration=3)
        return (
            gr.update(choices=voices, value=voices[0] if voices else None),
            gr.update(value=get_voices_table()),
            f"Voice '{voice_name}' deleted.",
            gr.update(interactive=False),
            gr.update(interactive=len(voices) > 0),
            None,
        )
    except gr.Error:
        raise
    except Exception as e:
        error_msg = f"Delete failed: {str(e)}"
        gr.Warning(error_msg, duration=10)
        return (
            gr.update(),
            gr.update(),
            error_msg,
            gr.update(),
            gr.update(),
            None,
        )


def clear_all_voices():
    """Delete all cached voices."""
    try:
        voices = get_cached_voices()
        if not voices:
            gr.Info("No voices to delete", duration=3)
            return (
                gr.update(choices=[], value=None),
                gr.update(value=[]),
                "No voices to delete.",
                gr.update(interactive=False),
                gr.update(interactive=False),
                None,
            )
        
        count = 0
        for voice_name in voices:
            pt_file = get_voice_cache_path(voice_name)
            meta_file = get_voice_meta_path(voice_name)
            if pt_file.exists():
                pt_file.unlink()
                count += 1
            if meta_file.exists():
                meta_file.unlink()
        
        gr.Info(f"Deleted {count} voice(s)", duration=3)
        return (
            gr.update(choices=[], value=None),
            gr.update(value=[]),
            f"Deleted {count} voice(s).",
            gr.update(interactive=False),
            gr.update(interactive=False),
            None,
        )
    except Exception as e:
        error_msg = f"Clear failed: {str(e)}"
        gr.Warning(error_msg, duration=10)
        return (
            gr.update(),
            gr.update(),
            error_msg,
            gr.update(),
            gr.update(),
            None,
        )


def refresh_voices():
    """Refresh the voices list."""
    voices = get_cached_voices()
    return (
        gr.update(choices=voices, value=voices[0] if voices else None),
        gr.update(value=get_voices_table()),
        gr.update(interactive=False),           # delete_btn (no selection after refresh)
        gr.update(interactive=len(voices) > 0), # clear_all_btn
        None,                                   # selected_voice reset
    )


def load_transcript_file(file):
    """Load transcript from uploaded file."""
    if file is None:
        return ""
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


# =============================================================================
# GRADIO UI
# =============================================================================
def create_ui():
    # Get initial data
    initial_voices = get_cached_voices()
    initial_table = get_voices_table()
    
    # Theme: Monochrome with mono fonts
    theme = gr.themes.Monochrome(
        font=gr.themes.GoogleFont("JetBrains Mono"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )
    
    with gr.Blocks(title="Voice Cloning with Qwen3-TTS", theme=theme) as app:
        gr.Markdown("# Voice Cloning with Qwen3-TTS")
        gr.Markdown("Clone voices and generate speech using [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts)")
        
        # ----- CLONE SECTION -----
        with gr.Group():
            gr.Markdown("## Clone a Voice")
            
            with gr.Row():
                with gr.Column(scale=1):
                    clone_audio = gr.Audio(
                        label="Reference Audio (5-15 seconds)",
                        type="filepath",
                        sources=["upload"],
                    )
                
                with gr.Column(scale=1):
                    clone_transcript_file = gr.File(
                        label="Transcript File (optional)",
                        file_types=[".txt"],
                    )
                    clone_transcript = gr.Textbox(
                        label="Transcript Text",
                        placeholder="Type or paste the exact words spoken in the audio...",
                        lines=3,
                    )
            
            with gr.Row():
                clone_voice_name = gr.Textbox(
                    label="Voice Name",
                    placeholder="e.g., rick, morgan, custom_voice",
                    scale=2,
                )
                clone_model = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_VARIANTS.keys()),
                    value="1.7B",
                    scale=1,
                )
                clone_language = gr.Dropdown(
                    label="Language",
                    choices=LANGUAGES,
                    value="English",
                    scale=1,
                )
            
            clone_btn = gr.Button("Clone Voice", variant="primary")
            clone_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # ----- GENERATE SECTION -----
        with gr.Group():
            gr.Markdown("## Generate Speech")
            
            with gr.Row():
                gen_voice = gr.Dropdown(
                    label="Voice",
                    choices=initial_voices,
                    value=initial_voices[0] if initial_voices else None,
                    scale=2,
                )
                gen_model = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_VARIANTS.keys()),
                    value="1.7B",
                    scale=1,
                )
                gen_language = gr.Dropdown(
                    label="Language",
                    choices=LANGUAGES,
                    value="English",
                    scale=1,
                )
            
            gen_text = gr.Textbox(
                label="Text to Speak",
                placeholder="Enter the text you want to synthesize...",
                lines=3,
            )
            
            gen_btn = gr.Button("Generate Speech", variant="primary")
            gen_output = gr.Audio(label="Generated Audio", type="filepath")
        
        gr.Markdown("---")
        
        # ----- MANAGE SECTION -----
        with gr.Group():
            gr.Markdown("## Manage Voices")
            
            # State to track selected voice
            selected_voice = gr.State(None)
            
            voices_table = gr.Dataframe(
                headers=["Name", "Model", "Created", "Reference Audio"],
                value=initial_table,
                interactive=False,
                label="Click a row to select",
            )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh", scale=1)
                delete_btn = gr.Button("Delete Selected", variant="stop", scale=1, interactive=False)
                clear_all_btn = gr.Button("Clear All", variant="stop", scale=1, interactive=len(initial_voices) > 0)
            
            manage_status = gr.Textbox(label="Status", interactive=False)
        
        # ----- EVENT HANDLERS -----
        
        # Handle table row selection
        def on_table_select(evt: gr.SelectData, table_data):
            if evt.index is not None and table_data is not None and len(table_data) > 0:
                row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if row_idx < len(table_data):
                    voice_name = table_data[row_idx][0]  # First column is name
                    return voice_name, gr.update(interactive=True), f"Selected: {voice_name}"
            return None, gr.update(interactive=False), ""
        
        voices_table.select(
            fn=on_table_select,
            inputs=[voices_table],
            outputs=[selected_voice, delete_btn, manage_status],
        )
        
        # Load transcript file content into textbox
        clone_transcript_file.change(
            fn=load_transcript_file,
            inputs=[clone_transcript_file],
            outputs=[clone_transcript],
        )
        
        # Clone button
        clone_btn.click(
            fn=clone_voice,
            inputs=[
                clone_audio, 
                clone_transcript, 
                clone_transcript_file,
                clone_voice_name, 
                clone_model, 
                clone_language
            ],
            outputs=[gen_voice, voices_table, clone_status, delete_btn, clear_all_btn, selected_voice],
        )
        
        # Generate button
        gen_btn.click(
            fn=generate_speech,
            inputs=[gen_voice, gen_text, gen_model, gen_language],
            outputs=[gen_output],
        )
        
        # Refresh button
        refresh_btn.click(
            fn=refresh_voices,
            inputs=[],
            outputs=[gen_voice, voices_table, delete_btn, clear_all_btn, selected_voice],
        )
        
        # Delete button
        delete_btn.click(
            fn=delete_voice,
            inputs=[selected_voice],
            outputs=[gen_voice, voices_table, manage_status, delete_btn, clear_all_btn, selected_voice],
        )
        
        # Clear all button
        clear_all_btn.click(
            fn=clear_all_voices,
            inputs=[],
            outputs=[gen_voice, voices_table, manage_status, delete_btn, clear_all_btn, selected_voice],
        )
        
        # ----- FOOTER -----
        gr.Markdown("---")
        gr.Markdown(
            "Built by [Saurabh Datta](https://saurabhdatta.com) with Claude | "
            "Powered by [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts)",
            elem_classes="footer"
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Gradio interface for Qwen3-TTS Voice Cloning")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
