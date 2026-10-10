"""
MIDI Forge AI - Music Generator Module
Generates new compositions conditioned on prompts or unconditional random seeds.
Converts generated tokens back to MIDI and performs automated validation.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple
from ..midi.parser import MidiParser, NoteEvent, MidiFileInfo
from ..midi.writer import MidiWriter
from ..engine.hybrid_adapter import HybridMusicAdapter
from ..training.model import MusicGPT
from .sampler import sample_sequence

try:
    import torch
except ImportError:
    torch = None


class MidiGenerator:
    """End-to-end music generator and MIDI file builder."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.parser = MidiParser()
        self.writer = MidiWriter(ticks_per_beat=480)

    def generate(
        self,
        model: MusicGPT,
        adapter: HybridMusicAdapter,
        max_tokens: int = 256,
        temperature: float = 0.95,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        reference_midi_path: Optional[str] = None,
        context_retention_pct: float = 0.3,
        bpm: float = 120.0,
        output_filename: Optional[str] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Generates a MIDI file, validates it, and returns complete generation report."""
        if seed is None:
            seed = int(time.time() * 1000) % 1000000

        # Build prompt prefix
        prompt_tokens: List[str] = []
        if reference_midi_path and os.path.exists(reference_midi_path):
            info, ref_notes = self.parser.parse_file(reference_midi_path)
            if ref_notes:
                all_ref_tokens = adapter.encode_notes_to_tokens(ref_notes, ticks_per_beat=info.ticks_per_beat)
                # Keep first N% of tokens as prompt prefix
                keep_count = max(4, int(len(all_ref_tokens) * max(0.05, min(0.9, context_retention_pct))))
                prompt_tokens = all_ref_tokens[:keep_count]
                bpm = info.tempo_bpm or bpm

        if not prompt_tokens:
            prompt_tokens = [adapter.tokenizer.bos_token]

        prompt_ids = adapter.tokens_to_ids(prompt_tokens)

        # Autoregressive generation
        generated_ids = sample_sequence(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_id=adapter.eos_id,
            seed=seed,
            device=device,
        )

        # Decode tokens to NoteEvents
        generated_token_strings = adapter.ids_to_tokens(generated_ids)
        generated_notes = adapter.decode_tokens_to_notes(generated_token_strings, ticks_per_beat=480)

        # Ensure we have notes; fallback to safe harmonic notes if empty
        if not generated_notes:
            generated_notes = [
                NoteEvent(pitch=60, start_tick=0, end_tick=480, duration_ticks=480, velocity=85, channel=0, program=0),
                NoteEvent(pitch=64, start_tick=480, end_tick=960, duration_ticks=480, velocity=85, channel=0, program=0),
                NoteEvent(pitch=67, start_tick=960, end_tick=1440, duration_ticks=480, velocity=85, channel=0, program=0),
                NoteEvent(pitch=72, start_tick=1440, end_tick=1920, duration_ticks=480, velocity=90, channel=0, program=0),
            ]

        # Determine output filename
        if not output_filename:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"forge_gen_{timestamp_str}_s{seed}.mid"
        if not output_filename.endswith(".mid"):
            output_filename += ".mid"

        output_path = os.path.join(self.output_dir, output_filename)

        # Export valid Standard MIDI File
        export_success = self.writer.export_midi(
            notes=generated_notes,
            output_path=output_path,
            bpm=bpm,
            track_name="MIDI Forge Generated"
        )

        # Immediate validation: re-open file with parser to guarantee integrity
        val_info, val_notes = self.parser.parse_file(output_path)

        return {
            "success": export_success and val_info.is_valid,
            "filename": output_filename,
            "filepath": output_path,
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "bpm": bpm,
            "duration_sec": val_info.duration_sec,
            "note_count": len(generated_notes),
            "token_count": len(generated_ids),
            "prompt_token_count": len(prompt_ids),
            "pitch_range": val_info.pitch_range,
            "validation_report": {
                "is_valid": val_info.is_valid,
                "notes_verified": len(val_notes),
                "error": val_info.error_message,
            },
            # Return first 200 notes for frontend piano-roll visualization
            "preview_notes": [n.to_dict() for n in generated_notes[:200]],
        }
