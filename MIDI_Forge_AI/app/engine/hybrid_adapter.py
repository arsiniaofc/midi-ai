"""
MIDI Forge AI - Hybrid Engine Adapter
Integrates REMI (aiMIDI) and Serialized BPE (Theremin MIDI) representations
into a cohesive, unified music modeling interface.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from ..midi.parser import NoteEvent
from .remi_tokenizer import RemiTokenizer
from .theremin_tokenizer import ThereminBpeTokenizer


class HybridMusicAdapter:
    """Unified engine adapter orchestrating REMI and Theremin BPE representations."""

    def __init__(self, mode: str = "remi", **kwargs):
        self.mode = mode.lower()
        if self.mode == "remi":
            self.tokenizer = RemiTokenizer(**kwargs)
        elif self.mode in ("theremin", "theremin_bpe"):
            self.tokenizer = ThereminBpeTokenizer(**kwargs)
        else:
            raise ValueError(f"Modo desconhecido: {mode}. Opções: 'remi', 'theremin_bpe'")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_id(self) -> int:
        return self.tokenizer.vocab[self.tokenizer.pad_token]

    @property
    def bos_id(self) -> int:
        return self.tokenizer.vocab[self.tokenizer.bos_token]

    @property
    def eos_id(self) -> int:
        return self.tokenizer.vocab[self.tokenizer.eos_token]

    def encode_notes_to_tokens(self, notes: List[NoteEvent], ticks_per_beat: int = 480) -> List[str]:
        if self.mode == "remi":
            return self.tokenizer.encode(notes, ticks_per_beat=ticks_per_beat)
        else:
            return self.tokenizer.encode(notes)

    def decode_tokens_to_notes(self, tokens: List[str], ticks_per_beat: int = 480) -> List[NoteEvent]:
        if self.mode == "remi":
            return self.tokenizer.decode(tokens, ticks_per_beat=ticks_per_beat)
        else:
            return self.tokenizer.decode(tokens)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.tokens_to_ids(tokens)

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.ids_to_tokens(ids)

    def transpose_notes(self, notes: List[NoteEvent], semitones: int) -> List[NoteEvent]:
        """Transposes notes by semitones within valid MIDI range (21-108)."""
        transposed = []
        for n in notes:
            new_pitch = max(21, min(108, n.pitch + semitones))
            transposed.append(NoteEvent(
                pitch=new_pitch,
                start_tick=n.start_tick,
                end_tick=n.end_tick,
                duration_ticks=n.duration_ticks,
                velocity=n.velocity,
                channel=n.channel,
                program=n.program,
                track=n.track
            ))
        return transposed

    def export_vocab_metadata(self) -> Dict[str, Any]:
        """Exports vocabulary mapping and metadata for model checkpoint packaging."""
        return {
            "mode": self.mode,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "vocab": self.tokenizer.vocab,
        }

    @classmethod
    def from_vocab_metadata(cls, meta: Dict[str, Any]) -> "HybridMusicAdapter":
        mode = meta.get("mode", "remi")
        adapter = cls(mode=mode)
        # Restore vocabulary mapping
        if "vocab" in meta:
            adapter.tokenizer.vocab = {str(k): int(v) for k, v in meta["vocab"].items()}
            adapter.tokenizer.inv_vocab = {int(v): str(k) for k, v in meta["vocab"].items()}
        return adapter
