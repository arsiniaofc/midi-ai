"""
MIDI Forge AI - REMI Tokenizer Module (inspired by aiMIDI)
Implements Revamped MIDI (REMI) event tokenization:
- Bar and Beat/Position tokens (16 steps per bar)
- Pitch tokens (21 to 108, standard piano range)
- Quantized Velocity bins (32 bins)
- Quantized Duration tokens (1 to 64 steps)
- Program/Instrument tokens
- Special tokens: <PAD>, <BOS>, <EOS>
"""

from typing import List, Dict, Optional, Tuple
from ..midi.parser import NoteEvent


class RemiTokenizer:
    """REMI Tokenizer for Transformer/GPT music modeling."""

    def __init__(
        self,
        steps_per_bar: int = 16,
        velocity_bins: int = 32,
        max_duration_steps: int = 64,
        min_pitch: int = 21,
        max_pitch: int = 108,
    ):
        self.steps_per_bar = steps_per_bar
        self.velocity_bins = velocity_bins
        self.max_duration_steps = max_duration_steps
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        """Constructs fixed REMI vocabulary."""
        tokens = [self.pad_token, self.bos_token, self.eos_token, "Bar"]

        # Positions within a bar
        for p in range(self.steps_per_bar):
            tokens.append(f"Position_{p}")

        # Pitches
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            tokens.append(f"Pitch_{pitch}")

        # Velocity bins
        for v in range(self.velocity_bins):
            tokens.append(f"Velocity_{v}")

        # Durations in quant-steps
        for d in range(1, self.max_duration_steps + 1):
            tokens.append(f"Duration_{d}")

        # Instruments / Programs (0-127)
        for prog in range(128):
            tokens.append(f"Program_{prog}")

        self.vocab = {t: idx for idx, t in enumerate(tokens)}
        self.inv_vocab = {idx: t for idx, t in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def quantize_velocity(self, vel: int) -> int:
        # map 1-127 to 0..(velocity_bins - 1)
        return min(self.velocity_bins - 1, max(0, int(vel / 128.0 * self.velocity_bins)))

    def dequantize_velocity(self, bin_idx: int) -> int:
        return max(1, min(127, int((bin_idx + 0.5) / self.velocity_bins * 128.0)))

    def encode(self, notes: List[NoteEvent], ticks_per_beat: int = 480) -> List[str]:
        """Encodes note events into sequence of REMI token strings."""
        if not notes:
            return [self.bos_token, self.eos_token]

        ticks_per_step = max(1, ticks_per_beat // (self.steps_per_bar // 4))
        ticks_per_bar = ticks_per_step * self.steps_per_bar

        # Sort notes by start_tick, pitch
        sorted_notes = sorted(notes, key=lambda n: (n.start_tick, n.pitch))

        tokens: List[str] = [self.bos_token]
        current_bar = -1
        current_pos = -1

        for n in sorted_notes:
            # Bound pitch
            pitch = max(self.min_pitch, min(self.max_pitch, n.pitch))
            bar_idx = n.start_tick // ticks_per_bar
            tick_in_bar = n.start_tick % ticks_per_bar
            pos_idx = min(self.steps_per_bar - 1, tick_in_bar // ticks_per_step)

            # Advance bars
            if bar_idx > current_bar:
                bars_to_add = bar_idx - current_bar
                # Cap excessive silent bars to 4
                for _ in range(min(bars_to_add, 4)):
                    tokens.append("Bar")
                current_bar = bar_idx
                current_pos = -1

            # Advance position
            if pos_idx != current_pos:
                tokens.append(f"Position_{pos_idx}")
                current_pos = pos_idx

            # Quantize duration
            dur_steps = max(1, min(self.max_duration_steps, round(n.duration_ticks / ticks_per_step)))
            vel_bin = self.quantize_velocity(n.velocity)

            tokens.append(f"Pitch_{pitch}")
            tokens.append(f"Velocity_{vel_bin}")
            tokens.append(f"Duration_{dur_steps}")

        tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: List[str], ticks_per_beat: int = 480) -> List[NoteEvent]:
        """Decodes REMI token strings back into NoteEvent objects."""
        ticks_per_step = max(1, ticks_per_beat // (self.steps_per_bar // 4))
        ticks_per_bar = ticks_per_step * self.steps_per_bar

        notes: List[NoteEvent] = []
        current_bar = 0
        current_pos = 0

        pending_pitch: Optional[int] = None
        pending_vel: Optional[int] = 80

        for t in tokens:
            if t == self.bos_token:
                current_bar = 0
                current_pos = 0
                continue
            if t == self.eos_token:
                break
            if t == "Bar":
                current_bar += 1
                current_pos = 0
                continue
            if t.startswith("Position_"):
                try:
                    current_pos = int(t.split("_")[1])
                except Exception:
                    current_pos = 0
                continue
            if t.startswith("Pitch_"):
                try:
                    pending_pitch = int(t.split("_")[1])
                except Exception:
                    pending_pitch = 60
                continue
            if t.startswith("Velocity_"):
                try:
                    bin_idx = int(t.split("_")[1])
                    pending_vel = self.dequantize_velocity(bin_idx)
                except Exception:
                    pending_vel = 80
                continue
            if t.startswith("Duration_"):
                if pending_pitch is not None:
                    try:
                        dur_steps = int(t.split("_")[1])
                    except Exception:
                        dur_steps = 1

                    start_tick = (current_bar * ticks_per_bar) + (current_pos * ticks_per_step)
                    dur_ticks = max(1, dur_steps * ticks_per_step)
                    end_tick = start_tick + dur_ticks

                    notes.append(NoteEvent(
                        pitch=pending_pitch,
                        start_tick=start_tick,
                        end_tick=end_tick,
                        duration_ticks=dur_ticks,
                        velocity=pending_vel or 80,
                        channel=0,
                        program=0
                    ))
                    pending_pitch = None

        return notes

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(t, self.vocab[self.pad_token]) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(i, self.pad_token) for i in ids]
