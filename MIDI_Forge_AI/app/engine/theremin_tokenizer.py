"""
MIDI Forge AI - Theremin Tokenizer Module (inspired by Theremin MIDI)
Implements serialized text event streams and Byte Pair Encoding (BPE) subtokenization:
- Note serialization: n:<pitch>:<dur_bin>:<vel_bin>:<wait_bin>
- BPE subtoken merging for common musical phrases / chord voicings
- Autoregressive prefix conditioning support
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
from ..midi.parser import NoteEvent


class ThereminBpeTokenizer:
    """Event serialization + Byte Pair Encoding (BPE) tokenizer inspired by Theremin MIDI."""

    def __init__(self, vocab_target_size: int = 512, time_step_ticks: int = 60):
        self.vocab_target_size = vocab_target_size
        self.time_step_ticks = time_step_ticks  # 60 ticks = 1/32 note at 480 PPQ

        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self._init_base_vocab()

    def _init_base_vocab(self):
        base_tokens = [self.pad_token, self.bos_token, self.eos_token]
        # Base pitch tokens (21 to 108)
        for p in range(21, 109):
            base_tokens.append(f"P{p}")
        # Base wait delta tokens (0 to 32 steps)
        for w in range(33):
            base_tokens.append(f"W{w}")
        # Base duration tokens (1 to 32 steps)
        for d in range(1, 33):
            base_tokens.append(f"D{d}")
        # Velocity bins (0 to 15)
        for v in range(16):
            base_tokens.append(f"V{v}")

        self.vocab = {t: i for i, t in enumerate(base_tokens)}
        self.inv_vocab = {i: t for i, t in enumerate(base_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def notes_to_event_symbols(self, notes: List[NoteEvent]) -> List[str]:
        """Converts note events to base event symbol stream."""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: (n.start_tick, n.pitch))
        symbols: List[str] = []

        last_start = 0
        for n in sorted_notes:
            # Wait delta from last start
            wait_ticks = max(0, n.start_tick - last_start)
            wait_steps = min(32, round(wait_ticks / self.time_step_ticks))
            last_start = n.start_tick

            # Pitch
            pitch = max(21, min(108, n.pitch))
            # Duration steps
            dur_steps = max(1, min(32, round(n.duration_ticks / self.time_step_ticks)))
            # Velocity bin (0-15)
            vel_bin = min(15, max(0, int(n.velocity / 128.0 * 16)))

            symbols.extend([f"W{wait_steps}", f"P{pitch}", f"D{dur_steps}", f"V{vel_bin}"])

        return symbols

    def train_bpe(self, list_of_symbol_sequences: List[List[str]], num_merges: int = 128):
        """Learns frequent byte pairs from a corpus of symbol sequences."""
        sequences = [list(seq) for seq in list_of_symbol_sequences]
        self.merges = []

        for _ in range(num_merges):
            pair_counts: Counter = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            self.merges.append(best_pair)
            merged_token = f"{best_pair[0]}+{best_pair[1]}"
            if merged_token not in self.vocab:
                new_idx = len(self.vocab)
                self.vocab[merged_token] = new_idx
                self.inv_vocab[new_idx] = merged_token

            # Apply merge across sequences
            new_sequences = []
            for seq in sequences:
                new_seq = []
                skip = False
                for i in range(len(seq)):
                    if skip:
                        skip = False
                        continue
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(merged_token)
                        skip = True
                    else:
                        new_seq.append(seq[i])
                new_sequences.append(new_seq)
            sequences = new_sequences

    def encode(self, notes: List[NoteEvent]) -> List[str]:
        """Encodes notes into BPE tokens with BOS/EOS."""
        symbols = self.notes_to_event_symbols(notes)
        if not symbols:
            return [self.bos_token, self.eos_token]

        # Apply learned merges in order
        tokens = list(symbols)
        for pair in self.merges:
            merged_tok = f"{pair[0]}+{pair[1]}"
            new_tokens = []
            skip = False
            for i in range(len(tokens)):
                if skip:
                    skip = False
                    continue
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged_tok)
                    skip = True
                else:
                    new_tokens.append(tokens[i])
            tokens = new_tokens

        return [self.bos_token] + tokens + [self.eos_token]

    def _unmerge_tokens(self, tokens: List[str]) -> List[str]:
        """Expands merged BPE tokens back to elemental symbols."""
        elemental: List[str] = []
        for tok in tokens:
            if tok in (self.bos_token, self.eos_token, self.pad_token):
                continue
            parts = tok.split("+")
            elemental.extend(parts)
        return elemental

    def decode(self, tokens: List[str]) -> List[NoteEvent]:
        """Decodes Theremin BPE tokens back to NoteEvent objects."""
        elemental = self._unmerge_tokens(tokens)
        notes: List[NoteEvent] = []

        current_tick = 0
        i = 0
        n_len = len(elemental)

        while i < n_len:
            sym = elemental[i]
            if sym.startswith("W"):
                try:
                    w_steps = int(sym[1:])
                except Exception:
                    w_steps = 0
                current_tick += w_steps * self.time_step_ticks
                i += 1
                # Expect P, D, V
                pitch = 60
                dur_steps = 4
                vel_bin = 10

                if i < n_len and elemental[i].startswith("P"):
                    try:
                        pitch = int(elemental[i][1:])
                    except Exception:
                        pitch = 60
                    i += 1
                if i < n_len and elemental[i].startswith("D"):
                    try:
                        dur_steps = int(elemental[i][1:])
                    except Exception:
                        dur_steps = 4
                    i += 1
                if i < n_len and elemental[i].startswith("V"):
                    try:
                        vel_bin = int(elemental[i][1:])
                    except Exception:
                        vel_bin = 10
                    i += 1

                vel = max(1, min(127, int((vel_bin + 0.5) / 16.0 * 128)))
                dur_ticks = max(1, dur_steps * self.time_step_ticks)
                notes.append(NoteEvent(
                    pitch=pitch,
                    start_tick=current_tick,
                    end_tick=current_tick + dur_ticks,
                    duration_ticks=dur_ticks,
                    velocity=vel,
                    channel=0,
                    program=0
                ))
            else:
                i += 1

        return notes

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(t, self.vocab[self.pad_token]) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(i, self.pad_token) for i in ids]
