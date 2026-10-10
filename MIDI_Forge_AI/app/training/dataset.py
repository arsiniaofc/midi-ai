"""
MIDI Forge AI - Dataset Preparation Module
Converts MIDI collections into tokenized sliding-window sequence datasets for training.
Supports augmentation by transposition, train/validation split, and cached serialization.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from ..midi.parser import MidiParser, NoteEvent, MidiFileInfo
from ..engine.hybrid_adapter import HybridMusicAdapter

try:
    import torch
except ImportError:
    torch = None


class MusicDataset:
    """Sliding-window sequence dataset with batched tensor generation."""

    def __init__(
        self,
        token_sequences: List[List[int]],
        block_size: int = 256,
        stride: Optional[int] = None,
        val_ratio: float = 0.1,
    ):
        self.block_size = block_size
        self.stride = stride or (block_size // 2)

        # Slice all token sequences into blocks of size (block_size + 1)
        all_samples: List[List[int]] = []
        for seq in token_sequences:
            if len(seq) < 4:
                continue
            # If sequence is shorter than block_size + 1, pad it
            if len(seq) <= block_size + 1:
                padded = seq + [0] * (block_size + 1 - len(seq))
                all_samples.append(padded)
            else:
                for i in range(0, len(seq) - block_size, self.stride):
                    chunk = seq[i : i + block_size + 1]
                    all_samples.append(chunk)

        # Shuffle and split into train / val
        random.seed(42)
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1.0 - val_ratio))
        self.train_samples = all_samples[:split_idx] if split_idx > 0 else all_samples
        self.val_samples = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[:1]

    def get_batch(self, split: str = "train", batch_size: int = 8, device: str = "cpu") -> Tuple[Any, Any]:
        """Extracts a batch of (x, y) tensors."""
        samples = self.train_samples if split == "train" else self.val_samples
        if not samples:
            samples = self.train_samples

        batch_items = random.sample(samples, min(batch_size, len(samples)))
        x_list = [item[:-1] for item in batch_items]
        y_list = [item[1:] for item in batch_items]

        if torch is not None:
            x_tensor = torch.tensor(x_list, dtype=torch.long, device=device)
            y_tensor = torch.tensor(y_list, dtype=torch.long, device=device)
            return x_tensor, y_tensor
        return x_list, y_list


class DatasetBuilder:
    """Orchestrates ingestion of MIDI files and creation of tokenized datasets."""

    def __init__(self, parser: Optional[MidiParser] = None):
        self.parser = parser or MidiParser()

    def prepare_dataset(
        self,
        midi_filepaths: List[str],
        adapter: HybridMusicAdapter,
        block_size: int = 256,
        augment_transposition: bool = False,
        transposition_range: Tuple[int, int] = (-2, 2),
    ) -> Dict[str, Any]:
        """Processes MIDI files, tokenizes them, and returns dataset stats and token IDs."""
        processed_files = []
        skipped_files = []
        all_token_ids: List[List[int]] = []
        total_notes = 0

        for path in midi_filepaths:
            info, notes = self.parser.parse_file(path)
            if not info.is_valid or not notes:
                skipped_files.append({"path": path, "reason": info.error_message or "Sem notas musicais"})
                continue

            processed_files.append(info.to_dict())
            total_notes += len(notes)

            # Base sequence
            tokens = adapter.encode_notes_to_tokens(notes, ticks_per_beat=info.ticks_per_beat)
            token_ids = adapter.tokens_to_ids(tokens)
            all_token_ids.append(token_ids)

            # Data augmentation: Transpose if requested
            if augment_transposition:
                for semi in range(transposition_range[0], transposition_range[1] + 1):
                    if semi == 0:
                        continue
                    trans_notes = adapter.transpose_notes(notes, semi)
                    t_tokens = adapter.encode_notes_to_tokens(trans_notes, ticks_per_beat=info.ticks_per_beat)
                    all_token_ids.append(adapter.tokens_to_ids(t_tokens))

        dataset = MusicDataset(all_token_ids, block_size=block_size)

        report = {
            "mode": adapter.mode,
            "vocab_size": adapter.vocab_size,
            "block_size": block_size,
            "total_files_analyzed": len(midi_filepaths),
            "files_processed_count": len(processed_files),
            "files_skipped_count": len(skipped_files),
            "skipped_details": skipped_files,
            "total_notes_analyzed": total_notes,
            "total_sequences_generated": len(dataset.train_samples) + len(dataset.val_samples),
            "train_sequences_count": len(dataset.train_samples),
            "val_sequences_count": len(dataset.val_samples),
            "estimated_memory_mb": round((len(all_token_ids) * block_size * 4) / (1024 * 1024), 2),
        }

        return {
            "report": report,
            "token_ids": all_token_ids,
            "dataset": dataset,
        }
