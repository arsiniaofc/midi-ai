"""MIDI Forge AI - Training package."""
from .model import MusicGPT, GPTConfig
from .dataset import MusicDataset, DatasetBuilder
from .trainer import LocalMusicTrainer, TrainProgress

__all__ = ["MusicGPT", "GPTConfig", "MusicDataset", "DatasetBuilder", "LocalMusicTrainer", "TrainProgress"]
