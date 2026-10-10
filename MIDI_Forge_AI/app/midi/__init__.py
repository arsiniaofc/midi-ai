"""MIDI Forge AI - MIDI processing package."""
from .parser import MidiParser, NoteEvent, MidiFileInfo
from .writer import MidiWriter
from .synthetic_dataset import generate_synthetic_dataset

__all__ = ["MidiParser", "NoteEvent", "MidiFileInfo", "MidiWriter", "generate_synthetic_dataset"]
