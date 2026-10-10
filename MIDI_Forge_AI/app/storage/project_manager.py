"""
MIDI Forge AI - Storage and Project Manager Module
Manages datasets, models, checkpoints, generation history, and disk space.
"""

import os
import json
import time
import shutil
from typing import List, Dict, Any, Optional
from ..midi.parser import MidiParser, MidiFileInfo


class ProjectManager:
    """Central repository for applet state, file indexing, and disk persistence."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.datasets_dir = os.path.join(self.root_dir, "datasets")
        self.models_dir = os.path.join(self.root_dir, "models")
        self.checkpoints_dir = os.path.join(self.root_dir, "checkpoints")
        self.output_dir = os.path.join(self.root_dir, "output")
        self.logs_dir = os.path.join(self.root_dir, "logs")

        for d in [self.datasets_dir, self.models_dir, self.checkpoints_dir, self.output_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

        self.history_file = os.path.join(self.root_dir, "generation_history.json")
        self.settings_file = os.path.join(self.root_dir, "settings.json")
        self.parser = MidiParser()

    def get_stats(self) -> Dict[str, Any]:
        """Calculates global dashboard statistics."""
        def dir_size(path: str) -> int:
            total = 0
            for root, _, files in os.walk(path):
                for f in files:
                    total += os.path.getsize(os.path.join(root, f))
            return total

        # Count MIDIs in datasets and output
        mid_files = []
        for root, _, files in os.walk(self.datasets_dir):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    mid_files.append(os.path.join(root, f))

        models_count = len([f for f in os.listdir(self.models_dir) if f.endswith(".json") or f.endswith(".pt")])
        checkpoints_count = len([f for f in os.listdir(self.checkpoints_dir) if f.endswith(".pt")])
        output_count = len([f for f in os.listdir(self.output_dir) if f.endswith(".mid")])

        total_bytes = dir_size(self.root_dir)

        return {
            "imported_midi_count": len(mid_files),
            "models_count": models_count,
            "checkpoints_count": checkpoints_count,
            "output_midi_count": output_count,
            "disk_usage_mb": round(total_bytes / (1024 * 1024), 2),
        }

    def list_library_files(self) -> List[Dict[str, Any]]:
        """Scans dataset directory and returns file info for each MIDI."""
        results = []
        for root, _, files in os.walk(self.datasets_dir):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    fp = os.path.join(root, f)
                    info, notes = self.parser.parse_file(fp)
                    d = info.to_dict()
                    d["rel_path"] = os.path.relpath(fp, self.datasets_dir)
                    results.append(d)
        return results

    def add_generation_record(self, record: Dict[str, Any]):
        """Appends generation record to history."""
        records = self.get_generation_history()
        records.insert(0, record)
        # Keep latest 100 records
        records = records[:100]
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

    def get_generation_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
