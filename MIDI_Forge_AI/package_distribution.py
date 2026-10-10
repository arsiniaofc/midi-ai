"""
MIDI Forge AI - Windows Portable Packaging Script
Creates a clean, distribution-ready ZIP archive of the MIDI_Forge_AI folder.
Excludes .pyc, __pycache__, and temporary files.
"""

import os
import sys
import zipfile

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ZIP = os.path.join(os.path.dirname(PROJECT_DIR), "MIDI_Forge_AI_Windows_Portable.zip")

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".git",
    ".venv",
    ".pytest_cache",
    ".DS_Store",
]


def create_portable_zip(zip_path: str = OUTPUT_ZIP) -> str:
    print(f"Empacotando MIDI Forge AI para {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Prune excluded dirs
            dirs[:] = [d for d in dirs if not any(p in d for p in EXCLUDE_PATTERNS)]
            for file in files:
                if any(p in file for p in EXCLUDE_PATTERNS):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.join("MIDI_Forge_AI", os.path.relpath(file_path, PROJECT_DIR))
                zipf.write(file_path, arcname)
    print(f"[OK] Arquivo gerado com sucesso: {zip_path} ({os.path.getsize(zip_path)} bytes)")
    return zip_path


if __name__ == "__main__":
    create_portable_zip()
