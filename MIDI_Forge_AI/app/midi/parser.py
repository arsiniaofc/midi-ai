"""
MIDI Forge AI - MIDI Parser Module
Extracts notes, tempos, time signatures, channels, and tracks from Standard MIDI Files (SMF Type 0 and 1).
Provides corruption detection, duplicate detection, and normalization.
"""

import os
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

try:
    import mido
except ImportError:
    mido = None


@dataclass
class NoteEvent:
    pitch: int            # 0-127 (MIDI pitch)
    start_tick: int       # Ticks from start of piece
    end_tick: int         # Ticks when note terminates
    duration_ticks: int   # Length in ticks
    velocity: int         # 1-127
    channel: int = 0      # 0-15 (channel 9 = drums in 0-indexed)
    program: int = 0      # Instrument program (0-127)
    track: int = 0        # Track index

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MidiFileInfo:
    filename: str
    filepath: str
    file_size_bytes: int
    duration_sec: float
    ticks_per_beat: int
    tempo_bpm: float
    time_signature: str
    track_count: int
    note_count: int
    pitch_range: Tuple[int, int]
    is_valid: bool
    error_message: Optional[str] = None
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pitch_range"] = list(self.pitch_range)
        return d


class MidiParser:
    """Robust MIDI reader supporting mido and binary SMF fallback."""

    def __init__(self, default_ticks_per_beat: int = 480):
        self.default_ticks_per_beat = default_ticks_per_beat

    def compute_content_hash(self, notes: List[NoteEvent]) -> str:
        """Computes a deterministic hash of notes for duplicate detection."""
        hasher = hashlib.sha256()
        for n in sorted(notes, key=lambda x: (x.start_tick, x.pitch, x.channel)):
            hasher.update(f"{n.pitch}:{n.start_tick}:{n.duration_ticks}:{n.velocity}:{n.channel};".encode("utf-8"))
        return hasher.hexdigest()

    def parse_file(self, filepath: str) -> Tuple[MidiFileInfo, List[NoteEvent]]:
        """Parses a MIDI file and returns its metadata and extracted NoteEvent list."""
        if not os.path.exists(filepath):
            info = MidiFileInfo(
                filename=os.path.basename(filepath),
                filepath=filepath,
                file_size_bytes=0,
                duration_sec=0.0,
                ticks_per_beat=self.default_ticks_per_beat,
                tempo_bpm=120.0,
                time_signature="4/4",
                track_count=0,
                note_count=0,
                pitch_range=(0, 0),
                is_valid=False,
                error_message="Arquivo não encontrado",
            )
            return info, []

        file_size = os.path.getsize(filepath)
        if file_size < 14:  # SMF header chunk is minimum 14 bytes
            info = MidiFileInfo(
                filename=os.path.basename(filepath),
                filepath=filepath,
                file_size_bytes=file_size,
                duration_sec=0.0,
                ticks_per_beat=self.default_ticks_per_beat,
                tempo_bpm=120.0,
                time_signature="4/4",
                track_count=0,
                note_count=0,
                pitch_range=(0, 0),
                is_valid=False,
                error_message="Arquivo corrompido: tamanho menor que o cabeçalho SMF mínimo (14 bytes)",
            )
            return info, []

        # Validate MThd header magic bytes
        with open(filepath, "rb") as f:
            header_bytes = f.read(4)
            if header_bytes != b"MThd":
                info = MidiFileInfo(
                    filename=os.path.basename(filepath),
                    filepath=filepath,
                    file_size_bytes=file_size,
                    duration_sec=0.0,
                    ticks_per_beat=self.default_ticks_per_beat,
                    tempo_bpm=120.0,
                    time_signature="4/4",
                    track_count=0,
                    note_count=0,
                    pitch_range=(0, 0),
                    is_valid=False,
                    error_message="Assinatura MIDI inválida: não inicia com o cabeçalho 'MThd'",
                )
                return info, []

        if mido is not None:
            try:
                return self._parse_with_mido(filepath)
            except Exception:
                pass
        return self._parse_binary_fallback(filepath)

    def _parse_binary_fallback(self, filepath: str) -> Tuple[MidiFileInfo, List[NoteEvent]]:
        """Pure-Python binary SMF parser without third-party dependencies."""
        import struct

        with open(filepath, "rb") as f:
            raw = f.read()

        file_size = len(raw)
        if len(raw) < 14 or raw[:4] != b"MThd":
            info = MidiFileInfo(
                filename=os.path.basename(filepath),
                filepath=filepath,
                file_size_bytes=file_size,
                duration_sec=0.0,
                ticks_per_beat=self.default_ticks_per_beat,
                tempo_bpm=120.0,
                time_signature="4/4",
                track_count=0,
                note_count=0,
                pitch_range=(0, 0),
                is_valid=False,
                error_message="Cabeçalho SMF inválido",
            )
            return info, []

        hdr_len, fmt, n_tracks, division = struct.unpack(">IHHH", raw[4:14])
        ticks_per_beat = division if division > 0 else self.default_ticks_per_beat

        pos = 8 + hdr_len
        tempo_us = 500000
        time_sig = "4/4"
        notes: List[NoteEvent] = []

        track_count = 0
        while pos + 8 <= len(raw):
            chunk_type = raw[pos:pos+4]
            chunk_len = struct.unpack(">I", raw[pos+4:pos+8])[0]
            pos += 8
            if chunk_type != b"MTrk":
                pos += chunk_len
                continue

            track_idx = track_count
            track_count += 1
            trk_end = min(len(raw), pos + chunk_len)
            trk_pos = pos
            current_tick = 0
            active_notes: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (channel, pitch) -> (start_tick, velocity)
            running_status = 0

            while trk_pos < trk_end:
                # Read varlen delta
                delta = 0
                while trk_pos < trk_end:
                    b = raw[trk_pos]
                    trk_pos += 1
                    delta = (delta << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                current_tick += delta

                if trk_pos >= trk_end:
                    break

                b = raw[trk_pos]
                if b & 0x80:
                    status = b
                    trk_pos += 1
                    running_status = status
                else:
                    status = running_status

                ev_type = status & 0xF0
                channel = status & 0x0F

                if status == 0xFF:  # Meta event
                    if trk_pos >= trk_end:
                        break
                    meta_type = raw[trk_pos]
                    trk_pos += 1
                    # varlen length
                    meta_len = 0
                    while trk_pos < trk_end:
                        mb = raw[trk_pos]
                        trk_pos += 1
                        meta_len = (meta_len << 7) | (mb & 0x7F)
                        if not (mb & 0x80):
                            break
                    meta_data = raw[trk_pos:trk_pos + meta_len]
                    trk_pos += meta_len

                    if meta_type == 0x51 and len(meta_data) >= 3:
                        tempo_us = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                    elif meta_type == 0x58 and len(meta_data) >= 2:
                        time_sig = f"{meta_data[0]}/{2**meta_data[1]}"
                elif status in (0xF0, 0xF7):  # SysEx
                    syx_len = 0
                    while trk_pos < trk_end:
                        sb = raw[trk_pos]
                        trk_pos += 1
                        syx_len = (syx_len << 7) | (sb & 0x7F)
                        if not (sb & 0x80):
                            break
                    trk_pos += syx_len
                elif ev_type in (0x80, 0x90):  # Note On / Off
                    if trk_pos + 1 >= trk_end:
                        break
                    pitch = raw[trk_pos]
                    vel = raw[trk_pos + 1]
                    trk_pos += 2
                    key = (channel, pitch)

                    if ev_type == 0x90 and vel > 0:
                        if key in active_notes:
                            prev_start, prev_vel = active_notes.pop(key)
                            dur = max(1, current_tick - prev_start)
                            notes.append(NoteEvent(pitch=pitch, start_tick=prev_start, end_tick=current_tick, duration_ticks=dur, velocity=prev_vel, channel=channel, track=track_idx))
                        active_notes[key] = (current_tick, vel)
                    else:  # Note off
                        if key in active_notes:
                            prev_start, prev_vel = active_notes.pop(key)
                            dur = max(1, current_tick - prev_start)
                            notes.append(NoteEvent(pitch=pitch, start_tick=prev_start, end_tick=current_tick, duration_ticks=dur, velocity=prev_vel, channel=channel, track=track_idx))
                elif ev_type in (0xA0, 0xB0, 0xE0):
                    trk_pos += 2
                elif ev_type in (0xC0, 0xD0):
                    trk_pos += 1
                else:
                    trk_pos += 1

            for (ch, pitch), (prev_start, prev_vel) in list(active_notes.items()):
                dur = max(ticks_per_beat // 2, current_tick - prev_start)
                notes.append(NoteEvent(pitch=pitch, start_tick=prev_start, end_tick=prev_start + dur, duration_ticks=dur, velocity=prev_vel, channel=ch, track=track_idx))

            pos = trk_end

        notes.sort(key=lambda n: (n.start_tick, n.pitch))
        tempo_bpm = round(60000000.0 / max(1, tempo_us), 2)
        max_tick = max((n.end_tick for n in notes), default=0)
        duration_sec = round((max_tick / max(1, ticks_per_beat)) * (tempo_us / 1000000.0), 2)
        min_p = min((n.pitch for n in notes), default=0)
        max_p = max((n.pitch for n in notes), default=0)

        info = MidiFileInfo(
            filename=os.path.basename(filepath),
            filepath=filepath,
            file_size_bytes=file_size,
            duration_sec=duration_sec,
            ticks_per_beat=ticks_per_beat,
            tempo_bpm=tempo_bpm,
            time_signature=time_sig,
            track_count=max(1, track_count),
            note_count=len(notes),
            pitch_range=(min_p, max_p),
            is_valid=True,
            content_hash=self.compute_content_hash(notes),
        )
        return info, notes

    def _parse_with_mido(self, filepath: str) -> Tuple[MidiFileInfo, List[NoteEvent]]:
        mid = mido.MidiFile(filepath, clip=True)
        ticks_per_beat = mid.ticks_per_beat or self.default_ticks_per_beat

        tempo_us = 500000  # Default 120 BPM (500,000 microseconds per beat)
        time_sig = "4/4"
        notes: List[NoteEvent] = []

        # Track active note-ons per (track_idx, channel, pitch)
        # Value is (start_tick, velocity, program)
        active_notes: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        channel_programs: Dict[Tuple[int, int], int] = {}  # (track, channel) -> program

        for track_idx, track in enumerate(mid.tracks):
            current_tick = 0
            for msg in track:
                current_tick += msg.time

                if msg.is_meta:
                    if msg.type == "set_tempo":
                        tempo_us = msg.tempo
                    elif msg.type == "time_signature":
                        time_sig = f"{msg.numerator}/{msg.denominator}"
                else:
                    channel = getattr(msg, "channel", 0)
                    if msg.type == "program_change":
                        channel_programs[(track_idx, channel)] = msg.program
                    elif msg.type == "note_on" and msg.velocity > 0:
                        prog = channel_programs.get((track_idx, channel), 0)
                        key = (track_idx, channel, msg.note)
                        # If already active, close previous note first
                        if key in active_notes:
                            prev_start, prev_vel, prev_prog = active_notes.pop(key)
                            dur = max(1, current_tick - prev_start)
                            notes.append(NoteEvent(
                                pitch=msg.note,
                                start_tick=prev_start,
                                end_tick=current_tick,
                                duration_ticks=dur,
                                velocity=prev_vel,
                                channel=channel,
                                program=prev_prog,
                                track=track_idx
                            ))
                        active_notes[key] = (current_tick, msg.velocity, prog)
                    elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
                        key = (track_idx, channel, msg.note)
                        if key in active_notes:
                            prev_start, prev_vel, prev_prog = active_notes.pop(key)
                            dur = max(1, current_tick - prev_start)
                            notes.append(NoteEvent(
                                pitch=msg.note,
                                start_tick=prev_start,
                                end_tick=current_tick,
                                duration_ticks=dur,
                                velocity=prev_vel,
                                channel=channel,
                                program=prev_prog,
                                track=track_idx
                            ))

            # Close any dangling note-ons at end of track
            for (t_idx, ch, pitch), (prev_start, prev_vel, prev_prog) in list(active_notes.items()):
                if t_idx == track_idx:
                    dur = max(ticks_per_beat // 2, current_tick - prev_start)
                    notes.append(NoteEvent(
                        pitch=pitch,
                        start_tick=prev_start,
                        end_tick=prev_start + dur,
                        duration_ticks=dur,
                        velocity=prev_vel,
                        channel=ch,
                        program=prev_prog,
                        track=track_idx
                    ))
                    active_notes.pop((t_idx, ch, pitch))

        # Sort notes by start_tick then pitch
        notes.sort(key=lambda n: (n.start_tick, n.pitch))

        tempo_bpm = round(60000000.0 / tempo_us, 2) if tempo_us > 0 else 120.0
        duration_sec = mid.length if hasattr(mid, "length") and mid.length > 0 else 0.0

        if duration_sec <= 0.0 and notes:
            max_tick = max(n.end_tick for n in notes)
            duration_sec = round((max_tick / ticks_per_beat) * (tempo_us / 1000000.0), 2)

        min_pitch = min((n.pitch for n in notes), default=0)
        max_pitch = max((n.pitch for n in notes), default=0)

        c_hash = self.compute_content_hash(notes)

        info = MidiFileInfo(
            filename=os.path.basename(filepath),
            filepath=filepath,
            file_size_bytes=os.path.getsize(filepath),
            duration_sec=round(duration_sec, 2),
            ticks_per_beat=ticks_per_beat,
            tempo_bpm=tempo_bpm,
            time_signature=time_sig,
            track_count=len(mid.tracks),
            note_count=len(notes),
            pitch_range=(min_pitch, max_pitch),
            is_valid=True,
            content_hash=c_hash,
        )

        return info, notes
