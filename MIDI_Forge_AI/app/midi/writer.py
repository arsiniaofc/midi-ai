"""
MIDI Forge AI - MIDI Writer Module
Converts sequences of NoteEvent objects into valid Standard MIDI Files (.mid).
Optimized for compatibility with DAWs including FL Studio, Ableton Live, Reaper, and Logic Pro.
"""

import os
from typing import List, Optional
from .parser import NoteEvent

try:
    import mido
except ImportError:
    mido = None


class MidiWriter:
    """Converts NoteEvent sequences into clean, DAW-compatible Standard MIDI Files."""

    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = ticks_per_beat

    def export_midi(
        self,
        notes: List[NoteEvent],
        output_path: str,
        bpm: float = 120.0,
        time_signature: str = "4/4",
        track_name: str = "MIDI Forge AI",
        default_program: int = 0,
        format_type: int = 0
    ) -> bool:
        """Exports notes to a Standard MIDI File (.mid)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if mido is not None:
            return self._export_with_mido(
                notes=notes,
                output_path=output_path,
                bpm=bpm,
                time_signature=time_signature,
                track_name=track_name,
                default_program=default_program,
                format_type=format_type
            )
        else:
            return self._export_binary_fallback(
                notes=notes,
                output_path=output_path,
                bpm=bpm,
                ticks_per_beat=self.ticks_per_beat
            )

    def _export_with_mido(
        self,
        notes: List[NoteEvent],
        output_path: str,
        bpm: float,
        time_signature: str,
        track_name: str,
        default_program: int,
        format_type: int
    ) -> bool:
        mid = mido.MidiFile(type=format_type, ticks_per_beat=self.ticks_per_beat)

        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Meta events at tick 0
        tempo_us = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

        # Time signature
        try:
            num, den = (int(x) for x in time_signature.split("/"))
        except Exception:
            num, den = 4, 4
        track.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))

        # Program change
        track.append(mido.Message("program_change", program=max(0, min(127, default_program)), time=0))

        # Build absolute tick event list: (tick, event_type, pitch, velocity, channel)
        # Note-Off must precede Note-On at the same tick to prevent note choking
        events = []
        for n in notes:
            # Note On: order priority 1
            events.append((n.start_tick, 1, n.pitch, max(1, min(127, n.velocity)), n.channel, n.program))
            # Note Off: order priority 0
            end_t = max(n.start_tick + 1, n.end_tick)
            events.append((end_t, 0, n.pitch, 0, n.channel, n.program))

        # Sort primarily by tick, then note_off (0) before note_on (1)
        events.sort(key=lambda e: (e[0], e[1]))

        last_tick = 0
        for tick, ev_type, pitch, vel, channel, prog in events:
            delta = max(0, tick - last_tick)
            last_tick = tick

            if ev_type == 1:
                track.append(mido.Message("note_on", note=pitch, velocity=vel, channel=channel, time=delta))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0, channel=channel, time=delta))

        track.append(mido.MetaMessage("end_of_track", time=self.ticks_per_beat // 2))

        mid.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    def _export_binary_fallback(
        self,
        notes: List[NoteEvent],
        output_path: str,
        bpm: float,
        ticks_per_beat: int
    ) -> bool:
        """Pure-Python binary SMF generator fallback (zero dependencies)."""
        import struct

        def encode_varlen(value: int) -> bytes:
            buf = bytearray()
            buf.append(value & 0x7F)
            value >>= 7
            while value > 0:
                buf.insert(0, (value & 0x7F) | 0x80)
                value >>= 7
            return bytes(buf)

        tempo_us = int(60000000 / max(1.0, bpm))
        events = []
        for n in notes:
            events.append((n.start_tick, 1, n.pitch, max(1, min(127, n.velocity)), n.channel))
            events.append((max(n.start_tick + 1, n.end_tick), 0, n.pitch, 0, n.channel))
        events.sort(key=lambda e: (e[0], e[1]))

        trk_data = bytearray()
        # Tempo meta: FF 51 03 [tempo_us as 3 bytes]
        trk_data.extend(encode_varlen(0))
        trk_data.extend(b"\xFF\x51\x03" + struct.pack(">I", tempo_us)[1:])

        last_tick = 0
        for tick, ev_type, pitch, vel, channel in events:
            delta = max(0, tick - last_tick)
            last_tick = tick
            trk_data.extend(encode_varlen(delta))
            status = (0x90 if ev_type == 1 else 0x80) | (channel & 0x0F)
            trk_data.extend(bytes([status, pitch & 0x7F, vel & 0x7F]))

        # End of track
        trk_data.extend(encode_varlen(ticks_per_beat // 2))
        trk_data.extend(b"\xFF\x2F\x00")

        # Write MThd + MTrk
        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
        track_chunk = b"MTrk" + struct.pack(">I", len(trk_data)) + bytes(trk_data)

        with open(output_path, "wb") as f:
            f.write(header + track_chunk)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
