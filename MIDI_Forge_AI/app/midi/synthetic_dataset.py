"""
MIDI Forge AI - Synthetic Dataset Generator
Generates clean, diverse, royalty-free MIDI compositions for local testing and training.
Covers polyphony, arpeggios, chords, basslines, and varied tempos.
"""

import os
from typing import List
from .parser import NoteEvent
from .writer import MidiWriter


def generate_synthetic_dataset(output_dir: str) -> List[str]:
    """Generates 8 varied MIDI pieces into output_dir and returns their filepaths."""
    os.makedirs(output_dir, exist_ok=True)
    writer = MidiWriter(ticks_per_beat=480)
    created_files = []

    # 1. Classical Bach-style Polyphonic Invention (C major / A minor, 100 BPM)
    # Ticks per 16th note = 120
    notes_classical: List[NoteEvent] = []
    # Measure 1-4 motif in C Major
    c_maj_scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]
    t = 0
    # Voice 1 (Right hand)
    for _ in range(2):
        for pitch in c_maj_scale:
            notes_classical.append(NoteEvent(pitch=pitch, start_tick=t, end_tick=t + 110, duration_ticks=110, velocity=85, channel=0, program=0))
            t += 120
    # Voice 2 (Left hand counterpoint, delayed 480 ticks)
    t_bass = 480
    bass_pattern = [48, 55, 52, 55, 45, 52, 48, 52, 41, 48, 45, 48, 43, 50, 47, 50]
    for _ in range(2):
        for pitch in bass_pattern:
            notes_classical.append(NoteEvent(pitch=pitch, start_tick=t_bass, end_tick=t_bass + 220, duration_ticks=220, velocity=80, channel=0, program=0))
            t_bass += 240
    p1 = os.path.join(output_dir, "01_classical_invention_c_major.mid")
    writer.export_midi(notes_classical, p1, bpm=100.0, track_name="Bach Invention Study")
    created_files.append(p1)

    # 2. Synthwave Arpeggio & Bass (D minor, 120 BPM)
    notes_synthwave: List[NoteEvent] = []
    d_minor_chords = [
        [50, 53, 57, 62], # Dm
        [46, 50, 53, 58], # Bb
        [48, 52, 55, 60], # C
        [45, 48, 52, 57], # Am
    ]
    t = 0
    for chord in d_minor_chords * 2:
        # Bass note (pumping 8th notes)
        root = chord[0] - 12
        for b_i in range(8):
            notes_synthwave.append(NoteEvent(pitch=root, start_tick=t + b_i * 240, end_tick=t + b_i * 240 + 200, duration_ticks=200, velocity=95, channel=0, program=38))
        # Lead Arp (16th notes)
        arp_pitches = chord + [chord[1] + 12, chord[2] + 12, chord[1] + 12, chord[0] + 12]
        for a_i, pitch in enumerate(arp_pitches * 2):
            notes_synthwave.append(NoteEvent(pitch=pitch + 12, start_tick=t + a_i * 120, end_tick=t + a_i * 120 + 90, duration_ticks=90, velocity=88, channel=0, program=81))
        t += 1920 # 4 beats
    p2 = os.path.join(output_dir, "02_synthwave_d_minor_arp.mid")
    writer.export_midi(notes_synthwave, p2, bpm=120.0, track_name="Neon Drive")
    created_files.append(p2)

    # 3. Lo-Fi Chillhop Jazz Chords (F minor / Ab, 84 BPM)
    notes_lofi: List[NoteEvent] = []
    # Fm9 -> Bbm9 -> Eb13 -> Abmaj7
    lofi_chords = [
        [53, 60, 63, 67, 68], # Fm9
        [58, 65, 68, 72, 73], # Bbm9
        [51, 58, 61, 65, 67], # Eb13
        [56, 63, 67, 70, 72], # Abmaj7
    ]
    t = 0
    for chord in lofi_chords * 3:
        # Laid-back chord sustained
        for pitch in chord:
            notes_lofi.append(NoteEvent(pitch=pitch, start_tick=t + 30, end_tick=t + 1800, duration_ticks=1770, velocity=68, channel=0, program=4)) # Electric Piano
        t += 1920
    p3 = os.path.join(output_dir, "03_lofi_jazz_chords_fm.mid")
    writer.export_midi(notes_lofi, p3, bpm=84.0, track_name="Rainy Cafe Rhodes")
    created_files.append(p3)

    # 4. Blues Pentatonic Riff & Walking Bass (E minor, 105 BPM)
    notes_blues: List[NoteEvent] = []
    blues_riff = [64, 67, 69, 70, 69, 67, 64, 62, 64]
    t = 0
    for _ in range(4):
        for pitch in blues_riff:
            notes_blues.append(NoteEvent(pitch=pitch, start_tick=t, end_tick=t + 180, duration_ticks=180, velocity=90, channel=0, program=29)) # Overdriven Guitar
            t += 240
    p4 = os.path.join(output_dir, "04_blues_pentatonic_riff.mid")
    writer.export_midi(notes_blues, p4, bpm=105.0, track_name="Chicago Blues Riff")
    created_files.append(p4)

    # 5. Uplifting Trance Anthem (G major, 138 BPM)
    notes_trance: List[NoteEvent] = []
    trance_lead = [71, 74, 79, 78, 74, 76, 74, 71, 67, 71, 74, 76, 74, 71, 69, 71]
    t = 0
    for _ in range(4):
        for pitch in trance_lead:
            notes_trance.append(NoteEvent(pitch=pitch, start_tick=t, end_tick=t + 100, duration_ticks=100, velocity=100, channel=0, program=80))
            # Pad chord underneath
            if t % 1920 == 0:
                for chord_p in [55, 59, 62, 67]:
                    notes_trance.append(NoteEvent(pitch=chord_p, start_tick=t, end_tick=t + 1900, duration_ticks=1900, velocity=70, channel=0, program=89))
            t += 120
    p5 = os.path.join(output_dir, "05_uplifting_trance_melody.mid")
    writer.export_midi(notes_trance, p5, bpm=138.0, track_name="Aura Trance Lead")
    created_files.append(p5)

    # 6. Ambient Piano Study (C major, 64 BPM)
    notes_ambient: List[NoteEvent] = []
    amb_motifs = [[60, 67, 72, 76], [57, 64, 69, 72], [53, 60, 65, 69], [55, 62, 67, 71]]
    t = 0
    for block in amb_motifs * 2:
        for i, pitch in enumerate(block):
            notes_ambient.append(NoteEvent(pitch=pitch, start_tick=t + i * 480, end_tick=t + i * 480 + 900, duration_ticks=900, velocity=58, channel=0, program=0))
        t += 1920
    p6 = os.path.join(output_dir, "06_ambient_piano_study.mid")
    writer.export_midi(notes_ambient, p6, bpm=64.0, track_name="Nocturne Space")
    created_files.append(p6)

    # 7. Funk Bass & Clavinet Stabs (A Dorian, 112 BPM)
    notes_funk: List[NoteEvent] = []
    funk_bass = [45, 45, 48, 50, 52, 50, 48, 45, 55, 52, 50, 48]
    t = 0
    for _ in range(3):
        for pitch in funk_bass:
            notes_funk.append(NoteEvent(pitch=pitch, start_tick=t, end_tick=t + 100, duration_ticks=100, velocity=105, channel=0, program=33)) # Fingered Bass
            t += 160
    p7 = os.path.join(output_dir, "07_funk_dorian_groove.mid")
    writer.export_midi(notes_funk, p7, bpm=112.0, track_name="Funky Groove")
    created_files.append(p7)

    # 8. Cinematic String Progression (D minor, 72 BPM)
    notes_cinema: List[NoteEvent] = []
    cinematic_chords = [
        [50, 57, 62, 65], # Dm
        [46, 53, 58, 62], # Bb
        [45, 52, 57, 60], # A
        [48, 55, 60, 64], # C
    ]
    t = 0
    for chord in cinematic_chords * 2:
        for pitch in chord:
            notes_cinema.append(NoteEvent(pitch=pitch, start_tick=t, end_tick=t + 1920, duration_ticks=1920, velocity=75, channel=0, program=48)) # Strings Ensemble
        t += 1920
    p8 = os.path.join(output_dir, "08_cinematic_strings_d_minor.mid")
    writer.export_midi(notes_cinema, p8, bpm=72.0, track_name="Epic Overture")
    created_files.append(p8)

    return created_files
