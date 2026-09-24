"""
generate_midi_data.py
Generates a rich set of sample MIDI files (classical & jazz styles)
using only Python stdlib + our midi_utils module.

Run:  python3 generate_midi_data.py
Output: ../data/midi/  (creates ~30 MIDI files)
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
from midi_utils import write_midi

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "midi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TPB = 480  # ticks per beat
TEMPO = 500000  # 120 BPM

# ── Music theory helpers ─────────────────────────────────────────────────────

SCALES = {
    "major":       [0,2,4,5,7,9,11],
    "minor":       [0,2,3,5,7,8,10],
    "dorian":      [0,2,3,5,7,9,10],
    "mixolydian":  [0,2,4,5,7,9,10],
    "pentatonic":  [0,2,4,7,9],
    "blues":       [0,3,5,6,7,10],
    "whole_tone":  [0,2,4,6,8,10],
    "diminished":  [0,2,3,5,6,8,9,11],
}

# Common chord types: (intervals from root)
CHORDS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus4": [0, 5, 7],
}

# Jazz ii-V-I progressions (roots relative to key, in semitones)
JAZZ_251 = [2, 7, 0]   # ii, V, I
JAZZ_BLUES = [0,0,0,0, 5,5,0,0, 7,5,0,7]  # 12-bar blues roots


def scale_pitches(root: int, scale_name: str, octaves: int = 2) -> list:
    """Return MIDI pitches for a scale spanning `octaves`."""
    intervals = SCALES[scale_name]
    pitches = []
    for oct in range(octaves):
        for i in intervals:
            pitches.append(root + oct * 12 + i)
    return pitches


def chord_pitches(root: int, chord_type: str, octave: int = 4) -> list:
    base = 12 * octave + root
    return [base + i for i in CHORDS[chord_type]]


def make_note(pitch, vel, start, dur):
    return {"pitch": pitch, "velocity": vel,
            "start_tick": start, "end_tick": start + dur}


# ── Pattern generators ───────────────────────────────────────────────────────

def alberti_bass(root, beats, start_tick, vel=55):
    """Alberti bass pattern: bottom-top-middle-top"""
    chord = chord_pitches(root, "maj", octave=3)
    if len(chord) < 3:
        chord = chord + chord
    pattern = [chord[0], chord[2], chord[1], chord[2]]
    notes = []
    tick = start_tick
    dur = TPB // 2  # eighth note
    for b in range(beats * 2):
        p = pattern[b % 4]
        notes.append(make_note(p, vel, tick, dur))
        tick += dur
    return notes


def arpeggiate(root, chord_type, octave, beats, start_tick, vel=70, ascending=True):
    pitches = chord_pitches(root, chord_type, octave)
    if not ascending:
        pitches = list(reversed(pitches))
    notes = []
    tick = start_tick
    dur = TPB // 2
    for b in range(beats * 2):
        p = pitches[b % len(pitches)]
        notes.append(make_note(p, vel, tick, dur))
        tick += dur
    return notes


def melody_line(pitches, start_tick, note_dur, vel_range=(60, 90), rests=0.1, rng=None):
    """Turn a pitch list into a melody with slight velocity variation and occasional rests."""
    if rng is None:
        rng = random.Random(42)
    notes = []
    tick = start_tick
    for p in pitches:
        if rng.random() > rests:
            vel = rng.randint(*vel_range)
            notes.append(make_note(p, vel, tick, note_dur - TPB // 16))
        tick += note_dur
    return notes


def chord_block(roots, chord_types, beats_each, start_tick, vel=65, octave=4):
    notes = []
    tick = start_tick
    dur = beats_each * TPB
    for root, ct in zip(roots, chord_types):
        for p in chord_pitches(root, ct, octave):
            notes.append(make_note(p, vel, tick, dur - TPB // 8))
        tick += dur
    return notes


# ── Classical pieces ─────────────────────────────────────────────────────────

def gen_classical_waltz(seed=1):
    rng = random.Random(seed)
    root = rng.choice([60, 62, 65, 67])  # C, D, F, G
    scale = scale_pitches(root, "major", octaves=2)
    notes = []
    bars = 16
    for bar in range(bars):
        start = bar * TPB * 3  # 3/4 time
        # Melody: quarter notes on beats 1 & 3
        mel_pitches = [rng.choice(scale[7:]) for _ in range(3)]
        notes += melody_line(mel_pitches, start, TPB, vel_range=(65, 85), rests=0.05, rng=rng)
        # Bass: root on beat 1
        bass_root = root - 12 + rng.choice([0, 5, 7])
        notes.append(make_note(bass_root, 60, start, TPB - 20))
        # Chord: beats 2 & 3
        for beat in [1, 2]:
            ch = chord_pitches(root + rng.choice([0, 5, 7]), "maj", octave=3)
            for p in ch:
                notes.append(make_note(p, 55, start + beat * TPB, TPB - 20))
    return notes


def gen_classical_sonata(seed=2):
    rng = random.Random(seed)
    root = 60  # C major
    scale = scale_pitches(root, "major", octaves=3)
    notes = []
    # Exposition: 8 bars melody over alberti bass
    for bar in range(8):
        start = bar * TPB * 4
        notes += alberti_bass(root, 4, start)
        mel = [rng.choice(scale[7:14]) for _ in range(8)]
        notes += melody_line(mel, start, TPB // 2, vel_range=(70, 90), rng=rng)
    # Development: modulate, more tension
    for bar in range(8, 16):
        start = bar * TPB * 4
        mod_root = root + 7  # G major
        mod_scale = scale_pitches(mod_root, "major", octaves=2)
        notes += alberti_bass(mod_root, 4, start)
        mel = [rng.choice(mod_scale[7:]) for _ in range(8)]
        notes += melody_line(mel, start, TPB // 2, vel_range=(75, 95), rng=rng)
    # Recapitulation: return to tonic
    for bar in range(16, 24):
        start = bar * TPB * 4
        notes += alberti_bass(root, 4, start)
        mel = [rng.choice(scale[7:14]) for _ in range(8)]
        notes += melody_line(mel, start, TPB // 2, vel_range=(65, 85), rng=rng)
    return notes


def gen_classical_minuet(seed=3):
    rng = random.Random(seed)
    root = 65  # F major
    scale = scale_pitches(root, "major", octaves=2)
    notes = []
    prog_roots = [root, root+5, root+7, root, root+5, root+7, root+2, root]
    prog_types = ["maj","maj","maj","maj","maj","maj","min","maj"]
    # A section: 8 bars
    for bar in range(8):
        start = bar * TPB * 3
        r, ct = prog_roots[bar % len(prog_roots)], prog_types[bar % len(prog_types)]
        for p in chord_pitches(r, ct, octave=3):
            notes.append(make_note(p, 55, start, TPB * 3 - 20))
        mel = [rng.choice(scale) for _ in range(6)]
        notes += melody_line(mel, start, TPB // 2, vel_range=(62, 82), rng=rng)
    return notes


def gen_baroque_fugue(seed=4):
    """Two-voice fugue: subject enters in answer a 5th higher."""
    rng = random.Random(seed)
    root = 60
    scale = scale_pitches(root, "minor", octaves=2)
    subject = [scale[i % len(scale)] for i in [0,2,3,4,5,4,3,2]]
    answer  = [p + 7 for p in subject]

    notes = []
    # Voice 1: subject at tick 0
    notes += melody_line(subject, 0, TPB // 2, vel_range=(75,90), rng=rng)
    # Voice 2: answer enters 4 bars later
    delay = TPB * 4
    notes += melody_line(answer, delay, TPB // 2, vel_range=(70,85), rng=rng)
    # Continue both voices for several repetitions
    for rep in range(1, 6):
        offset1 = rep * TPB * 4 + len(subject) * TPB // 2
        offset2 = offset1 + delay
        notes += melody_line(subject, offset1, TPB // 2, vel_range=(70,88), rng=rng)
        notes += melody_line(answer,  offset2, TPB // 2, vel_range=(65,82), rng=rng)
    return notes


def gen_romantic_nocturne(seed=5):
    rng = random.Random(seed)
    root = 64  # E
    scale = scale_pitches(root, "major", octaves=3)
    notes = []
    bars = 20
    prog = [root, root+5, root+9, root+7, root+5, root+4, root+2, root]
    for bar in range(bars):
        start = bar * TPB * 4
        r = prog[bar % len(prog)]
        # Long flowing melody
        mel = [rng.choice(scale[7:]) for _ in range(4)]
        notes += melody_line(mel, start, TPB, vel_range=(55,78), rests=0.08, rng=rng)
        # Arpeggiated left hand
        notes += arpeggiate(r, "maj7", 3, 4, start, vel=50, ascending=True)
    return notes


# ── Jazz pieces ───────────────────────────────────────────────────────────────

def gen_jazz_blues(seed=10):
    rng = random.Random(seed)
    root = 60  # C blues
    scale = scale_pitches(root, "blues", octaves=2)
    notes = []
    reps = 3  # 3 choruses of 12-bar blues
    for rep in range(reps):
        for bar, bass_root_offset in enumerate(JAZZ_BLUES):
            start = (rep * 12 + bar) * TPB * 4
            bass_root = root + bass_root_offset - 12
            # Walking bass: 4 quarter notes
            walk = [bass_root, bass_root+4, bass_root+7, bass_root+10]
            for beat, bp in enumerate(walk):
                notes.append(make_note(bp, 65, start + beat * TPB, TPB - 20))
            # Chord on beats 2 & 4
            for beat in [1, 3]:
                for p in chord_pitches(root + bass_root_offset, "dom7", octave=4):
                    notes.append(make_note(p, 55, start + beat * TPB, TPB - 30))
            # Melody: improvised feel
            mel_len = rng.randint(4, 8)
            mel = [rng.choice(scale) for _ in range(mel_len)]
            dur = (TPB * 4) // mel_len
            notes += melody_line(mel, start, dur, vel_range=(68,88), rests=0.15, rng=rng)
    return notes


def gen_jazz_swing(seed=11):
    rng = random.Random(seed)
    root = 62  # D
    scale = scale_pitches(root, "dorian", octaves=2)
    # ii-V-I in D
    prog_roots = [root, root+5, root-2]   # Dm7, Gm7, Cmaj7 (relative)
    prog_types = ["min7", "dom7", "maj7"]
    notes = []
    bars = 24
    for bar in range(bars):
        start = bar * TPB * 4
        r = prog_roots[bar % 3]
        ct = prog_types[bar % 3]
        # Comping: chords on beats 2 and 4 (swung)
        for beat in [1, 3]:
            swing_offset = TPB // 6  # swing feel
            for p in chord_pitches(r, ct, octave=4):
                notes.append(make_note(p, 58, start + beat * TPB + swing_offset, TPB // 2))
        # Melody line
        mel = [rng.choice(scale) for _ in range(8)]
        notes += melody_line(mel, start, TPB // 2, vel_range=(65,85), rests=0.12, rng=rng)
        # Upright bass roots
        notes.append(make_note(r - 12, 70, start, TPB - 20))
        notes.append(make_note(r - 12 + 7, 65, start + 2 * TPB, TPB - 20))
    return notes


def gen_jazz_bossa_nova(seed=12):
    rng = random.Random(seed)
    root = 67  # G
    scale = scale_pitches(root, "major", octaves=2)
    notes = []
    bars = 20
    # Bossa nova rhythm: syncopated
    beat_pattern = [0, TPB*0.75, TPB*1.5, TPB*2.25, TPB*3]
    prog = [(root,"maj7"),(root+5,"maj7"),(root+2,"min7"),(root+7,"dom7")]
    for bar in range(bars):
        start = bar * TPB * 4
        r, ct = prog[bar % len(prog)]
        for offset in beat_pattern:
            for p in chord_pitches(r, ct, octave=4):
                notes.append(make_note(p, 52, start + int(offset), TPB // 3))
        # Melody
        mel = [rng.choice(scale[4:]) for _ in range(6)]
        notes += melody_line(mel, start, TPB * 2 // 3, vel_range=(60,80), rng=rng)
        # Bass on 1 and 3
        notes.append(make_note(r - 12, 68, start, TPB - 20))
        notes.append(make_note(r - 12 + 5, 62, start + 2 * TPB, TPB - 20))
    return notes


def gen_jazz_modal(seed=13):
    """Miles Davis–style modal jazz."""
    rng = random.Random(seed)
    root = 60  # C
    scale_d = scale_pitches(root+2, "dorian", octaves=2)   # D Dorian
    scale_m = scale_pitches(root+9, "mixolydian", octaves=2) # A Mixolydian
    notes = []
    for section, (scale, chord_root, ct) in enumerate([
        (scale_d, root+2, "min7"),
        (scale_m, root+9, "dom7"),
        (scale_d, root+2, "min7"),
    ]):
        for bar in range(8):
            start = (section * 8 + bar) * TPB * 4
            # Sparse chord on beat 1
            for p in chord_pitches(chord_root, ct, octave=4):
                notes.append(make_note(p, 50, start, TPB * 2))
            # Free melodic phrases
            mel_pitches = [rng.choice(scale) for _ in range(rng.randint(5, 10))]
            notes += melody_line(mel_pitches, start, TPB * 4 // len(mel_pitches),
                                 vel_range=(62, 85), rests=0.2, rng=rng)
            # Bass drone
            notes.append(make_note(chord_root - 12, 65, start, TPB * 4 - 10))
    return notes


def gen_jazz_ballad(seed=14):
    rng = random.Random(seed)
    root = 64  # E
    scale = scale_pitches(root, "major", octaves=3)
    notes = []
    prog = [
        (root, "maj7"), (root+2, "min7"), (root+5, "maj7"),
        (root+7, "dom7"), (root, "maj7"), (root-1, "dom7"),
        (root+5, "maj7"), (root+7, "dom7"),
    ]
    for bar in range(16):
        start = bar * TPB * 4
        r, ct = prog[bar % len(prog)]
        # Full chord whole notes
        for p in chord_pitches(r, ct, octave=4):
            notes.append(make_note(p, 55, start, TPB * 4 - 30))
        # Slow melody
        mel = [rng.choice(scale[7:]) for _ in range(4)]
        notes += melody_line(mel, start, TPB, vel_range=(58, 80), rests=0.1, rng=rng)
        # Bass
        notes.append(make_note(r - 12, 62, start, TPB * 2 - 20))
        notes.append(make_note(r - 12 + 7, 58, start + 2 * TPB, TPB * 2 - 20))
    return notes


def gen_pentatonic_improv(seed=15, style="jazz"):
    rng = random.Random(seed)
    root = rng.choice([60, 62, 65, 67, 69])
    scale = scale_pitches(root, "pentatonic", octaves=3)
    notes = []
    bars = 16
    for bar in range(bars):
        start = bar * TPB * 4
        num_notes = rng.randint(4, 12)
        mel = [rng.choice(scale) for _ in range(num_notes)]
        dur = (TPB * 4) // num_notes
        notes += melody_line(mel, start, dur, vel_range=(65, 90), rests=0.15, rng=rng)
        # Light chord backing
        for p in chord_pitches(root, "min7", octave=3):
            notes.append(make_note(p, 45, start, TPB * 4 - 10))
    return notes


def gen_whole_tone_study(seed=16):
    """Debussy-inspired whole-tone scale study."""
    rng = random.Random(seed)
    root = 60
    scale = scale_pitches(root, "whole_tone", octaves=2)
    notes = []
    bars = 12
    for bar in range(bars):
        start = bar * TPB * 4
        mel = [rng.choice(scale) for _ in range(rng.randint(4, 8))]
        notes += melody_line(mel, start, TPB // 2, vel_range=(50, 75), rests=0.1, rng=rng)
        notes += arpeggiate(root + (bar % 6) * 2, "aug", 4, 4, start, vel=45, ascending=bar % 2 == 0)
    return notes


def gen_diminished_etude(seed=17):
    rng = random.Random(seed)
    root = 60
    scale = scale_pitches(root, "diminished", octaves=2)
    notes = []
    for bar in range(12):
        start = bar * TPB * 4
        mel = [rng.choice(scale) for _ in range(8)]
        notes += melody_line(mel, start, TPB // 2, vel_range=(65, 88), rng=rng)
        for p in chord_pitches(root + bar % 3, "dim", octave=3):
            notes.append(make_note(p, 50, start, TPB * 2))
    return notes


# ── Main: generate and save all pieces ──────────────────────────────────────

PIECES = [
    # Classical
    ("classical_waltz_01",       lambda: gen_classical_waltz(1)),
    ("classical_waltz_02",       lambda: gen_classical_waltz(7)),
    ("classical_sonata_01",      lambda: gen_classical_sonata(2)),
    ("classical_sonata_02",      lambda: gen_classical_sonata(9)),
    ("classical_minuet_01",      lambda: gen_classical_minuet(3)),
    ("classical_minuet_02",      lambda: gen_classical_minuet(11)),
    ("baroque_fugue_01",         lambda: gen_baroque_fugue(4)),
    ("baroque_fugue_02",         lambda: gen_baroque_fugue(13)),
    ("romantic_nocturne_01",     lambda: gen_romantic_nocturne(5)),
    ("romantic_nocturne_02",     lambda: gen_romantic_nocturne(17)),
    ("whole_tone_study_01",      lambda: gen_whole_tone_study(16)),
    ("whole_tone_study_02",      lambda: gen_whole_tone_study(23)),
    ("diminished_etude_01",      lambda: gen_diminished_etude(17)),
    ("diminished_etude_02",      lambda: gen_diminished_etude(31)),
    # Jazz
    ("jazz_blues_01",            lambda: gen_jazz_blues(10)),
    ("jazz_blues_02",            lambda: gen_jazz_blues(20)),
    ("jazz_blues_03",            lambda: gen_jazz_blues(30)),
    ("jazz_swing_01",            lambda: gen_jazz_swing(11)),
    ("jazz_swing_02",            lambda: gen_jazz_swing(22)),
    ("jazz_bossa_nova_01",       lambda: gen_jazz_bossa_nova(12)),
    ("jazz_bossa_nova_02",       lambda: gen_jazz_bossa_nova(24)),
    ("jazz_modal_01",            lambda: gen_jazz_modal(13)),
    ("jazz_modal_02",            lambda: gen_jazz_modal(26)),
    ("jazz_ballad_01",           lambda: gen_jazz_ballad(14)),
    ("jazz_ballad_02",           lambda: gen_jazz_ballad(28)),
    ("pentatonic_improv_01",     lambda: gen_pentatonic_improv(15)),
    ("pentatonic_improv_02",     lambda: gen_pentatonic_improv(25)),
    ("pentatonic_improv_03",     lambda: gen_pentatonic_improv(35)),
]


if __name__ == "__main__":
    print(f"Generating {len(PIECES)} MIDI files into {OUTPUT_DIR}/")
    for name, fn in PIECES:
        notes = fn()
        path = os.path.join(OUTPUT_DIR, f"{name}.mid")
        write_midi(path, notes, ticks_per_beat=TPB, tempo=TEMPO)
        print(f"  ✓  {name}.mid  ({len(notes)} notes)")
    print(f"\nDone. {len(PIECES)} files written.")
