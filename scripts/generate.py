"""
generate.py — Load the trained MusicLSTM and compose new MIDI files.

Usage
-----
  python3 generate.py [--num N] [--steps S] [--temp T] [--topk K] [--seed SEED]

Outputs
-------
  output/midi/generated_XX.mid   — generated MIDI files
  output/audio/generated_XX.wav  — WAV audio files (pure-Python synthesis)
"""

import sys, os, json, struct, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lstm_model import MusicLSTM
from midi_utils  import write_midi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(__file__)
PROC_DIR    = os.path.join(SCRIPT_DIR, "..", "data", "processed")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "..", "models")
MIDI_OUT    = os.path.join(SCRIPT_DIR, "..", "output", "midi")
AUDIO_OUT   = os.path.join(SCRIPT_DIR, "..", "output", "audio")

os.makedirs(MIDI_OUT,  exist_ok=True)
os.makedirs(AUDIO_OUT, exist_ok=True)

VOCAB_PATH  = os.path.join(PROC_DIR, "vocab.json")
CONFIG_PATH = os.path.join(PROC_DIR, "config.json")
MODEL_PATH  = os.path.join(MODEL_DIR, "music_lstm.npz")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate music with MusicLSTM")
    p.add_argument("--num",   type=int,   default=3,   help="Number of pieces to generate")
    p.add_argument("--steps", type=int,   default=120, help="Notes per piece")
    p.add_argument("--temp",  type=float, default=0.9, help="Sampling temperature (0.5=conservative, 1.2=creative)")
    p.add_argument("--topk",  type=int,   default=10,  help="Top-k filtering (0=disabled)")
    p.add_argument("--seed",  type=int,   default=0)
    return p.parse_args()


# ── Token decoding ─────────────────────────────────────────────────────────────

def decode_token(token: str):
    """
    Parse a token like 'p060_d04_v2' → (pitch=60, dur_steps=4, vel_bin=2)
    Returns None for special tokens.
    """
    if not token.startswith("p"):
        return None
    try:
        parts = token.split("_")
        pitch    = int(parts[0][1:])
        dur_s    = int(parts[1][1:])
        vel_bin  = int(parts[2][1:])
        return pitch, dur_s, vel_bin
    except Exception:
        return None


# ── Tokens → MIDI notes ───────────────────────────────────────────────────────

def tokens_to_notes(token_ids: list, id2token: list,
                    ticks_per_beat: int = 480, steps_per_beat: int = 4) -> list:
    """Convert a sequence of token IDs to MIDI note dicts."""
    step_ticks = ticks_per_beat // steps_per_beat
    notes = []
    current_tick = 0

    for tid in token_ids:
        tok = id2token[tid] if tid < len(id2token) else None
        if tok is None:
            continue
        decoded = decode_token(tok)
        if decoded is None:
            current_tick += step_ticks  # treat special tokens as a rest
            continue
        pitch, dur_steps, vel_bin = decoded
        duration_ticks = dur_steps * step_ticks
        velocity = 32 + vel_bin * 32  # map bin 0-3 → 32,64,96,128 → cap at 127
        velocity = min(127, velocity)

        notes.append({
            "pitch":      pitch,
            "velocity":   velocity,
            "start_tick": current_tick,
            "end_tick":   current_tick + duration_ticks,
        })
        current_tick += step_ticks  # advance one 16th note per token

    return notes


# ── Simple WAV synthesis (additive sine waves, no external libs) ──────────────

SAMPLE_RATE = 22050

def midi_pitch_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

def piano_envelope(n_samp: int, dur_s: float, velocity: float) -> np.ndarray:
    """
    Realistic piano amplitude envelope:
      - Very fast attack  (~3 ms)
      - Quick decay to sustain level  (~80 ms, higher partials decay faster)
      - Gentle sustain with slow natural roll-off
      - Short release tail
    velocity shapes how bright/loud the attack is.
    """
    sr = SAMPLE_RATE
    attack  = max(2, int(0.003 * sr))
    decay   = min(int(0.08 * sr), n_samp // 3)
    release = min(int(0.12 * sr), n_samp // 4)

    sustain_level = 0.45 + 0.25 * velocity   # 0.45–0.70 depending on velocity

    env = np.ones(n_samp, dtype=np.float32)

    # Attack
    env[:attack] = np.linspace(0.0, 1.0, attack)

    # Decay → sustain
    if decay > 0 and attack < n_samp:
        decay_end = min(attack + decay, n_samp)
        env[attack:decay_end] = np.linspace(1.0, sustain_level,
                                             decay_end - attack)

    # Sustain: slow exponential roll-off (piano strings lose energy)
    sustain_start = min(attack + decay, n_samp)
    if sustain_start < n_samp:
        sustain_len = n_samp - sustain_start
        roll = np.exp(-2.5 * np.linspace(0, dur_s, sustain_len))
        env[sustain_start:] = sustain_level * roll.astype(np.float32)

    # Release fade
    if release > 0 and n_samp > release:
        env[-release:] *= np.linspace(1.0, 0.0, release)

    return env


def piano_wave(hz: float, n_samp: int, dur_s: float, velocity: float) -> np.ndarray:
    """
    Additive piano tone:
    Harmonic series with amplitudes and decay rates matching a real piano.
    Higher harmonics decay faster (inharmonicity approximation).
    """
    sr   = SAMPLE_RATE
    t    = np.linspace(0, dur_s, n_samp, endpoint=False, dtype=np.float64)

    # (harmonic_number, relative_amplitude, decay_rate)
    # Real piano: bright attack from harmonics 2-6, slow fundamental sustain
    harmonics = [
        (1,  1.00, 0.5),   # fundamental  — long sustain
        (2,  0.60, 1.2),   # octave        — decays faster
        (3,  0.25, 2.5),   # fifth+octave
        (4,  0.18, 4.0),   # two octaves
        (5,  0.12, 6.0),   # major third above
        (6,  0.08, 8.0),   # fifth above
        (7,  0.05, 12.0),  # minor seventh (gives warmth)
        (8,  0.03, 18.0),  # brightness sparkle
        (9,  0.02, 25.0),
        (10, 0.01, 35.0),
    ]

    wave = np.zeros(n_samp, dtype=np.float64)
    for harmonic, amp, decay_rate in harmonics:
        freq = hz * harmonic
        if freq > sr / 2:          # above Nyquist — skip
            break
        # Per-harmonic exponential decay (higher partials fade faster)
        partial_env = np.exp(-decay_rate * t)
        wave += amp * partial_env * np.sin(2 * np.pi * freq * t)

    # Slight odd-even harmonic imbalance (piano string characteristic)
    wave *= (1.0 + 0.04 * np.sin(2 * np.pi * hz * 0.5 * t))  # subtle beating

    return wave.astype(np.float32)


def synthesise_notes(notes: list, ticks_per_beat: int = 480,
                     tempo: int = 500000) -> np.ndarray:
    """
    Render notes to a float32 mono PCM array using piano synthesis.
    """
    if not notes:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    sec_per_tick = (tempo / 1_000_000) / ticks_per_beat
    max_tick     = max(n["end_tick"] for n in notes)
    total_sec    = max_tick * sec_per_tick + 2.5   # 2.5 s reverb tail
    total_samp   = int(total_sec * SAMPLE_RATE)
    audio        = np.zeros(total_samp, dtype=np.float32)

    for n in notes:
        hz      = midi_pitch_to_hz(n["pitch"])
        vel     = n.get("velocity", 80) / 127.0       # 0.0–1.0
        amp     = 0.18 + 0.22 * vel                   # 0.18–0.40, softer overall
        start_s = n["start_tick"] * sec_per_tick

        # Piano notes ring longer than their notated duration
        notated_dur = (n["end_tick"] - n["start_tick"]) * sec_per_tick
        ring_dur    = notated_dur + 0.6 + 0.8 * vel   # ring-off time

        start_samp = int(start_s * SAMPLE_RATE)
        dur_samp   = int(ring_dur * SAMPLE_RATE)
        end_samp   = min(start_samp + dur_samp, total_samp)
        n_samp     = end_samp - start_samp
        if n_samp <= 0:
            continue

        wave = piano_wave(hz, n_samp, ring_dur, vel)
        env  = piano_envelope(n_samp, ring_dur, vel)
        audio[start_samp:end_samp] += wave * env * amp

    # Soft knee limiter — gentle, not a hard clip
    audio = np.tanh(audio * 1.5) / 1.5

    # Final normalise
    peak = np.abs(audio).max()
    if peak > 0.85:
        audio = audio * (0.85 / peak)

    return audio


def write_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write float32 mono PCM to a standard WAV file (no external libs)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int = (pcm * 32767).astype(np.int16)
    data_bytes = pcm_int.tobytes()
    n_samples  = len(pcm_int)
    byte_rate  = sample_rate * 2   # 16-bit mono
    data_chunk_size = n_samples * 2
    riff_size  = 36 + data_chunk_size

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # chunk size
        f.write(struct.pack("<H", 1))            # PCM
        f.write(struct.pack("<H", 1))            # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", 2))            # block align
        f.write(struct.pack("<H", 16))           # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_chunk_size))
        f.write(data_bytes)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load vocab & config
    if not os.path.exists(VOCAB_PATH):
        print("Vocab not found. Run preprocess.py first.")
        sys.exit(1)
    with open(VOCAB_PATH)  as f: vocab  = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)

    id2token  = vocab["id2token"]
    token2id  = vocab["token2id"]
    sos_id    = config["sos_id"]
    window    = config["window_size"]
    spb       = config["steps_per_beat"]

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)
    model = MusicLSTM.load(MODEL_PATH)

    print(f"\nGenerating {args.num} piece(s) × {args.steps} notes "
          f"| temp={args.temp} | top_k={args.topk}")
    print("-" * 60)

    rng = np.random.RandomState(args.seed)

    for i in range(1, args.num + 1):
        # Build a seed: SOS + random musical tokens from vocab
        music_ids = [j for j, t in enumerate(id2token) if t.startswith("p")]
        if len(music_ids) >= window // 2:
            seed_sample = rng.choice(music_ids, size=window // 2, replace=True)
        else:
            seed_sample = np.array([sos_id] * (window // 2), dtype=np.int32)
        seed = np.array([sos_id] + list(seed_sample), dtype=np.int32)

        # Generate
        gen_ids = model.generate(
            seed, num_steps=args.steps,
            temperature=args.temp, top_k=args.topk,
            rng=rng
        )

        # Decode tokens → MIDI notes
        all_ids = list(seed) + gen_ids
        notes = tokens_to_notes(all_ids, id2token, ticks_per_beat=480, steps_per_beat=spb)
        print(f"  Piece {i:02d}: {len(gen_ids)} tokens → {len(notes)} notes")

        if not notes:
            print(f"    ⚠  No decodeable notes — skipping.")
            continue

        # Save MIDI
        midi_path = os.path.join(MIDI_OUT, f"generated_{i:02d}.mid")
        write_midi(midi_path, notes, ticks_per_beat=480, tempo=500000)
        print(f"    MIDI  → {midi_path}")

        # Synthesise & save WAV
        audio = synthesise_notes(notes, ticks_per_beat=480, tempo=500000)
        wav_path = os.path.join(AUDIO_OUT, f"generated_{i:02d}.wav")
        write_wav(wav_path, audio)
        dur_s = len(audio) / SAMPLE_RATE
        print(f"    WAV   → {wav_path}  ({dur_s:.1f}s)")

    print("\nGeneration complete.")
    print(f"MIDI files : {MIDI_OUT}")
    print(f"WAV  files : {AUDIO_OUT}")


if __name__ == "__main__":
    main()
