"""
preprocess.py — Convert MIDI files into note sequences for LSTM training.

Pipeline:
  1. Parse every .mid file in data/midi/
  2. Quantise notes to a 16th-note grid
  3. Encode each note as (pitch, duration_steps, velocity_bin)
  4. Build a vocabulary and map every token to an integer
  5. Slide a fixed-length window to produce (X, y) training pairs
  6. Save   data/processed/sequences.npy   (int32 sequences)
             data/processed/vocab.json      (token ↔ id mapping)
             data/processed/config.json     (window size, vocab size …)

Run: python3 preprocess.py
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from midi_utils import parse_midi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "midi")
PROC_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STEPS_PER_BEAT  = 4          # 16th-note grid
WINDOW_SIZE     = 64         # input sequence length  (notes)
VELOCITY_BINS   = 4          # 0-31, 32-63, 64-95, 96-127
MAX_DUR_STEPS   = 16         # cap duration at 4 beats (16 × 16th)
MIN_NOTES       = WINDOW_SIZE + 1

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


# ── Helpers ───────────────────────────────────────────────────────────────────

def quantise(tick: int, tpb: int) -> int:
    """Map a tick value to a 16th-note grid step."""
    step_size = tpb // STEPS_PER_BEAT
    return max(0, round(tick / step_size))


def velocity_bin(v: int) -> int:
    return min(VELOCITY_BINS - 1, v * VELOCITY_BINS // 128)


def note_token(pitch: int, dur_steps: int, vel_bin: int) -> str:
    dur_steps = max(1, min(dur_steps, MAX_DUR_STEPS))
    return f"p{pitch:03d}_d{dur_steps:02d}_v{vel_bin}"


def midi_to_tokens(path: str) -> list:
    """Parse one MIDI file → list of token strings."""
    try:
        midi = parse_midi(path)
    except Exception as e:
        print(f"    [skip] {os.path.basename(path)}: {e}")
        return []

    tpb   = midi["ticks_per_beat"]
    notes = midi["notes"]
    if not notes:
        return []

    tokens = []
    for n in notes:
        start_q = quantise(n["start_tick"], tpb)
        end_q   = quantise(n["end_tick"],   tpb)
        dur     = max(1, end_q - start_q)
        vb      = velocity_bin(n.get("velocity", 64))
        tokens.append(note_token(n["pitch"], dur, vb))

    return tokens


# ── Build vocabulary ──────────────────────────────────────────────────────────

def build_vocab(all_token_lists: list) -> tuple:
    """
    Returns (token2id dict, id2token list).
    Special tokens are indices 0, 1, 2.
    """
    unique = set()
    for tlist in all_token_lists:
        unique.update(tlist)

    id2token = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(unique)
    token2id = {t: i for i, t in enumerate(id2token)}
    return token2id, id2token


# ── Sliding-window sequences ──────────────────────────────────────────────────

def make_windows(token_ids: list, window: int) -> list:
    """
    Slide a window of length `window+1` over token_ids.
    Each sample: X = ids[i : i+window],  y = ids[i+window]
    Returns list of (window,) arrays.
    """
    samples = []
    for i in range(len(token_ids) - window):
        samples.append(token_ids[i : i + window + 1])
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    midi_files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".mid")
    ])
    if not midi_files:
        print(f"No .mid files found in {DATA_DIR}")
        print("Run  python3 generate_midi_data.py  first.")
        return

    print(f"Found {len(midi_files)} MIDI files. Tokenising …")
    all_token_lists = []
    for path in midi_files:
        tlist = midi_to_tokens(path)
        if len(tlist) >= MIN_NOTES:
            all_token_lists.append(tlist)
            print(f"  ✓  {os.path.basename(path):40s} → {len(tlist)} tokens")
        else:
            print(f"  –  {os.path.basename(path):40s} too short, skipped")

    if not all_token_lists:
        print("No usable sequences found.")
        return

    # Vocabulary
    token2id, id2token = build_vocab(all_token_lists)
    vocab_size = len(id2token)
    print(f"\nVocabulary size: {vocab_size} unique tokens")

    # Encode
    all_samples = []
    for tlist in all_token_lists:
        ids = [token2id[t] for t in tlist]
        windows = make_windows(ids, WINDOW_SIZE)
        all_samples.extend(windows)

    if not all_samples:
        print("No training windows produced.")
        return

    sequences = np.array(all_samples, dtype=np.int32)
    print(f"Training samples:  {sequences.shape[0]} × {sequences.shape[1]}")

    # Save
    seq_path    = os.path.join(PROC_DIR, "sequences.npy")
    vocab_path  = os.path.join(PROC_DIR, "vocab.json")
    config_path = os.path.join(PROC_DIR, "config.json")

    np.save(seq_path, sequences)

    with open(vocab_path, "w") as f:
        json.dump({"token2id": token2id, "id2token": id2token}, f, indent=2)

    config = {
        "vocab_size":      vocab_size,
        "window_size":     WINDOW_SIZE,
        "steps_per_beat":  STEPS_PER_BEAT,
        "velocity_bins":   VELOCITY_BINS,
        "max_dur_steps":   MAX_DUR_STEPS,
        "num_sequences":   sequences.shape[0],
        "pad_id":          token2id[PAD_TOKEN],
        "sos_id":          token2id[SOS_TOKEN],
        "eos_id":          token2id[EOS_TOKEN],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved:")
    print(f"  {seq_path}")
    print(f"  {vocab_path}")
    print(f"  {config_path}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
