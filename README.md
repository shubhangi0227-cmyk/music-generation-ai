# Music Generation AI

A complete end-to-end music generation system using an LSTM trained on classical and jazz MIDI patterns. Built entirely with Python's standard library + NumPy — no TensorFlow or external MIDI libraries required.

---

## Architecture

```
MIDI data → Tokeniser → LSTM (NumPy) → Sampler → MIDI + WAV output
```

- **Data**: 28 procedurally-generated MIDI files (classical: waltz, sonata, minuet, fugue, nocturne; jazz: blues, swing, bossa nova, modal, ballad)
- **Preprocessing**: notes quantised to a 16th-note grid, encoded as `pPITCH_dDUR_vVEL` tokens, sliding-window sequences of length 64
- **Model**: single-layer LSTM with embedding, 49K parameters, Adam optimiser, BPTT
- **Generation**: autoregressive sampling with temperature and top-k filtering
- **Output**: MIDI (`.mid`) + synthesised WAV (`.wav`) using additive sine-wave synthesis with ADSR envelopes

---

## Project Layout

```
music_ai/
├── scripts/
│   ├── midi_utils.py          # Pure-Python MIDI parser & writer
│   ├── generate_midi_data.py  # Generates 28 training MIDI files
│   ├── preprocess.py          # Tokenises MIDIs → sequences.npy + vocab.json
│   ├── lstm_model.py          # MusicLSTM class (NumPy LSTM + Adam)
│   ├── train.py               # Training loop with LR schedule + checkpointing
│   └── generate.py            # Load model → generate MIDI + WAV
├── data/
│   ├── midi/                  # 28 training MIDI files
│   └── processed/             # sequences.npy, vocab.json, config.json
├── models/
│   ├── music_lstm.npz         # Best model checkpoint
│   └── training_log.json      # Loss history
├── output/
│   ├── midi/                  # generated_01.mid … generated_05.mid
│   └── audio/                 # generated_01.wav … generated_05.wav
├── run_pipeline.py            # One-shot pipeline runner
└── README.md
```

---

## Quick Start

Run the full pipeline in one command:
```bash
python3 run_pipeline.py
```

Or step by step:
```bash
# 1. Generate training data
python3 scripts/generate_midi_data.py

# 2. Preprocess
python3 scripts/preprocess.py

# 3. Train (adjust epochs/hidden for quality vs speed)
python3 scripts/train.py --epochs 30 --hidden 128 --embed 64 --batch 32

# 4. Generate music
python3 scripts/generate.py --num 5 --steps 160 --temp 0.9 --topk 12
```

---

## Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num` | 3 | Number of pieces to generate |
| `--steps` | 120 | Notes per piece |
| `--temp` | 0.9 | Sampling temperature (0.5 = conservative, 1.3 = wild) |
| `--topk` | 10 | Top-k filtering (0 = disabled) |
| `--seed` | 0 | Random seed for reproducibility |

---

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 30 | Training epochs |
| `--hidden` | 128 | LSTM hidden dimension (larger = more expressive) |
| `--embed` | 64 | Embedding dimension |
| `--batch` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |

---

## Playing the Output

The `.mid` files in `output/midi/` can be opened in any MIDI player or DAW (GarageBand, VLC, Windows Media Player, MuseScore, etc.).

The `.wav` files in `output/audio/` are standard 22050Hz 16-bit mono WAV files playable in any audio player.

---

## Extending the Project

- **More training data**: Add your own `.mid` files to `data/midi/` and re-run `preprocess.py` + `train.py`
- **Bigger model**: Increase `--hidden` to 256 or 512 and train longer
- **Two-voice model**: Encode multiple simultaneous notes per timestep using chord tokens
- **Conditional generation**: Add style tokens (`<JAZZ>`, `<CLASSICAL>`) to the vocabulary and condition generation on them
