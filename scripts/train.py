"""
train.py — Train the MusicLSTM on preprocessed note sequences.

Usage
-----
  python3 train.py [--epochs N] [--lr LR] [--hidden H] [--embed E] [--batch B]

Outputs
-------
  models/music_lstm.npz        — best checkpoint (lowest val loss)
  models/training_log.json     — per-epoch loss history
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lstm_model import MusicLSTM

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(__file__)
PROC_DIR    = os.path.join(SCRIPT_DIR, "..", "data", "processed")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_PATH    = os.path.join(PROC_DIR, "sequences.npy")
VOCAB_PATH  = os.path.join(PROC_DIR, "vocab.json")
CONFIG_PATH = os.path.join(PROC_DIR, "config.json")
MODEL_PATH  = os.path.join(MODEL_DIR, "music_lstm.npz")
LOG_PATH    = os.path.join(MODEL_DIR, "training_log.json")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MusicLSTM")
    p.add_argument("--epochs",  type=int,   default=30,    help="Training epochs")
    p.add_argument("--lr",      type=float, default=1e-3,  help="Learning rate")
    p.add_argument("--hidden",  type=int,   default=128,   help="LSTM hidden dim")
    p.add_argument("--embed",   type=int,   default=64,    help="Embedding dim")
    p.add_argument("--batch",   type=int,   default=32,    help="Batch size (sequences per update)")
    p.add_argument("--val_split", type=float, default=0.1, help="Validation fraction")
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


# ── Mini-batch loss & gradient accumulation ───────────────────────────────────

def batch_loss_and_grads(model: MusicLSTM, batch: np.ndarray) -> tuple:
    """
    batch : (B, window+1) int32
    Returns (mean_loss, accumulated_grads).
    Gradients are averaged over the batch.
    """
    B = batch.shape[0]
    total_loss = 0.0
    accum_grads = {k: np.zeros_like(getattr(model, k)) for k in model._param_names()}

    # Sample up to 8 items per batch for fast updates
    sample_size = min(B, 8)
    indices = np.random.choice(B, sample_size, replace=False)

    for i in indices:
        seq = batch[i]
        inputs  = seq[:-1]
        targets = seq[1:]
        logits, cache, _, _ = model.forward(inputs)
        loss = model.loss(logits, targets)
        grads = model.backward(logits, targets, cache)
        total_loss += loss
        for k in accum_grads:
            accum_grads[k] += grads[k]

    # Average
    for k in accum_grads:
        accum_grads[k] /= sample_size

    return total_loss / sample_size, accum_grads


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.RandomState(args.seed)

    # Load data
    if not os.path.exists(SEQ_PATH):
        print(f"sequences.npy not found. Run  python3 preprocess.py  first.")
        sys.exit(1)

    sequences = np.load(SEQ_PATH)              # (N, window+1)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    vocab_size = config["vocab_size"]
    window     = config["window_size"]
    N          = sequences.shape[0]

    print(f"Dataset: {N} sequences  |  vocab: {vocab_size}  |  window: {window}")

    # Subsample dataset for fast efficient training
    if len(sequences) > 1200:
        sub_idx = rng.choice(len(sequences), 1200, replace=False)
        sequences = sequences[sub_idx]
        N = len(sequences)

    # Train/val split
    idx       = rng.permutation(N)
    val_n     = max(1, int(N * args.val_split))
    val_idx   = idx[:val_n]
    train_idx = idx[val_n:]

    train_data = sequences[train_idx]
    val_data   = sequences[val_idx]
    print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

    # Build model
    model = MusicLSTM(
        vocab_size = vocab_size,
        embed_dim  = args.embed,
        hidden_dim = args.hidden,
        seed       = args.seed
    )
    print(f"\n{model}\n")

    lr          = args.lr
    best_val    = float("inf")
    log         = {"train_loss": [], "val_loss": [], "epochs": []}
    n_batches   = max(1, len(train_data) // args.batch)

    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Time':>7}  {'LR':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Shuffle training data
        perm = rng.permutation(len(train_data))
        train_data = train_data[perm]

        # Learning rate warm-up then decay
        if epoch <= 3:
            current_lr = lr * (epoch / 3)
        elif epoch > args.epochs * 0.7:
            current_lr = lr * 0.1
        else:
            current_lr = lr

        epoch_losses = []
        for b in range(n_batches):
            batch = train_data[b * args.batch : (b + 1) * args.batch]
            if len(batch) == 0:
                continue
            loss, grads = batch_loss_and_grads(model, batch)
            model._adam_step(grads, current_lr)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # Validation loss
        val_losses = []
        for b in range(max(1, len(val_data) // args.batch)):
            batch = val_data[b * args.batch : (b + 1) * args.batch]
            if len(batch) == 0:
                continue
            total = 0.0
            for i in range(len(batch)):
                seq = batch[i]
                logits, _, _, _ = model.forward(seq[:-1])
                total += model.loss(logits, seq[1:])
            val_losses.append(total / len(batch))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  {elapsed:>6.1f}s  {current_lr:>8.2e}")

        log["epochs"].append(epoch)
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            model.save(MODEL_PATH)
            print(f"         ★ best model saved (val={best_val:.4f})")

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved → {LOG_PATH}")
    print(f"Best val loss: {best_val:.4f}")
    print("\nDone. Run  python3 generate.py  to compose new music.")


if __name__ == "__main__":
    main()
