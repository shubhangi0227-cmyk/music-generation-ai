"""
lstm_model.py — Single-layer LSTM + softmax output, implemented in pure NumPy.

Architecture
------------
  Embedding  (vocab_size → embed_dim)
  LSTM       (embed_dim → hidden_dim)
  Dense      (hidden_dim → vocab_size)   + softmax

All parameters are stored as plain NumPy arrays; gradients are computed
via BPTT (Backprop Through Time) over the full sequence.

Usage
-----
  from lstm_model import MusicLSTM
  model = MusicLSTM(vocab_size=300, embed_dim=64, hidden_dim=128)
  model.save("models/model.npz")
  model = MusicLSTM.load("models/model.npz")
"""

import numpy as np
import os, json


# ── Activations ───────────────────────────────────────────────────────────────

def sigmoid(x):
    # numerically stable
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    # x shape: (vocab_size,)
    e = np.exp(x - x.max())
    return e / e.sum()

def softmax_batch(x):
    # x shape: (batch, vocab_size) or (seq, vocab_size)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── Xavier / Glorot initialisation ────────────────────────────────────────────

def glorot(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

def orthogonal(n, rng):
    """Orthogonal initialisation for recurrent weights (helps vanishing gradients)."""
    H = rng.randn(n, n).astype(np.float32)
    U, _, Vt = np.linalg.svd(H)
    return U if U.shape == (n, n) else Vt


# ── MusicLSTM ─────────────────────────────────────────────────────────────────

class MusicLSTM:
    """
    Single-layer LSTM language model.

    Parameters
    ----------
    vocab_size : int
    embed_dim  : int   (embedding dimension)
    hidden_dim : int   (LSTM hidden/cell dimension)
    seed       : int
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, seed: int = 42):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        E  = embed_dim
        H  = hidden_dim
        V  = vocab_size

        # Embedding
        self.We = (rng.randn(V, E) * 0.01).astype(np.float32)

        # LSTM gates: input (i), forget (f), cell (g), output (o)
        # Combined weight matrices for efficiency
        #   Wx : (E, 4H)   input  → gates
        #   Wh : (H, 4H)   hidden → gates
        #   b  : (4H,)
        self.Wx = glorot(E, 4 * H, rng)
        self.Wh = np.hstack([orthogonal(H, rng) for _ in range(4)]).astype(np.float32)
        self.b  = np.zeros(4 * H, dtype=np.float32)
        # Forget gate bias initialised to 1 to encourage long-term memory
        self.b[H:2*H] = 1.0

        # Output projection
        self.Wy = glorot(H, V, rng)
        self.by = np.zeros(V, dtype=np.float32)

        # Adam optimiser state
        self._init_adam()

    # ── Adam ──────────────────────────────────────────────────────────────────

    def _param_names(self):
        return ["We", "Wx", "Wh", "b", "Wy", "by"]

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}
        self._t = 0

    def _adam_step(self, grads: dict, lr: float,
                   beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for k in self._param_names():
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            param = getattr(self, k)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Forward pass (single sequence) ────────────────────────────────────────

    def forward(self, token_ids: np.ndarray, h0=None, c0=None):
        """
        token_ids : (seq_len,)  int32
        Returns
        -------
        logits : (seq_len, vocab_size)
        cache  : dict of intermediate values for BPTT
        """
        T  = len(token_ids)
        H  = self.hidden_dim

        h = np.zeros(H, dtype=np.float32) if h0 is None else h0
        c = np.zeros(H, dtype=np.float32) if c0 is None else c0

        xs, hs, cs = [], [h], [c]
        gates_cache = []   # (i_gate, f_gate, g_gate, o_gate, c_prev, h_prev)
        logits_all  = []

        for t in range(T):
            x   = self.We[token_ids[t]]          # (E,)
            xs.append(x)
            raw = x @ self.Wx + h @ self.Wh + self.b   # (4H,)
            i_g = sigmoid(raw[    :  H])
            f_g = sigmoid(raw[  H:2*H])
            g_g = tanh   (raw[2*H:3*H])
            o_g = sigmoid(raw[3*H:   ])
            c_new = f_g * c + i_g * g_g
            h_new = o_g * tanh(c_new)
            gates_cache.append((i_g, f_g, g_g, o_g, c, h))
            c, h = c_new, h_new
            hs.append(h); cs.append(c)
            logit = h @ self.Wy + self.by       # (V,)
            logits_all.append(logit)

        logits = np.stack(logits_all)            # (T, V)
        cache  = {"xs": xs, "hs": hs, "cs": cs,
                  "gates": gates_cache, "token_ids": token_ids}
        return logits, cache, h, c

    # ── Loss (cross-entropy) ──────────────────────────────────────────────────

    def loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits  : (T, V)
        targets : (T,)  int32
        """
        probs = softmax_batch(logits)           # (T, V)
        T = len(targets)
        loss = -np.mean(np.log(probs[np.arange(T), targets] + 1e-12))
        return float(loss)

    # ── BPTT ─────────────────────────────────────────────────────────────────

    def backward(self, logits: np.ndarray, targets: np.ndarray, cache: dict,
                 clip_norm: float = 5.0) -> dict:
        """
        Returns gradient dict (same keys as param_names).
        Gradients are clipped by global norm.
        """
        T  = len(targets)
        H  = self.hidden_dim
        E  = self.embed_dim

        probs = softmax_batch(logits)                    # (T, V)
        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1
        dlogits /= T                                     # mean over time

        # Output layer gradients
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros(H, dtype=np.float32)
        dc_next = np.zeros(H, dtype=np.float32)

        # Accumulate dWy, dby from all timesteps
        hs = cache["hs"]
        for t in range(T):
            dWy += np.outer(hs[t + 1], dlogits[t])
            dby += dlogits[t]

        # BPTT through LSTM
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b)
        dWe = np.zeros_like(self.We)

        token_ids = cache["token_ids"]
        gates     = cache["gates"]
        cs        = cache["cs"]
        xs        = cache["xs"]

        for t in reversed(range(T)):
            # gradient from output layer at t
            dh = dlogits[t] @ self.Wy.T      # (H,)
            dh += dh_next

            i_g, f_g, g_g, o_g, c_prev, h_prev = gates[t]
            c_t = cs[t + 1]

            # Gradient through h = o * tanh(c)
            tanh_c = tanh(c_t)
            do = dh * tanh_c
            dc = dh * o_g * (1 - tanh_c ** 2) + dc_next

            # Gate gradients
            di = dc * g_g
            df = dc * c_prev
            dg = dc * i_g
            dc_prev = dc * f_g

            # Pre-activation gradients
            d_i_raw = di * i_g * (1 - i_g)
            d_f_raw = df * f_g * (1 - f_g)
            d_g_raw = dg * (1 - g_g ** 2)
            d_o_raw = do * o_g * (1 - o_g)

            draw = np.concatenate([d_i_raw, d_f_raw, d_g_raw, d_o_raw])  # (4H,)

            dWx  += np.outer(xs[t], draw)
            dWh  += np.outer(h_prev, draw)
            db   += draw

            dx = draw @ self.Wx.T                  # (E,)
            dh_next = draw @ self.Wh.T             # (H,)
            dc_next = dc_prev

            # Embedding gradient (sparse update)
            dWe[token_ids[t]] += dx

        grads = {"We": dWe, "Wx": dWx, "Wh": dWh, "b": db, "Wy": dWy, "by": dby}

        # Global gradient clipping
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-6)
            grads = {k: v * scale for k, v in grads.items()}

        return grads

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self, token_ids: np.ndarray, lr: float = 1e-3) -> float:
        """
        token_ids : (seq_len+1,)   first seq_len are inputs, last seq_len are targets
        Returns scalar loss.
        """
        inputs  = token_ids[:-1]
        targets = token_ids[1:]
        logits, cache, _, _ = self.forward(inputs)
        l = self.loss(logits, targets)
        grads = self.backward(logits, targets, cache)
        self._adam_step(grads, lr)
        return l

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, seed_ids: np.ndarray, num_steps: int,
                 temperature: float = 1.0, top_k: int = 0,
                 rng: np.random.RandomState = None) -> list:
        """
        Autoregressively generate `num_steps` new tokens.

        seed_ids    : (seed_len,) int32 — priming sequence
        temperature : float  > 0  (higher = more random)
        top_k       : int    0 = disabled  (nucleus-style if > 0)

        Returns list of generated token ids (length = num_steps).
        """
        if rng is None:
            rng = np.random.RandomState(0)

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        c = np.zeros(self.hidden_dim, dtype=np.float32)

        # Prime the hidden state
        if len(seed_ids) > 0:
            _, _, h, c = self.forward(seed_ids, h, c)

        generated = []
        last_id = int(seed_ids[-1]) if len(seed_ids) > 0 else 1  # SOS

        for _ in range(num_steps):
            logits, _, h, c = self.forward(
                np.array([last_id], dtype=np.int32), h, c
            )
            logit = logits[0]

            # Temperature
            logit = logit / max(temperature, 1e-6)

            # Top-k filtering
            if top_k > 0:
                k = min(top_k, len(logit))
                threshold = np.sort(logit)[::-1][k - 1]
                logit = np.where(logit >= threshold, logit, -1e9)

            prob = softmax(logit)
            # Safety: re-normalise in case of numerical issues
            prob = np.clip(prob, 0, None)
            prob /= prob.sum()

            last_id = int(rng.choice(len(prob), p=prob))
            generated.append(last_id)

        return generated

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            We=self.We, Wx=self.Wx, Wh=self.Wh, b=self.b,
            Wy=self.Wy, by=self.by,
            meta=np.array([self.vocab_size, self.embed_dim, self.hidden_dim])
        )
        print(f"Model saved → {path}.npz" if not path.endswith(".npz") else f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MusicLSTM":
        if not path.endswith(".npz"):
            path += ".npz"
        d = np.load(path)
        meta = d["meta"]
        model = cls(int(meta[0]), int(meta[1]), int(meta[2]))
        for k in ["We", "Wx", "Wh", "b", "Wy", "by"]:
            setattr(model, k, d[k].astype(np.float32))
        model._init_adam()
        print(f"Model loaded ← {path}")
        return model

    # ── Info ──────────────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(getattr(self, k).size for k in self._param_names())

    def __repr__(self):
        return (f"MusicLSTM(vocab={self.vocab_size}, "
                f"embed={self.embed_dim}, hidden={self.hidden_dim}, "
                f"params={self.param_count():,})")
