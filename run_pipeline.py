#!/usr/bin/env python3
"""
run_pipeline.py — Run the complete Music AI pipeline end-to-end.

Steps:
  1. Generate MIDI training data
  2. Preprocess into note sequences
  3. Train the LSTM model
  4. Generate new music
  5. Report results
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run(script, extra_args=None):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "scripts", script)]
    if extra_args:
        cmd += extra_args
    print(f"\n{'='*60}")
    print(f"▶  {script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n✗ {script} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"✓ {script} complete")

if __name__ == "__main__":
    run("generate_midi_data.py")
    run("preprocess.py")
    # Train: 30 epochs, small model for speed; increase epochs/hidden for quality
    run("train.py", ["--epochs", "30", "--hidden", "128", "--embed", "64", "--batch", "16"])
    run("generate.py", ["--num", "3", "--steps", "120", "--temp", "0.9", "--topk", "10"])
    print("\n" + "="*60)
    print("Pipeline complete!")
    print(f"  MIDI output : {os.path.join(SCRIPT_DIR, 'output', 'midi')}")
    print(f"  WAV  output : {os.path.join(SCRIPT_DIR, 'output', 'audio')}")
    print("="*60)
