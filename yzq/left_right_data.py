"""
Use a trained HANNDs model to split a single-track piano MIDI into left/right hands.
The model outputs a per-window, per-pitch class (0=none, 1=left, 2=right); we map
each note to the class at its onset window and save two MIDIs.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import pretty_midi
from tqdm import tqdm
from torch.serialization import add_safe_globals

# Ensure bundled hannds code is importable even when run from repo root.
REPO_ROOT = Path(__file__).resolve().parent
HANND_DIR = REPO_ROOT / "hannds"
if str(HANND_DIR) not in sys.path:
    sys.path.append(str(HANND_DIR))

from hannds.network_zoo import Network88


DEFAULT_CHECKPOINT = "hannds/models/11-22-2325-network=88(LSTM)_hidden=70_layers=2_cv=1/model.pt"
DEFAULT_OUTPUT_DIR = "outputs/hannds_split"

# Architecture defaults from train_hannds.py when training Network88 on windowed data.
HIDDEN_SIZE = 70
LAYERS = 2
RNN_TYPE = "LSTM"
BIDIRECTIONAL = False
N_FEATURES = 88
N_CATEGORIES = 3  # 0: none, 1: left, 2: right
MS_WINDOW = 20    # must match training (convert_windowed uses 20ms windows)


def build_model(device, checkpoint_path, hidden_size=HIDDEN_SIZE, layers=LAYERS,
                bidirectional=BIDIRECTIONAL, rnn_type=RNN_TYPE):
    add_safe_globals([Network88])
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # weights_only not supported in older torch versions
        state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, torch.nn.Module):
        model = state
    else:
        model = Network88(hidden_size, layers, bidirectional, N_FEATURES, N_CATEGORIES, rnn_type)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def midi_to_model_input(midi_path, ms_window=MS_WINDOW):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [note for inst in pm.instruments for note in inst.notes]
    if not notes:
        raise ValueError(f"No notes found in {midi_path}")

    samples_per_sec = 1000 // ms_window
    end_time = max(note.end for note in notes)
    n_windows = max(1, int(math.ceil(end_time * samples_per_sec)))

    X = np.zeros((n_windows, N_FEATURES), dtype=np.float32)
    for note in notes:
        pitch_idx = note.pitch - 21
        if pitch_idx < 0 or pitch_idx >= N_FEATURES:
            continue
        start_idx = max(0, int(math.floor(note.start * samples_per_sec)))
        end_idx = max(start_idx + 1, int(math.ceil(note.end * samples_per_sec)))
        end_idx = min(end_idx, n_windows)
        X[start_idx:end_idx, pitch_idx] = 1.0

    return torch.tensor(X, dtype=torch.float32).unsqueeze(0), notes, samples_per_sec


def predict_hands(model, device, midi_path):
    X, notes, samples_per_sec = midi_to_model_input(midi_path)
    X = X.to(device)

    with torch.no_grad():
        output, _ = model(X, None)
    # output: [1, T, 88, 3]
    logits = output.squeeze(0).cpu()
    preds = torch.argmax(logits, dim=-1).numpy()  # shape [T, 88]

    labels = []
    num_steps = preds.shape[0]
    for note in notes:
        pitch_idx = note.pitch - 21
        if pitch_idx < 0 or pitch_idx >= N_FEATURES:
            labels.append('R')
            continue
        win = int(math.floor(note.start * samples_per_sec))
        win = min(max(win, 0), num_steps - 1)
        cls = int(preds[win, pitch_idx])
        if cls == 1:
            labels.append('L')
        elif cls == 2:
            labels.append('R')
        else:
            labels.append('R')
    return notes, labels


def write_split_midis(original_midi_path, notes, labels, out_dir, relative_subpath=None):
    left_pm = pretty_midi.PrettyMIDI()
    right_pm = pretty_midi.PrettyMIDI()

    left_inst = pretty_midi.Instrument(program=0, is_drum=False, name="Left Hand")
    right_inst = pretty_midi.Instrument(program=0, is_drum=False, name="Right Hand")

    for note, hand in zip(notes, labels):
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=note.end,
        )
        if hand == 'L':
            left_inst.notes.append(new_note)
        else:
            right_inst.notes.append(new_note)

    left_pm.instruments.append(left_inst)
    right_pm.instruments.append(right_inst)

    out_root = Path(out_dir)
    rel_parent = Path(relative_subpath).parent if relative_subpath else Path()
    stem = Path(relative_subpath).stem if relative_subpath else Path(original_midi_path).stem

    left_dir = out_root / "left" / rel_parent
    right_dir = out_root / "right" / rel_parent
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_path = left_dir / f"{stem}.mid"
    right_path = right_dir / f"{stem}.mid"
    left_pm.write(str(left_path))
    right_pm.write(str(right_path))
    print(f"Saved split MIDIs:\n  L: {left_path}\n  R: {right_path}")


def iter_midis(root: Path):
    for path in root.rglob("*"):
        if path.suffix.lower() in [".mid", ".midi"]:
            yield path


def parse_args():
    parser = argparse.ArgumentParser(description="Split MIDI into left/right hands using a HANNDs model.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input_midi", type=str, help="Path to a single input MIDI file")
    parser.add_argument("--input_dir", type=str, help="Directory of MIDIs to process recursively")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to write split MIDIs")
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE, help="Hidden size used to build Network88")
    parser.add_argument("--layers", type=int, default=LAYERS, help="Number of RNN layers used to build Network88")
    parser.add_argument("--rnn_type", type=str, default=RNN_TYPE, choices=["LSTM", "GRU"], help="RNN type")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional RNN (must match training)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_midi and not args.input_dir:
        raise ValueError("Provide either --input_midi or --input_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, args.checkpoint, args.hidden_size, args.layers, args.bidirectional, args.rnn_type)
    out_root = Path(args.output_dir)

    if args.input_midi:
        notes, labels = predict_hands(model, device, args.input_midi)
        write_split_midis(args.input_midi, notes, labels, out_root)
        return

    # Batch over directory
    in_root = Path(args.input_dir)
    midi_paths = list(iter_midis(in_root))
    if not midi_paths:
        raise ValueError(f"No MIDI files found under {in_root}")

    for midi_path in tqdm(midi_paths, desc="Splitting MIDIs"):
        rel = midi_path.relative_to(in_root)
        try:
            notes, labels = predict_hands(model, device, midi_path)
            write_split_midis(midi_path, notes, labels, out_root, rel)
        except Exception as exc:
            print(f"Skipping {midi_path} due to error: {exc}")


if __name__ == "__main__":
    main()
