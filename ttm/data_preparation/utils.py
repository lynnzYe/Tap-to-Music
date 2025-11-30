"""
Author: Lynn Ye
Created on: 2025/11/10
Brief: 
"""
import hashlib
import os
from functools import reduce, cmp_to_key
from pathlib import Path

import mido
import numpy as np
import pretty_midi as pm
from tqdm import tqdm

from ttm.config import MAX_PIANO_PITCH, dotenv_config
from ttm.utils import clog

class ChordConstants:
    NOTE_TO_PC = {
        "C": 0, "B#": 0,
        "C#": 1, "Db": 1,
        "D": 2,
        "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4,
        "F": 5, "E#": 5,
        "F#": 6, "Gb": 6,
        "G": 7,
        "G#": 8, "Ab": 8,
        "A": 9,
        "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
    }

    QUALITY_ORDER = ["maj", "min", "dim", "aug", "sus", "maj7", "min7", "7", "other"]
    QUALITY_TO_ID = {q: i for i, q in enumerate(QUALITY_ORDER)}

    NUM_ROOTS = 12
    NUM_QUALITIES = len(QUALITY_ORDER)
    N_ID = NUM_ROOTS * NUM_QUALITIES  # special ID for 'N' (no chord)

def collect_midi_files(root_dir):
    """Recursively collect all .mid/.midi files under a directory."""
    midi_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                midi_paths.append(os.path.join(root, f))
    return midi_paths


def midi_hash(path):
    """Compute a content hash of a MIDI file ignoring metadata."""
    try:
        mid = mido.MidiFile(path)
        # Normalize by extracting (type, note, time, velocity, channel)
        events = []
        for track in mid.tracks:
            for msg in track:
                if msg.type in ["note_on", "note_off"]:
                    events.append((msg.type, msg.note, msg.time, msg.velocity, msg.channel))
        return hashlib.sha256(str(events).encode()).hexdigest()
    except Exception as e:
        clog.error(f"Error reading {path}: {e}")
        return None


def find_duplicate_midi(ref_midi_files, tgt_midi_files):
    clog.debug(f"Collected {len(ref_midi_files)} ref midi files")
    clog.debug(f"Collected {len(tgt_midi_files)} tgt midi files")

    ref_hashes = {}
    clog.debug("Hashing datset 1 files...")
    for p in tqdm(ref_midi_files):
        h = midi_hash(p)
        if h:
            ref_hashes[h] = p
    overlaps = []
    clog.debug("Checking ASAP files...")
    for p in tqdm(tgt_midi_files):
        h = midi_hash(p)
        if h and h in ref_hashes:
            overlaps.append((p, ref_hashes[h]))

    clog.info(f"\nFound {len(overlaps)} overlapping MIDI performances.")
    return overlaps


def compare_note_order(note1, note2):
    """
    Compare two notes by firstly onset and then pitch.
    """
    if note1.start < note2.start:
        return -1
    elif note1.start == note2.start:
        if note1.pitch < note2.pitch:
            return -1
        elif note1.pitch == note2.pitch:
            return 0
        else:
            return 1
    else:
        return 1


def get_note_sequence_from_midi(midi_path):
    """
    Get note sequence from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    """
    midi_data = pm.PrettyMIDI(str(Path(midi_path)))
    if len(midi_data.instruments) == 0:
        clog.error("Flawed midi:")
    note_sequence = reduce(lambda x, y: x + y, [inst.notes for inst in midi_data.instruments])
    note_sequence = sorted(note_sequence, key=cmp_to_key(compare_note_order))

    # conver to numpy array
    note_sequence = np.array([(note.pitch, note.start, note.end - note.start, note.velocity) \
                              for note in note_sequence])
    return note_sequence


def midi_to_tap(notes, pad_value=MAX_PIANO_PITCH + 1):
    """
    notes: np.ndarray of shape (N, 4) = [pitch, onset, duration, velocity]
    returns:
        features: np.ndarray of shape (N, 4), pitch. log1p delta time, log1p duration, velocity
        labels: np.ndarray of shape (N,)
    """
    pitches, onsets, durations, velocities = notes.T
    n = len(notes)

    features = np.zeros((n, 4), dtype=float)
    labels = np.zeros(n, dtype=float)

    # first feature is padding
    features[0] = [pad_value, onsets[0], 0, 0]
    labels[0] = pitches[0]  # predict first pitch (don't learn to predict PAD, PAD for each tap...)

    """
    In theory we can leave the padding step to dataset.__getitem__ during training (less complexity)
    However: label is always paired with the previous pitch, dt. At t=0, we don't have that.
     - which means we won't have the ground truth for the first timestep's label, if we skip this pad step
    """

    for i in range(1, n):
        prev_pitch = pitches[i - 1]
        curr_onset = onsets[i]
        prev_onset = onsets[i - 1]
        prev_dur = durations[i - 1]
        prev_vel = velocities[i - 1]

        delta_t = curr_onset - prev_onset
        eff_dur = np.log1p(min(delta_t, prev_dur))  # causality constraint

        # Log delta time for more perceptually aware IOI encoding (Weber-Fechner law)
        features[i] = [prev_pitch, np.log1p(delta_t), eff_dur, prev_vel]
        labels[i] = pitches[i]

    return features, labels


def main():
    # notes = get_note_sequence_from_midi('/Users/kurono/Documents/code/data/acpas/asap/Bach/Fugue/bwv_846/Shi05M.mid')
    # taps, labels = midi_to_tap(notes)

    duplicates = find_duplicate_midi(collect_midi_files(dotenv_config['HANNDS_PATH']),
                                     collect_midi_files(dotenv_config['MAESTRO_PATH']))
    print(duplicates)


if __name__ == "__main__":
    main()
