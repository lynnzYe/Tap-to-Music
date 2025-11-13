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
        print(f"Error reading {path}: {e}")
        return None


def find_duplicate_midi(maestro_files, asap_files):
    print(f"Collected {len(maestro_files)} MAESTRO files")
    print(f"Collected {len(asap_files)} ASAP files")

    maestro_hashes = {}
    print("Hashing MAESTRO files...")
    for p in tqdm(maestro_files):
        h = midi_hash(p)
        if h:
            maestro_hashes[h] = p
    overlaps = []
    print("Checking ASAP files...")
    for p in tqdm(asap_files):
        h = midi_hash(p)
        if h and h in maestro_hashes:
            overlaps.append((p, maestro_hashes[h]))

    print(f"\nFound {len(overlaps)} overlapping MIDI performances.")
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
    note_sequence = reduce(lambda x, y: x + y, [inst.notes for inst in midi_data.instruments])
    note_sequence = sorted(note_sequence, key=cmp_to_key(compare_note_order))

    # conver to numpy array
    note_sequence = np.array([(note.pitch, note.start, note.end - note.start, note.velocity) \
                              for note in note_sequence])
    return note_sequence


def main():
    overlaps = find_duplicate_midi(
        '/Users/kurono/Documents/code/data/acpas/asap',
        '/Users/kurono/Documents/code/data/maestro-v3.0.0'
    )
    print(overlaps)


if __name__ == "__main__":
    main()
