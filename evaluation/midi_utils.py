
import os

import numpy as np

from ttm.utils import note_seq_to_midi

MIN_PIANO_PITCH = 21


def sequences_to_midi(pitch_sequence, feature_sequence, output_path):
    if len(pitch_sequence) != len(feature_sequence):
        raise ValueError("Pitch and feature sequences must have the same length")
    
    n = len(pitch_sequence)
    note_array = np.zeros((n, 4))
    
    note_array[0, 0] = pitch_sequence[0] + MIN_PIANO_PITCH
    note_array[0, 1] = 0.0
    note_array[0, 2] = np.expm1(feature_sequence[0, 2]) if feature_sequence[0, 2] > 0 else 0.1
    note_array[0, 3] = int(feature_sequence[0, 3]) if feature_sequence[0, 3] > 0 else 64
    
    current_time = 0.0
    for i in range(1, n):
        note_array[i, 0] = pitch_sequence[i] + MIN_PIANO_PITCH
        
        delta_t = np.expm1(feature_sequence[i, 1]) if feature_sequence[i, 1] > 0 else 0.1
        current_time += delta_t
        note_array[i, 1] = current_time
        
        note_array[i, 2] = np.expm1(feature_sequence[i, 2]) if feature_sequence[i, 2] > 0 else 0.1
        note_array[i, 3] = int(feature_sequence[i, 3]) if feature_sequence[i, 3] > 0 else 64
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mid = note_seq_to_midi(note_array)
    mid.save(output_path)
    print(f"Saved MIDI file to {output_path}")

