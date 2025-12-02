"""
Author: YZQ
Created on: 2025/11/27
Brief: Data augmentation classes for POP909 dataset features
       Extends BaseDataAugmentation from ttm module

       Two augmentation classes available:
       - HandDataAugmentation: For 5-feature data (pitch, dt, dur, vel, hand)
       - ClusterDataAugmentation: For 5-feature data (pitch, dt, dur, vel, window_avg)
       
       Reference: https://github.com/lynnzYe/Tap-to-Music/blob/9bf3016803e6038b302265dabbd68414c9a71b0a/ttm/data_preparation/data_augmentation.py
"""
import random

import numpy as np

# Import base classes from ttm module
from ttm.config import RD_SEED, MIN_PIANO_PITCH, MAX_PIANO_PITCH
from ttm.data_preparation.data_augmentation import (
    BaseDataAugmentation,
    UnconditionalDataAugmentation
)

random.seed(RD_SEED)
np.random.seed(RD_SEED)

# Default window size for cluster/window average computation
DEFAULT_WINDOW_SIZE = 8


def compute_window_avg(pitches, window_size=DEFAULT_WINDOW_SIZE):
    """
    Compute the bidirectional window average of pitches for each position.
    Uses past n and future n notes (excluding current position to avoid label leakage).
    
    Args:
        pitches: np.ndarray of shape (N,) containing pitch values
        window_size: Number of notes to include before AND after current position
    
    Returns:
        np.ndarray of shape (N,) with window averages
    """
    n = len(pitches)
    window_avg = np.zeros(n, dtype=float)
    
    for i in range(n):
        # Past window: pitches[max(0, i-window_size):i]
        # Future window: pitches[i+1:min(n, i+1+window_size)]
        past_start = max(0, i - window_size)
        future_end = min(n, i + 1 + window_size)
        
        # Combine past and future notes (exclude current position i)
        past_notes = pitches[past_start:i]
        future_notes = pitches[i+1:future_end]
        
        combined = np.concatenate([past_notes, future_notes])
        
        if len(combined) == 0:
            # Edge case: single note, no neighbors
            window_avg[i] = pitches[i]
        else:
            window_avg[i] = np.mean(combined)
    
    return window_avg


class HandDataAugmentation(BaseDataAugmentation):
    """
    Data augmentation for 5-feature hand data:
        - pitch (previous note pitch)
        - log1p delta time
        - log1p duration
        - velocity
        - hand (0=left, 1=right)
    
    Applies tempo change and pitch shift from UnconditionalDataAugmentation.
    The hand feature (column 4) remains unchanged after augmentation since
    hand assignment doesn't depend on absolute pitch.
    """
    
    def __init__(self, tempo_change_prob=1.0, tempo_change_range=(0.8, 1.2),
                 pitch_shift_prob=1.0, pitch_shift_range=(-12, 12), **kwargs):
        super().__init__()
        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
    
    def __call__(self, note_sequence, annotations):
        """
        Apply augmentation to note_sequence and annotations.
        
        Args:
            note_sequence: np.ndarray of shape (N, 5) 
                           [pitch, log1p_dt, log1p_dur, velocity, hand]
            annotations: np.ndarray of shape (N,) - pitch labels
        
        Returns:
            Augmented (note_sequence, annotations)
        """
        # Make copies to avoid modifying original
        note_sequence = note_sequence.copy()
        annotations = annotations.copy()
        
        # Apply tempo change (affects columns 1, 2: dt and duration)
        note_sequence, annotations = self._tempo_change(note_sequence, annotations)
        
        # Apply pitch shift (affects column 0 and annotations)
        note_sequence, annotations = self._pitch_shift(note_sequence, annotations)
        
        # Hand feature (column 4) remains unchanged
        # Hand assignment is relative to the piece, not absolute pitch
        
        return note_sequence, annotations
    
    def _tempo_change(self, note_sequence, annotations):
        """Scale time-related features by random tempo factor."""
        if random.random() > self.tempo_change_prob:
            return note_sequence, annotations
        
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        # Scale log1p_dt and log1p_dur (columns 1 and 2)
        note_sequence[:, 1:3] *= 1 / tempo_change_ratio
        
        return note_sequence, annotations
    
    def _pitch_shift(self, note_sequence, annotations):
        """Shift all pitches by random amount within valid range."""
        if random.random() > self.pitch_shift_prob:
            return note_sequence, annotations
        
        # First row is padding with pitch = MAX_PIANO_PITCH + 1
        assert int(note_sequence[0, 0]) == MAX_PIANO_PITCH + 1, \
            'First row should be padding with pitch = MAX_PIANO_PITCH + 1'
        
        # Get all pitches (skip pad row for feature pitches)
        pitches = note_sequence[1:, 0]
        annot_pitches = annotations
        all_pitches = np.concatenate([pitches, annot_pitches], axis=0)
        
        # Calculate valid shift range
        max_down = MIN_PIANO_PITCH - all_pitches.min().item()  # negative
        max_up = MAX_PIANO_PITCH - all_pitches.max().item()    # positive
        
        # Clamp within allowed shift range
        lower = max(self.pitch_shift_range[0], max_down)
        upper = min(self.pitch_shift_range[1], max_up)
        
        if lower > upper:
            return note_sequence, annotations
        
        shift = np.random.randint(lower, upper + 1)
        note_sequence[1:, 0] += shift
        annotations += shift
        
        # Verify pitch range
        assert np.min(note_sequence[1:, 0]) >= MIN_PIANO_PITCH
        assert np.max(note_sequence[1:, 0]) <= MAX_PIANO_PITCH
        assert np.min(annotations) >= MIN_PIANO_PITCH
        assert np.max(annotations) <= MAX_PIANO_PITCH
        
        return note_sequence, annotations


class ClusterDataAugmentation(BaseDataAugmentation):
    """
    Data augmentation for 5-feature cluster/window data:
        - pitch (previous note pitch)
        - log1p delta time
        - log1p duration
        - velocity
        - window_avg (average of previous N pitches)
    
    Applies tempo change and pitch shift from UnconditionalDataAugmentation.
    After pitch shift, the window_avg feature (column 4) is recalculated
    based on the shifted pitches.
    """
    
    def __init__(self, tempo_change_prob=1.0, tempo_change_range=(0.8, 1.2),
                 pitch_shift_prob=1.0, pitch_shift_range=(-12, 12),
                 window_size=DEFAULT_WINDOW_SIZE, **kwargs):
        super().__init__()
        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.window_size = window_size
    
    def __call__(self, note_sequence, annotations):
        """
        Apply augmentation to note_sequence and annotations.
        
        Args:
            note_sequence: np.ndarray of shape (N, 5) 
                           [pitch, log1p_dt, log1p_dur, velocity, window_avg]
            annotations: np.ndarray of shape (N,) - pitch labels
        
        Returns:
            Augmented (note_sequence, annotations)
        """
        # Make copies to avoid modifying original
        note_sequence = note_sequence.copy()
        annotations = annotations.copy()
        
        # Apply tempo change (affects columns 1, 2: dt and duration)
        note_sequence, annotations = self._tempo_change(note_sequence, annotations)
        
        # Apply pitch shift (affects column 0 and annotations)
        note_sequence, annotations = self._pitch_shift(note_sequence, annotations)
        
        # Recalculate window_avg (column 4) based on shifted pitches
        if note_sequence.shape[1] >= 5:
            note_sequence = self._update_window_avg(note_sequence, annotations)
        
        return note_sequence, annotations
    
    def _tempo_change(self, note_sequence, annotations):
        """Scale time-related features by random tempo factor."""
        if random.random() > self.tempo_change_prob:
            return note_sequence, annotations
        
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        # Scale log1p_dt and log1p_dur (columns 1 and 2)
        note_sequence[:, 1:3] *= 1 / tempo_change_ratio
        
        return note_sequence, annotations
    
    def _pitch_shift(self, note_sequence, annotations):
        """Shift all pitches by random amount within valid range."""
        if random.random() > self.pitch_shift_prob:
            return note_sequence, annotations
        
        # First row is padding with pitch = MAX_PIANO_PITCH + 1
        assert int(note_sequence[0, 0]) == MAX_PIANO_PITCH + 1, \
            'First row should be padding with pitch = MAX_PIANO_PITCH + 1'
        
        # Get all pitches (skip pad row for feature pitches)
        pitches = note_sequence[1:, 0]
        annot_pitches = annotations
        all_pitches = np.concatenate([pitches, annot_pitches], axis=0)
        
        # Calculate valid shift range
        max_down = MIN_PIANO_PITCH - all_pitches.min().item()  # negative
        max_up = MAX_PIANO_PITCH - all_pitches.max().item()    # positive
        
        # Clamp within allowed shift range
        lower = max(self.pitch_shift_range[0], max_down)
        upper = min(self.pitch_shift_range[1], max_up)
        
        if lower > upper:
            return note_sequence, annotations
        
        shift = np.random.randint(lower, upper + 1)
        note_sequence[1:, 0] += shift
        annotations += shift
        
        # Also shift window_avg by the same amount (it's an average of pitches)
        if note_sequence.shape[1] >= 5:
            note_sequence[:, 4] += shift
        
        # Verify pitch range
        assert np.min(note_sequence[1:, 0]) >= MIN_PIANO_PITCH
        assert np.max(note_sequence[1:, 0]) <= MAX_PIANO_PITCH
        assert np.min(annotations) >= MIN_PIANO_PITCH
        assert np.max(annotations) <= MAX_PIANO_PITCH
        
        return note_sequence, annotations
    
    def _update_window_avg(self, note_sequence, annotations):
        """
        Recalculate window_avg feature based on current pitches.
        This ensures consistency after pitch shift.
        """
        # Recompute window average from shifted annotations (target pitches)
        new_window_avg = compute_window_avg(annotations, self.window_size)
        note_sequence[:, 4] = new_window_avg
        return note_sequence


def main():
    """Test the augmentation classes."""
    print("Testing HandDataAugmentation...")
    
    # Create dummy 5-feature data
    n = 10
    note_seq = np.zeros((n, 5))
    note_seq[0, 0] = MAX_PIANO_PITCH + 1  # pad
    note_seq[1:, 0] = np.random.randint(MIN_PIANO_PITCH, MAX_PIANO_PITCH, n-1)  # pitches
    note_seq[:, 1] = np.random.rand(n)  # dt
    note_seq[:, 2] = np.random.rand(n)  # dur
    note_seq[:, 3] = np.random.randint(0, 128, n)  # velocity
    note_seq[:, 4] = np.random.randint(0, 2, n)  # hand (0 or 1)
    
    annotations = note_seq[1:, 0].copy()
    annotations = np.concatenate([[note_seq[1, 0]], annotations[:-1]])
    
    print(f"Original pitches: {note_seq[:, 0]}")
    print(f"Original hand: {note_seq[:, 4]}")
    
    aug = HandDataAugmentation(pitch_shift_prob=1.0, tempo_change_prob=1.0)
    aug_seq, aug_annot = aug(note_seq, annotations)
    
    print(f"Augmented pitches: {aug_seq[:, 0]}")
    print(f"Augmented hand (unchanged): {aug_seq[:, 4]}")
    
    print("\nTesting ClusterDataAugmentation...")
    
    # Create dummy 5-feature data with window_avg
    note_seq2 = np.zeros((n, 5))
    note_seq2[0, 0] = MAX_PIANO_PITCH + 1  # pad
    note_seq2[1:, 0] = np.random.randint(MIN_PIANO_PITCH + 12, MAX_PIANO_PITCH - 12, n-1)
    note_seq2[:, 1] = np.random.rand(n)
    note_seq2[:, 2] = np.random.rand(n)
    note_seq2[:, 3] = np.random.randint(0, 128, n)
    
    annotations2 = note_seq2[1:, 0].copy()
    annotations2 = np.concatenate([[note_seq2[1, 0]], annotations2[:-1]])
    note_seq2[:, 4] = compute_window_avg(annotations2)
    
    print(f"Original pitches: {note_seq2[:, 0]}")
    print(f"Original window_avg: {note_seq2[:, 4]}")
    
    aug2 = ClusterDataAugmentation(pitch_shift_prob=1.0, tempo_change_prob=1.0)
    aug_seq2, aug_annot2 = aug2(note_seq2, annotations2)
    
    print(f"Augmented pitches: {aug_seq2[:, 0]}")
    print(f"Augmented window_avg (recalculated): {aug_seq2[:, 4]}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

