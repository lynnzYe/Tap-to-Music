"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: reference - PM2S data augmentation
"""
import random

import numpy as np

from ttm.config import onset_tolerance, RD_SEED, MIN_PIANO_PITCH, MAX_PIANO_PITCH
from ttm.data_preparation.utils import ChordConstants

random.seed(RD_SEED)
np.random.seed(RD_SEED)


class BaseDataAugmentation:
    def __init__(self):
        pass

    def __call__(self, note_sequence, annotations):
        raise NotImplementedError


class UnconditionalDataAugmentation(BaseDataAugmentation):
    def __init__(self, tempo_change_prob=1.0, tempo_change_range=(0.8, 1.2), pitch_shift_prob=1.0,
                 pitch_shift_range=(-12, 12), missing_note_prob=0.2, perturb_onset_prob=0.3):
        super().__init__()
        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.missing_note_prob = missing_note_prob
        self.perturb_onset_prob = perturb_onset_prob

    def __call__(self, note_sequence, annotations):
        note_sequence, annotations = self.tempo_change(note_sequence, annotations)
        note_sequence, annotations = self.pitch_shift(note_sequence, annotations)
        # note_sequence, annotations = self.missing_note(note_sequence, annotations)
        # note_sequence, annotations = self.perturb_onset(note_sequence, annotations)
        return note_sequence, annotations

    def tempo_change(self, note_sequence, annotations):
        if random.random() > self.tempo_change_prob:
            return note_sequence, annotations
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        note_sequence[:, 1:3] *= 1 / tempo_change_ratio

        # If annotations have timestamps, scale accordingly:
        return note_sequence, annotations

    def pitch_shift(self, note_sequence, annotations):
        if random.random() > self.pitch_shift_prob:
            return note_sequence, annotations

        assert int(note_sequence[0, 0]) == MAX_PIANO_PITCH + 1, \
            'see comment below this assertion'  # always a pad input tuple at the start of input seq.
        """
        During data preparation, I padded one input before all note sequences so the model can start from nothing.
        I can also skip this step and leave it to Dataset.__getitem__, 
        but then the first ground truth label will be lost forever. 
            see midi_to_tap:
        """

        pitches = note_sequence[1:, 0]  # skip first pad pitch
        annot_pitches = annotations
        all_pitches = np.concatenate([pitches, annot_pitches], axis=0)

        max_down = MIN_PIANO_PITCH - all_pitches.min().item()  # negative number
        max_up = MAX_PIANO_PITCH - all_pitches.max().item()  # positive number

        # Clamp within allowed shift range
        lower = max(self.pitch_shift_range[0], max_down)
        upper = min(self.pitch_shift_range[1], max_up)
        shift = np.random.randint(lower, upper + 1)
        note_sequence[1:, 0] += shift
        annotations += shift

        assert np.min(note_sequence[1:, 0]) >= MIN_PIANO_PITCH and np.max(note_sequence[1:, 0]) <= MAX_PIANO_PITCH
        assert np.min(annotations) >= MIN_PIANO_PITCH and np.max(annotations) <= MAX_PIANO_PITCH
        return note_sequence, annotations

    def missing_note(self, note_sequence, annotations):
        """
        Need to consider note importance - randomly dropping notes could sound terrible
        :param note_sequence:
        :param annotations:
        :return:
        """
        raise Exception("some problems see comment")
        missing = random.random()
        if missing < 1. - self.missing_note_prob:
            return note_sequence, annotations

        # Ignore first pitch (pad token)
        actual_notes = note_sequence[1:]
        actual_annots = annotations[1:]

        # Find concurrent notes (small onset gaps)
        candidates = np.diff(actual_notes[:, 1]) < onset_tolerance

        if not np.any(candidates):
            return note_sequence, annotations

        # randomly select a ratio of candidates to be removed
        ratio = random.random()
        candidates_probs = np.random.random(len(candidates))
        keep_mask = np.concatenate([[True], candidates_probs < (1 - ratio)])

        # Apply mask
        actual_notes = actual_notes[keep_mask]
        actual_annots = actual_annots[keep_mask]

        # Reattach the first (pad) note
        note_sequence = np.concatenate([note_sequence[:1], actual_notes], axis=0)
        annotations = np.concatenate([annotations[:1], actual_annots], axis=0)

        return note_sequence, annotations

    def perturb_onset(self, note_sequence, annotations, epsilon=onset_tolerance / 8.0):
        """
        Need to first revert back to onset domain, then log1p. can introduce noise
        :param note_sequence:
        :param annotations:
        :param epsilon:
        :return:
        """
        raise Exception("some problems see comment")
        # Separate first (pad) row and the rest
        pad_row = note_sequence[:1]
        pad_annot = annotations[:1]

        notes = note_sequence[1:].copy()
        annots = annotations[1:].copy()

        onsets = notes[:, 1]
        N = len(onsets)

        # Which notes to perturb
        mask = np.random.rand(N) < self.perturb_onset_prob
        perturb = np.random.uniform(-epsilon, epsilon, size=N)
        onsets[mask] += perturb[mask]

        # Re-sort within the non-pad region
        sorted_indices = np.argsort(onsets, kind='stable')
        notes = notes[sorted_indices]
        annots = annots[sorted_indices]
        notes[:, 1] = onsets[sorted_indices]

        # Reattach the pad
        note_sequence = np.concatenate([pad_row, notes], axis=0)
        annotations = np.concatenate([pad_annot, annots], axis=0)

        return note_sequence, annotations


class ChordDataAugmentation(BaseDataAugmentation):
    """
    Data augmentation for chord-conditioned data.

    - [prev_pitch, log_dt, log_eff_dur, prev_vel, chord_id]

    - Tempo change: same as unconditional.
    - Pitch shift: same as unconditional, also transpose chord_id.

    Chord ID semantics are the same as in ChordFeatureExtractor
    """
    def __init__(self, tempo_change_prob=1.0, tempo_change_range=(0.8, 1.2),
                 pitch_shift_prob=1.0, pitch_shift_range=(-12, 12)):
        super().__init__()
        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range


    def __call__(self, note_sequence, annotations):
        note_sequence, annotations = self.tempo_change(note_sequence, annotations)
        note_sequence, annotations = self.pitch_shift(note_sequence, annotations)
        return note_sequence, annotations

    def tempo_change(self, note_sequence, annotations):
        if random.random() > self.tempo_change_prob:
            return note_sequence, annotations
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        
        note_sequence[:, 1:3] *= 1 / tempo_change_ratio
        return note_sequence, annotations

    def _transpose_chord_id(self, chord_id: int, shift: int) -> int:
        """
        chord_id = qual_id * 12 + root_pc → transpose root_pc, keep quality.
        """
        if chord_id == ChordConstants.N_ID:
            return ChordConstants.N_ID
        qual_id, root_pc = divmod(chord_id, ChordConstants.NUM_ROOTS)
        root_pc = (root_pc + shift) % ChordConstants.NUM_ROOTS
        return qual_id * ChordConstants.NUM_ROOTS + root_pc
    def pitch_shift(self, note_sequence, annotations):
        if random.random() > self.pitch_shift_prob:
            return note_sequence, annotations

        assert int(note_sequence[0, 0]) == MAX_PIANO_PITCH + 1, \
            'expected a pad input tuple at the start of input seq.'

        pitches = note_sequence[1:, 0] 
        annot_pitches = annotations
        all_pitches = np.concatenate([pitches, annot_pitches], axis=0)

        max_down = MIN_PIANO_PITCH - all_pitches.min().item()  # negative number
        max_up = MAX_PIANO_PITCH - all_pitches.max().item()    # positive number

        lower = max(self.pitch_shift_range[0], max_down)
        upper = min(self.pitch_shift_range[1], max_up)
        if lower > upper:
            return note_sequence, annotations

        shift = np.random.randint(lower, upper + 1)

        note_sequence[1:, 0] += shift
        annotations += shift

        assert np.min(note_sequence[1:, 0]) >= MIN_PIANO_PITCH and np.max(note_sequence[1:, 0]) <= MAX_PIANO_PITCH
        assert np.min(annotations) >= MIN_PIANO_PITCH and np.max(annotations) <= MAX_PIANO_PITCH

        if note_sequence.shape[1] > 4:
            chord_ids = note_sequence[1:, 4].astype(int)
            for i in range(len(chord_ids)):
                chord_ids[i] = self._transpose_chord_id(int(chord_ids[i]), shift)
            note_sequence[1:, 4] = chord_ids.astype(note_sequence.dtype)

        return note_sequence, annotations


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
