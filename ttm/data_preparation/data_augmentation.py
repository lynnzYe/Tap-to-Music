"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: reference - PM2S data augmentation
"""
import random

import numpy as np

from ttm.config import onset_tolerance, RD_SEED, MIN_PIANO_PITCH, MAX_PIANO_PITCH
from ttm.utils import clog

random.seed(RD_SEED)


class BaseDataAugmentation:
    def __init__(self):
        pass

    def __call__(self, note_sequence, annotations):
        raise NotImplementedError


class UnconditionalDataAugmentation(BaseDataAugmentation):
    def __init__(self, tempo_change_prob=1.0, tempo_change_range=(0.8, 1.2), pitch_shift_prob=1.0,
                 pitch_shift_range=(-12, 12), extra_note_prob=0.5, missing_note_prob=0.5, perturb_onset_prob=0.3):
        super().__init__()
        if extra_note_prob + missing_note_prob > 1.:
            extra_note_prob, missing_note_prob = extra_note_prob / (extra_note_prob + missing_note_prob), \
                                                 missing_note_prob / (extra_note_prob + missing_note_prob)
            clog.info('Reset extra_note_prob and missing_note_prob to', extra_note_prob, missing_note_prob)

        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.extra_note_prob = extra_note_prob
        self.missing_note_prob = missing_note_prob
        self.perturb_onset_prob = perturb_onset_prob

    def __call__(self, note_sequence, annotations):
        note_sequence, annotations = self.tempo_change(note_sequence, annotations)
        note_sequence, annotations = self.pitch_shift(note_sequence, annotations)
        note_sequence, annotations = self.missing_note(note_sequence, annotations)
        note_sequence, annotations = self.perturb_onset(note_sequence, annotations)
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
        assert note_sequence[0:0] == MAX_PIANO_PITCH
        pitches = note_sequence[1:, 0]  # skip first pad pitch
        annot_pitches = annotations[:, 0]
        all_pitches = np.concatenate([pitches, annot_pitches], axis=0)

        max_down = MIN_PIANO_PITCH - all_pitches.min().item()  # negative number
        max_up = MAX_PIANO_PITCH - all_pitches.max().item()  # positive number

        # Clamp within allowed shift range
        lower = max(self.pitch_shift_range[0], max_down)
        upper = min(self.pitch_shift_range[1], max_up)
        shift = np.random.randint(lower, upper + 1)
        note_sequence[:, 0] += shift
        annotations[:, 0] += shift

        assert np.min(note_sequence[:, 0]) >= MIN_PIANO_PITCH and np.max(note_sequence[:, 0]) <= MAX_PIANO_PITCH
        assert np.min(annotations[:, 0]) >= MIN_PIANO_PITCH and np.max(annotations[:, 0]) <= MAX_PIANO_PITCH
        return note_sequence, annotations

    def missing_note(self, note_sequence, annotations):
        extra_or_missing = random.random()
        if extra_or_missing < 1. - self.missing_note_prob:
            return note_sequence, annotations

        # find successing concurrent notes
        candidates = np.diff(note_sequence[:, 1]) < onset_tolerance

        # randomly select a ratio of candidates to be removed
        ratio = random.random()
        candidates_probs = candidates * np.random.random(len(candidates))
        remaining = np.concatenate([np.array([True]), candidates_probs < (1 - ratio)])

        # remove selected candidates
        note_sequence = note_sequence[remaining]
        annotations = annotations[remaining]

        return note_sequence, annotations

    def perturb_onset(self, note_sequence, annotations, epsilon=onset_tolerance / 8.0):
        # Perturb absolute time randomly with epsilon values
        onsets = note_sequence[:, 1].copy()
        N = len(onsets)
        # Decide which notes to perturb
        mask = np.random.rand(N) < self.perturb_onset_prob
        perturb = np.random.uniform(-epsilon, epsilon, size=N)

        # Apply perturbation only to selected notes
        onsets[mask] += perturb[mask]
        sorted_indices = np.argsort(onsets, kind='stable')
        note_sequence = note_sequence[sorted_indices]
        annotations = annotations[sorted_indices]
        note_sequence[:, 1] = onsets[sorted_indices]

        return note_sequence, annotations


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
