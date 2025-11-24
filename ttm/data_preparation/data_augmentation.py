"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: reference - PM2S data augmentation
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.serialization import add_safe_globals

from ttm.config import onset_tolerance, RD_SEED, MIN_PIANO_PITCH, MAX_PIANO_PITCH

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


class RangeDataAugmentation(BaseDataAugmentation):
    """
    Wrap unconditional augmentation but recompute a coarse octave range afterwards.
    The optional fifth column of note_sequence (range id) is updated in place if present.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Reuse unconditional augmentation settings; kwargs forward overrides if needed.
        self.base_aug = UnconditionalDataAugmentation(**kwargs)

    @staticmethod
    def _octave_range(pitches: np.ndarray):
        #TODO: remove outliers, compute more robustly
        valid = pitches[(pitches > 0) & (pitches <= MAX_PIANO_PITCH)]
        if len(valid) == 0:
            return None
        min_oct = int(valid.min()) // 12 - 1
        max_oct = int(valid.max()) // 12 - 1
        return (min_oct, max_oct)

    def __call__(self, note_sequence, annotations, piece_range=None):
        # Apply the standard unconditional augmentations first.
        note_sequence, annotations = self.base_aug(note_sequence, annotations)

        # Recompute octave span from augmented annotations.
        octave_range = self._octave_range(annotations)
        range_val = None
        if octave_range is not None:
            range_val = octave_range[1]  # use max octave as a scalar condition
        elif isinstance(piece_range, tuple):
            range_val = piece_range[1]
        elif piece_range is not None:
            range_val = piece_range
        else:
            range_val = 0

        # If a range column exists (5th column), update it so downstream sees the new condition.
        if note_sequence.shape[1] >= 5:
            note_sequence[:, 4] = range_val

        return note_sequence, annotations, octave_range


class ClusterAugmentation(BaseDataAugmentation):
    """
    Compute hand labels via a HANNDs checkpoint, store a left/right flag, and per-hand medians.
    Columns appended (in order): median (col 4 if present), hand_flag (0=L,1=R), left_median, right_median.
    """

    def __init__(self, checkpoint_path: str = None, device: str = "cpu", **kwargs):
        super().__init__()
        # Mirror UC augmentation to keep tempo/pitch shift consistent
        self.base_aug = UnconditionalDataAugmentation(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        if not self.checkpoint_path:
            return
        hannds_dir = Path(__file__).resolve().parents[2] / "hannds"
        if hannds_dir.exists() and str(hannds_dir) not in sys.path:
            sys.path.append(str(hannds_dir))
        try:
            from hannds.network_zoo import Network88  # noqa: WPS433
        except Exception as exc:
            print(f"[ClusterAugmentation] Failed to import HANNDs model: {exc}")
            return
        add_safe_globals([Network88])
        try:
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(state, torch.nn.Module):
            self.model = state.to(self.device).eval()
        else:
            # Assume default Network88 signature used in training; fall back if mismatch.
            try:
                self.model = Network88(
                    hidden_size=70, n_layers=2, bidirectional=False,
                    n_features=88, n_categories=3, rnn_type="LSTM"
                ).to(self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    self.model.load_state_dict(state["state_dict"])
                else:
                    self.model.load_state_dict(state)
                self.model.eval()
            except Exception as exc:
                print(f"[ClusterAugmentation] Failed to construct/load model: {exc}")
                self.model = None

    def _predict_hands(self, annotations: np.ndarray):
        """
        Predict hand flags (0=L, 1=R) using the HANNDs model on a pitch-only binary sequence.
        """
        self._load_model()
        if self.model is None:
            return np.zeros_like(annotations, dtype=float)

        valid_mask = (annotations >= MIN_PIANO_PITCH) & (annotations <= MAX_PIANO_PITCH) & (annotations != 88)
        T = len(annotations)
        X = np.zeros((1, T, 88), dtype=np.float32)
        for t in range(T):
            if valid_mask[t]:
                p_idx = int(annotations[t] - MIN_PIANO_PITCH)
                X[0, t, p_idx] = 1.0
        x_tensor = torch.tensor(X, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output, _ = self.model(x_tensor, None)
        # output shape: [1, T, 88, 3]; classes: 0=none,1=left,2=right
        preds = torch.argmax(output, dim=-1).cpu().numpy()[0]  # (T,88)
        hand_flags = np.zeros(T, dtype=float)
        for t in range(T):
            if valid_mask[t]:
                cls = int(preds[t, int(annotations[t] - MIN_PIANO_PITCH)])
                hand_flags[t] = 1.0 if cls == 2 else 0.0  # right=1, left/none=0
        return hand_flags

    def __call__(self, note_sequence, annotations):
        note_sequence, annotations = self.base_aug(note_sequence, annotations)
        valid_mask = (annotations > 0) & (annotations <= MAX_PIANO_PITCH)
        median_val = float(np.median(annotations[valid_mask])) if np.any(valid_mask) else 0.0

        hand_flags = self._predict_hands(annotations)
        left_mask = (hand_flags == 0) & valid_mask
        right_mask = (hand_flags == 1) & valid_mask
        left_median = float(np.median(annotations[left_mask])) if np.any(left_mask) else median_val
        right_median = float(np.median(annotations[right_mask])) if np.any(right_mask) else median_val

        if note_sequence.shape[1] >= 5:
            updated = note_sequence.copy()
            updated[:, 4] = median_val
        else:
            updated = np.concatenate([note_sequence, np.full((len(note_sequence), 1), median_val)], axis=1)

        # Ensure hand flag column (col 5)
        if updated.shape[1] >= 6:
            updated[:, 5] = hand_flags
        else:
            updated = np.concatenate([updated, hand_flags.reshape(-1, 1)], axis=1)

        # Ensure left and right medians (cols 6, 7)
        if updated.shape[1] >= 7:
            updated[:, 6] = left_median
        else:
            updated = np.concatenate([updated, np.full((len(updated), 1), left_median)], axis=1)
        if updated.shape[1] >= 8:
            updated[:, 7] = right_median
        else:
            updated = np.concatenate([updated, np.full((len(updated), 1), right_median)], axis=1)

        return updated, annotations, (median_val, left_median, right_median)


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
