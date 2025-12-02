"""
Author: YZQ
Created on: 2025/11/27
Brief: Data preparation for POP909 dataset with HandDataAugmentation
       Processes POP909 MIDI files into train/valid/test pkl files
       
       Three classes available:
       - POP909DataPreparation: 4 features (pitch, dt, dur, vel)
       - POP909HandDataPreparation: 5 features (pitch, dt, dur, vel, hand)
       - POP909WindowDataPreparation: 5 features (pitch, dt, dur, vel, window_avg)

       Reference: https://github.com/lynnzYe/Tap-to-Music/blob/9bf3016803e6038b302265dabbd68414c9a71b0a/ttm/data_preparation/data_preparation.py
"""
import math
import os
import pickle
import random
import sys
from functools import cmp_to_key
from pathlib import Path

import numpy as np
import torch
import pretty_midi
from tqdm import tqdm
from torch.serialization import add_safe_globals

# Import from ttm module
from ttm.config import RD_SEED, MAX_PIANO_PITCH
from ttm.data_preparation.data_augmentation import HandDataAugmentation, UnconditionalDataAugmentation
from ttm.data_preparation.utils import get_note_sequence_from_midi, midi_to_tap, compare_note_order

random.seed(RD_SEED)
np.random.seed(RD_SEED)

# ============================================================================
# Path Setup
# ============================================================================

YZQ_DIR = Path(__file__).resolve().parent
REPO_ROOT = YZQ_DIR.parent
HANNDS_DIR = REPO_ROOT / "hannds"

# ============================================================================
# HANNDs Model Integration for Left/Right Hand Prediction
# ============================================================================

# Ensure hannds code is importable
if str(HANNDS_DIR) not in sys.path:
    sys.path.insert(0, str(HANNDS_DIR))

from network_zoo import Network88

# HANNDs model configuration (must match training)
HIDDEN_SIZE = 70
LAYERS = 2
RNN_TYPE = "LSTM"
BIDIRECTIONAL = False  # Unidirectional = causal prediction
N_FEATURES = 88
N_CATEGORIES = 3  # 0: none, 1: left, 2: right
MS_WINDOW = 20    # 20ms windows

DEFAULT_HANNDS_CHECKPOINT = HANNDS_DIR / "models" / "11-22-2325-network=88(LSTM)_hidden=70_layers=2_cv=1" / "model.pt"

# Window average configuration
DEFAULT_WINDOW_SIZE = 8  # Number of previous notes to average for context


def build_hannds_model(device, checkpoint_path):
    """Load the trained HANNDs model for hand prediction."""
    add_safe_globals([Network88])
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, torch.nn.Module):
        model = state
    else:
        model = Network88(HIDDEN_SIZE, LAYERS, BIDIRECTIONAL, N_FEATURES, N_CATEGORIES, RNN_TYPE)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def causal_filter(predicted_classes, label_not_played=0):
    """
    Apply causal filtering: once a note is assigned to a hand at onset,
    it stays assigned to that hand for its duration.
    This ensures we only use information up to the current time step.
    """
    predicted_classes = predicted_classes.copy()
    for row in range(1, predicted_classes.shape[0]):
        last_line = predicted_classes[row - 1]
        current_line = predicted_classes[row]
        both_note_on = np.logical_and(current_line != label_not_played, last_line != label_not_played)
        predicted_classes[row, both_note_on] = last_line[both_note_on]
    return predicted_classes


def predict_hands_for_midi(model, device, midi_path, apply_causal_filter=True):
    """
    Predict left/right hand labels for each note in a MIDI file.
    
    Args:
        model: Trained HANNDs model
        device: torch device
        midi_path: Path to MIDI file
        apply_causal_filter: Whether to apply causal filtering (recommended for causal prediction)
    
    Returns:
        notes: List of pretty_midi Note objects (sorted by onset, then pitch)
        labels: List of 'L' or 'R' for each note
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [note for inst in pm.instruments for note in inst.notes]
    if not notes:
        raise ValueError(f"No notes found in {midi_path}")
    
    # Sort notes by onset time, then pitch (consistent with midi_to_tap)
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    
    samples_per_sec = 1000 // MS_WINDOW
    end_time = max(note.end for note in notes)
    n_windows = max(1, int(math.ceil(end_time * samples_per_sec)))
    
    # Build input tensor: piano roll representation
    X = np.zeros((n_windows, N_FEATURES), dtype=np.float32)
    for note in notes:
        pitch_idx = note.pitch - 21  # Piano pitch starts at 21
        if pitch_idx < 0 or pitch_idx >= N_FEATURES:
            continue
        start_idx = max(0, int(math.floor(note.start * samples_per_sec)))
        end_idx = max(start_idx + 1, int(math.ceil(note.end * samples_per_sec)))
        end_idx = min(end_idx, n_windows)
        X[start_idx:end_idx, pitch_idx] = 1.0
    
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, _ = model(X_tensor, None)
    
    # output: [1, T, 88, 3]
    logits = output.squeeze(0).cpu()
    preds = torch.argmax(logits, dim=-1).numpy()  # shape [T, 88]
    
    # Apply causal filter to ensure prediction only uses info up to current time
    if apply_causal_filter:
        preds = causal_filter(preds)
    
    # Map each note to its hand label at onset time
    labels = []
    num_steps = preds.shape[0]
    for note in notes:
        pitch_idx = note.pitch - 21
        if pitch_idx < 0 or pitch_idx >= N_FEATURES:
            labels.append('R')  # Default to right hand for out-of-range
            continue
        win = int(math.floor(note.start * samples_per_sec))
        win = min(max(win, 0), num_steps - 1)
        cls = int(preds[win, pitch_idx])
        if cls == 1:
            labels.append('L')
        elif cls == 2:
            labels.append('R')
        else:
            labels.append('R')  # Default to right hand for "none"
    
    return notes, labels


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


# ============================================================================
# POP909DataPreparation: 4 Features (No Hand)
# ============================================================================

class POP909DataPreparation:
    """
    Prepare POP909 dataset for training with 4 features:
        - pitch (previous note pitch)
        - log1p delta time (log1p of inter-onset interval)
        - log1p duration (log1p of effective duration)
        - velocity (previous note velocity)
    
    Output files: pop909-train.pkl, pop909-validation.pkl, pop909-test.pkl
    """

    def __init__(self, pop909_dir, output_dir, train_val_test_split=(0.8, 0.1, 0.1)):
        """
        Args:
            pop909_dir: Path to POP909 dataset directory
            output_dir: Path to save output pkl files
            train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio)
        """
        self.pop909_dir = Path(pop909_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_val_test_split = train_val_test_split
        self.prefix = "pop909"
        
        # Data augmentation for training
        self.data_aug = HandDataAugmentation()

    def collect_midi_files(self):
        """Collect all main MIDI files from POP909 dataset."""
        midi_paths = []
        pop909_root = self.pop909_dir / 'POP909'
        
        if not pop909_root.exists():
            pop909_root = self.pop909_dir
        
        for folder in sorted(pop909_root.iterdir()):
            if not folder.is_dir():
                continue
            if not folder.name.isdigit():
                continue
            
            main_midi = folder / f"{folder.name}.mid"
            if main_midi.exists():
                midi_paths.append(main_midi)
            else:
                for f in folder.iterdir():
                    if f.is_file() and f.suffix.lower() in ['.mid', '.midi']:
                        midi_paths.append(f)
                        break
        
        return midi_paths

    def split_data(self, midi_paths):
        """Split MIDI paths into train/validation/test sets."""
        random.seed(RD_SEED)
        shuffled = midi_paths.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_val_test_split[0])
        n_val = int(n_total * self.train_val_test_split[1])
        
        return {
            'train': shuffled[:n_train],
            'validation': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }

    def process_midi(self, midi_path):
        """
        Process a single MIDI file and extract 4 features.
        
        Returns:
            Tuple of (features, labels) where:
                - features: np.ndarray of shape (N, 4)
                - labels: np.ndarray of shape (N,)
        """
        try:
            note_sequence = get_note_sequence_from_midi(str(midi_path))
            if len(note_sequence) < 2:
                return None
            features, labels = midi_to_tap(note_sequence)
            return features, labels
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    def prepare_features(self):
        """Main method to prepare all features from POP909 dataset."""
        print(f"\n{'='*60}")
        print(f"POP909DataPreparation (4 features)")
        print(f"{'='*60}")
        print("Collecting MIDI files from POP909 dataset...")
        midi_paths = self.collect_midi_files()
        print(f"Found {len(midi_paths)} MIDI files")
        
        if len(midi_paths) == 0:
            raise ValueError(f"No MIDI files found in {self.pop909_dir}")
        
        split_paths = self.split_data(midi_paths)
        
        print(f"Split: train={len(split_paths['train'])}, "
              f"validation={len(split_paths['validation'])}, "
              f"test={len(split_paths['test'])}")
        
        for split_name, paths in split_paths.items():
            print(f"\nProcessing {split_name} split...")
            data = []
            
            for midi_path in tqdm(paths, desc=f"Processing {split_name}"):
                result = self.process_midi(midi_path)
                if result is not None:
                    data.append(result)
            
            output_path = self.output_dir / f"{self.prefix}-{split_name}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Saved {len(data)} samples to {output_path}")
        
        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics after processing."""
        print("\n" + "=" * 60)
        print(f"Dataset Statistics: {self.prefix}")
        print("=" * 60)
        
        for split in ['train', 'validation', 'test']:
            pkl_path = self.output_dir / f"{self.prefix}-{split}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                n_samples = len(data)
                total_notes = sum(len(d[0]) for d in data)
                avg_notes = total_notes / n_samples if n_samples > 0 else 0
                n_features = data[0][0].shape[1] if n_samples > 0 else 0
                
                print(f"\n{split.upper()}:")
                print(f"  Samples: {n_samples}")
                print(f"  Total notes: {total_notes}")
                print(f"  Average notes per sample: {avg_notes:.1f}")
                print(f"  Feature dimensions: {n_features}")


# ============================================================================
# POP909HandDataPreparation: 5 Features (With Hand)
# ============================================================================

class POP909HandDataPreparation:
    """
    Prepare POP909 dataset for training with 5 features including hand prediction:
        - pitch (previous note pitch)
        - log1p delta time (log1p of inter-onset interval)
        - log1p duration (log1p of effective duration)
        - velocity (previous note velocity)
        - hand (0=left, 1=right) - predicted using HANNDs model (causal)
    
    Output files: pop909hand-train.pkl, pop909hand-validation.pkl, pop909hand-test.pkl
    
    Note: Hand prediction uses only information up to the current time step (causal).
    """

    def __init__(self, pop909_dir, output_dir, train_val_test_split=(0.8, 0.1, 0.1),
                 hannds_checkpoint=None):
        """
        Args:
            pop909_dir: Path to POP909 dataset directory
            output_dir: Path to save output pkl files
            train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio)
            hannds_checkpoint: Path to HANNDs model checkpoint (uses default if None)
        """
        self.pop909_dir = Path(pop909_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_val_test_split = train_val_test_split
        self.prefix = "pop909hand"
        
        # Initialize HANNDs model
        if hannds_checkpoint is None:
            hannds_checkpoint = DEFAULT_HANNDS_CHECKPOINT
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading HANNDs model from {hannds_checkpoint}...")
        print(f"Using device: {self.device}")
        self.hannds_model = build_hannds_model(self.device, str(hannds_checkpoint))
        print("HANNDs model loaded successfully.")
        
        # Data augmentation for training
        self.data_aug = HandDataAugmentation()

    def collect_midi_files(self):
        """Collect all main MIDI files from POP909 dataset."""
        midi_paths = []
        pop909_root = self.pop909_dir / 'POP909'
        
        if not pop909_root.exists():
            pop909_root = self.pop909_dir
        
        for folder in sorted(pop909_root.iterdir()):
            if not folder.is_dir():
                continue
            if not folder.name.isdigit():
                continue
            
            main_midi = folder / f"{folder.name}.mid"
            if main_midi.exists():
                midi_paths.append(main_midi)
            else:
                for f in folder.iterdir():
                    if f.is_file() and f.suffix.lower() in ['.mid', '.midi']:
                        midi_paths.append(f)
                        break
        
        return midi_paths

    def split_data(self, midi_paths):
        """Split MIDI paths into train/validation/test sets."""
        random.seed(RD_SEED)
        shuffled = midi_paths.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_val_test_split[0])
        n_val = int(n_total * self.train_val_test_split[1])
        
        return {
            'train': shuffled[:n_train],
            'validation': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }

    def process_midi(self, midi_path):
        """
        Process a single MIDI file and extract 5 features (including hand).
        
        Returns:
            Tuple of (features, labels) where:
                - features: np.ndarray of shape (N, 5)
                - labels: np.ndarray of shape (N,)
        """
        try:
            note_sequence = get_note_sequence_from_midi(str(midi_path))
            if len(note_sequence) < 2:
                return None
            
            features, labels = midi_to_tap(note_sequence)
            
            # Get hand predictions (causal - only uses info up to current time)
            notes, hand_labels = predict_hands_for_midi(
                self.hannds_model, self.device, midi_path, apply_causal_filter=True
            )
            
            # Convert hand labels to numeric (0=left, 1=right)
            hand_feature = np.array([0 if h == 'L' else 1 for h in hand_labels], dtype=float)
            
            # Handle potential length mismatch
            if len(hand_feature) != len(features):
                print(f"Warning: Hand labels length ({len(hand_feature)}) != features length ({len(features)}) for {midi_path}")
                if len(hand_feature) > len(features):
                    hand_feature = hand_feature[:len(features)]
                else:
                    hand_feature = np.pad(hand_feature, (0, len(features) - len(hand_feature)), constant_values=1)
            
            # Shift hand feature: use previous note's hand as feature (consistent with other features)
            # For the first row (padding), use hand of first actual note
            hand_feature_shifted = np.zeros(len(features), dtype=float)
            hand_feature_shifted[0] = hand_feature[0] if len(hand_feature) > 0 else 1
            hand_feature_shifted[1:] = hand_feature[:-1]
            
            # Add hand feature as 5th column
            features = np.column_stack([features, hand_feature_shifted])
            
            return features, labels
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    def prepare_features(self):
        """Main method to prepare all features from POP909 dataset."""
        print(f"\n{'='*60}")
        print(f"POP909HandDataPreparation (5 features with hand)")
        print(f"{'='*60}")
        print("Collecting MIDI files from POP909 dataset...")
        midi_paths = self.collect_midi_files()
        print(f"Found {len(midi_paths)} MIDI files")
        
        if len(midi_paths) == 0:
            raise ValueError(f"No MIDI files found in {self.pop909_dir}")
        
        split_paths = self.split_data(midi_paths)
        
        print(f"Split: train={len(split_paths['train'])}, "
              f"validation={len(split_paths['validation'])}, "
              f"test={len(split_paths['test'])}")
        
        for split_name, paths in split_paths.items():
            print(f"\nProcessing {split_name} split...")
            data = []
            
            for midi_path in tqdm(paths, desc=f"Processing {split_name}"):
                result = self.process_midi(midi_path)
                if result is not None:
                    data.append(result)
            
            output_path = self.output_dir / f"{self.prefix}-{split_name}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Saved {len(data)} samples to {output_path}")
        
        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics after processing."""
        print("\n" + "=" * 60)
        print(f"Dataset Statistics: {self.prefix}")
        print("=" * 60)
        
        for split in ['train', 'validation', 'test']:
            pkl_path = self.output_dir / f"{self.prefix}-{split}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                n_samples = len(data)
                total_notes = sum(len(d[0]) for d in data)
                avg_notes = total_notes / n_samples if n_samples > 0 else 0
                n_features = data[0][0].shape[1] if n_samples > 0 else 0
                
                print(f"\n{split.upper()}:")
                print(f"  Samples: {n_samples}")
                print(f"  Total notes: {total_notes}")
                print(f"  Average notes per sample: {avg_notes:.1f}")
                print(f"  Feature dimensions: {n_features}")
                
                # Show hand distribution
                if n_samples > 0 and n_features >= 5:
                    left_count = sum((d[0][:, 4] == 0).sum() for d in data)
                    right_count = sum((d[0][:, 4] == 1).sum() for d in data)
                    print(f"  Left hand notes: {left_count} ({100*left_count/total_notes:.1f}%)")
                    print(f"  Right hand notes: {right_count} ({100*right_count/total_notes:.1f}%)")


# ============================================================================
# POP909WindowDataPreparation: 5 Features (With Window Average)
# ============================================================================

class POP909WindowDataPreparation:
    """
    Prepare POP909 dataset for training with 5 features including window average:
        - pitch (previous note pitch)
        - log1p delta time (log1p of inter-onset interval)
        - log1p duration (log1p of effective duration)
        - velocity (previous note velocity)
        - window_avg (average pitch of previous window_size notes)
    
    Output files: pop909window-train.pkl, pop909window-validation.pkl, pop909window-test.pkl
    
    Note: Window average is causal - only uses information from previous notes.
    """

    def __init__(self, pop909_dir, output_dir, train_val_test_split=(0.8, 0.1, 0.1),
                 window_size=DEFAULT_WINDOW_SIZE):
        """
        Args:
            pop909_dir: Path to POP909 dataset directory
            output_dir: Path to save output pkl files
            train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio)
            window_size: Number of previous notes to average for context
        """
        self.pop909_dir = Path(pop909_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_val_test_split = train_val_test_split
        self.window_size = window_size
        self.prefix = "pop909window"
        
        print(f"Window size for averaging: {self.window_size}")
        
        # Data augmentation for training
        self.data_aug = HandDataAugmentation()

    def collect_midi_files(self):
        """Collect all main MIDI files from POP909 dataset."""
        midi_paths = []
        pop909_root = self.pop909_dir / 'POP909'
        
        if not pop909_root.exists():
            pop909_root = self.pop909_dir
        
        for folder in sorted(pop909_root.iterdir()):
            if not folder.is_dir():
                continue
            if not folder.name.isdigit():
                continue
            
            main_midi = folder / f"{folder.name}.mid"
            if main_midi.exists():
                midi_paths.append(main_midi)
            else:
                for f in folder.iterdir():
                    if f.is_file() and f.suffix.lower() in ['.mid', '.midi']:
                        midi_paths.append(f)
                        break
        
        return midi_paths

    def split_data(self, midi_paths):
        """Split MIDI paths into train/validation/test sets."""
        random.seed(RD_SEED)
        shuffled = midi_paths.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_val_test_split[0])
        n_val = int(n_total * self.train_val_test_split[1])
        
        return {
            'train': shuffled[:n_train],
            'validation': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }

    def process_midi(self, midi_path):
        """
        Process a single MIDI file and extract 5 features (including window_avg).
        
        Returns:
            Tuple of (features, labels) where:
                - features: np.ndarray of shape (N, 5)
                - labels: np.ndarray of shape (N,)
        """
        try:
            note_sequence = get_note_sequence_from_midi(str(midi_path))
            if len(note_sequence) < 2:
                return None
            
            features, labels = midi_to_tap(note_sequence)
            
            # Compute window average of pitches (causal - only previous notes)
            # labels contains the actual pitches at each position
            window_avg_feature = compute_window_avg(labels, self.window_size)
            
            # Add window_avg feature as 5th column
            features = np.column_stack([features, window_avg_feature])
            
            return features, labels
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    def prepare_features(self):
        """Main method to prepare all features from POP909 dataset."""
        print(f"\n{'='*60}")
        print(f"POP909WindowDataPreparation (5 features with window_avg)")
        print(f"Window size: {self.window_size}")
        print(f"{'='*60}")
        print("Collecting MIDI files from POP909 dataset...")
        midi_paths = self.collect_midi_files()
        print(f"Found {len(midi_paths)} MIDI files")
        
        if len(midi_paths) == 0:
            raise ValueError(f"No MIDI files found in {self.pop909_dir}")
        
        split_paths = self.split_data(midi_paths)
        
        print(f"Split: train={len(split_paths['train'])}, "
              f"validation={len(split_paths['validation'])}, "
              f"test={len(split_paths['test'])}")
        
        for split_name, paths in split_paths.items():
            print(f"\nProcessing {split_name} split...")
            data = []
            
            for midi_path in tqdm(paths, desc=f"Processing {split_name}"):
                result = self.process_midi(midi_path)
                if result is not None:
                    data.append(result)
            
            output_path = self.output_dir / f"{self.prefix}-{split_name}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Saved {len(data)} samples to {output_path}")
        
        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics after processing."""
        print("\n" + "=" * 60)
        print(f"Dataset Statistics: {self.prefix}")
        print("=" * 60)
        
        for split in ['train', 'validation', 'test']:
            pkl_path = self.output_dir / f"{self.prefix}-{split}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                n_samples = len(data)
                total_notes = sum(len(d[0]) for d in data)
                avg_notes = total_notes / n_samples if n_samples > 0 else 0
                n_features = data[0][0].shape[1] if n_samples > 0 else 0
                
                print(f"\n{split.upper()}:")
                print(f"  Samples: {n_samples}")
                print(f"  Total notes: {total_notes}")
                print(f"  Average notes per sample: {avg_notes:.1f}")
                print(f"  Feature dimensions: {n_features}")
                
                # Show window_avg statistics
                if n_samples > 0 and n_features >= 5:
                    all_window_avg = np.concatenate([d[0][:, 4] for d in data])
                    print(f"  Window avg - min: {all_window_avg.min():.1f}, "
                          f"max: {all_window_avg.max():.1f}, "
                          f"mean: {all_window_avg.mean():.1f}")


# ============================================================================
# Main Entry Points
# ============================================================================

def prepare_pop909_basic():
    """Prepare POP909 dataset with 4 features (no hand)."""
    pop909_dir = REPO_ROOT / "POP909-Dataset"
    output_dir = YZQ_DIR / "output"
    
    print(f"POP909 directory: {pop909_dir}")
    print(f"Output directory: {output_dir}")
    
    prep = POP909DataPreparation(
        pop909_dir=pop909_dir,
        output_dir=output_dir,
        train_val_test_split=(0.8, 0.1, 0.1)
    )
    prep.prepare_features()


def prepare_pop909_hand():
    """Prepare POP909 dataset with 5 features (including causal hand prediction)."""
    pop909_dir = REPO_ROOT / "POP909-Dataset"
    output_dir = YZQ_DIR / "output"
    
    print(f"POP909 directory: {pop909_dir}")
    print(f"Output directory: {output_dir}")
    
    prep = POP909HandDataPreparation(
        pop909_dir=pop909_dir,
        output_dir=output_dir,
        train_val_test_split=(0.8, 0.1, 0.1)
    )
    prep.prepare_features()


def prepare_pop909_window(window_size=DEFAULT_WINDOW_SIZE):
    """Prepare POP909 dataset with 5 features (including window average)."""
    pop909_dir = REPO_ROOT / "POP909-Dataset"
    output_dir = YZQ_DIR / "output"
    
    print(f"POP909 directory: {pop909_dir}")
    print(f"Output directory: {output_dir}")
    
    prep = POP909WindowDataPreparation(
        pop909_dir=pop909_dir,
        output_dir=output_dir,
        train_val_test_split=(0.8, 0.1, 0.1),
        window_size=window_size
    )
    prep.prepare_features()


def main():
    """
    Main entry point - prepares datasets:
    1. pop909-*.pkl (4 features)
    2. pop909hand-*.pkl (5 features with hand)
    3. pop909window-*.pkl (5 features with window_avg)
    """
    print("=" * 70)
    print("POP909 Data Preparation")
    print("=" * 70)
    
    # Prepare window-averaged dataset (faster, no model loading)
    prepare_pop909_window()
    
    print("\n" + "=" * 70)
    print("All data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
