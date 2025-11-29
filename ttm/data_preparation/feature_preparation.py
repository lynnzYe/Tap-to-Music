"""
Author: Lynn Ye
Created on: 2025/11/12
Brief: 
"""

import json
import os
import pickle
import random
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi as pm
from tqdm import tqdm

from ttm.config import RD_SEED, dotenv_config
from ttm.data_preparation.utils import find_duplicate_midi, get_note_sequence_from_midi, midi_to_tap, ChordConstants
from ttm.utils import clog

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

random.seed(RD_SEED)
unconditional_datacols = ['source', 'composer', 'piece_name', 'midi_path', 'split', 'annot_file']


def parse_any_data(data_dir, source_name, train_val_test_split=(0.8, 0.1, 0.1)):
    """Recursively collect all MIDI file paths and split into train/val/test,
    skipping folders that contain 'versions' in their name (case-insensitive)."""

    midi_paths = []
    for root, _, files in os.walk(data_dir):
        # Skip directories with 'versions' in their path
        if 'versions' in os.path.basename(root).lower():
            continue

        for f in files:
            if 'midi_score' in f: continue
            if f.lower().endswith(('.mid', '.midi')):
                midi_paths.append(os.path.join(root, f))

    if not midi_paths:
        clog.warning(f"No MIDI files found in {data_dir}")
        return pd.DataFrame(columns=["midi_path", "split", "source"])

    # Shuffle for reproducible split
    random.seed(RD_SEED)
    random.shuffle(midi_paths)

    n_total = len(midi_paths)
    n_train = int(n_total * train_val_test_split[0])
    n_val = int(n_total * train_val_test_split[1])

    splits = (
            ["train"] * n_train +
            ["validation"] * n_val +
            ["test"] * (n_total - n_train - n_val)
    )

    df = pd.DataFrame({
        "midi_path": midi_paths,
        "split": splits,
        "source": source_name
    })

    return df


def parse_maestro_data(maestro_dir):
    maestro_dir = Path(maestro_dir)
    meta_path = maestro_dir / 'maestro-v3.0.0.json'
    with open(meta_path, 'r') as f:
        maestro_metadata = json.load(f)

    # Convert metadata to DataFrame
    df = pd.DataFrame(maestro_metadata)

    # Build absolute MIDI paths
    df['midi_path'] = df['midi_filename'].apply(lambda x: str(maestro_dir / x))

    # Clean up piece names — remove "(Complete)" or "(complete)"
    def normalize_title(title):
        return re.sub(r'\s*\(complete\).*$', '', title, flags=re.IGNORECASE).strip()

    df['piece_name'] = df['canonical_title'].apply(normalize_title)
    df['composer'] = df['canonical_composer']
    df['split'] = df['split']
    df['duration'] = df['duration']

    # Keep only relevant columns
    df = df[['split', 'midi_path', 'composer', 'piece_name', 'duration']]

    # Sort for readability
    df = df.sort_values(by=['composer', 'piece_name']).reset_index(drop=True)

    return df


def parse_asap_data(asap_dir):
    """
    Returns: pd.DataFrame
        columns ['composer', 'piece_name', 'midi_path', 'maestro_midi_performance']
    """
    asap_dir = Path(asap_dir)
    csv_path = asap_dir / 'metadata.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path, sep='\t') if '\t' in open(csv_path).readline() else pd.read_csv(csv_path)

    # Select and rename relevant columns
    keep_cols = {
        'composer': 'composer',
        'folder': 'piece_name',
        'midi_performance': 'midi_path',
        'maestro_midi_performance': 'maestro_midi_performance'
    }

    # Check that all columns exist
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in ASAP metadata: {missing}")

    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    # Append full path to midi_path
    df['midi_path'] = df['midi_path'].apply(lambda x: str(asap_dir / x) if pd.notna(x) else None)

    # Sort for readability
    df = df.sort_values(by=['composer', 'piece_name']).reset_index(drop=True)

    return df


def parse_pop909_dataset(data_dir, train_val_test_split=(0.8, 0.1, 0.1)):
    data_dir = Path(data_dir)

    if (data_dir / "POP909").exists():
        pop_root = data_dir / "POP909"
    else:
        pop_root = data_dir

    rows = []

    for idx_dir in sorted(pop_root.iterdir()):
        if not idx_dir.is_dir():
            continue

        midi_candidates = list(idx_dir.glob("*.mid")) + list(idx_dir.glob("*.midi"))
        if not midi_candidates:
            continue

        midi_path = str(midi_candidates[0])

        # Find chord annotation
        chord_path = None

        if (idx_dir / "chord_midi.txt").exists():
            chord_path = str(idx_dir / "chord_midi.txt")
        elif (idx_dir / "chord_audio" / "beat_audio.txt").exists():
            chord_path = str(idx_dir / "chord_audio" / "beat_audio.txt")

        if chord_path is None:
            clog.warning(f"No chord annotation found for {idx_dir}")
            continue

        piece_name = idx_dir.name
        composer = "POP909"         # POP909 do not have composer

        rows.append({
            "source": "pop909",
            "composer": composer,
            "piece_name": piece_name,
            "midi_path": midi_path,
            "split": None,          # filled below
            "annot_file": chord_path
        })

    if not rows:
        clog.warning(f"No POP909 MIDI files with chord annotations found in {data_dir}")
        return pd.DataFrame(columns=unconditional_datacols)

    # Train split (same logic as parse_any_data)
    random.seed(RD_SEED)
    random.shuffle(rows)

    n_total = len(rows)
    n_train = int(n_total * train_val_test_split[0])
    n_val = int(n_total * train_val_test_split[1])

    splits = (
        ["train"] * n_train +
        ["validation"] * n_val +
        ["test"] * (n_total - n_train - n_val)
    )

    for r, s in zip(rows, splits):
        r["split"] = s

    df = pd.DataFrame(rows, columns=unconditional_datacols)
    return df


def assign_remaining_splits(asap_data, target_ratios=(0.8, 0.1, 0.1)):
    """
    Rebalance and assign splits to ASAP rows that currently have split='none',
    keeping the total split distribution close to target_ratios.
    """
    # Count current splits
    total_count = len(asap_data)
    assigned_counts = asap_data['split'].value_counts().reindex(['train', 'validation', 'test', 'none'], fill_value=0)

    # Compute how many we ideally want
    target_counts = np.array(target_ratios) * total_count
    current_counts = np.array([assigned_counts['train'], assigned_counts['validation'], assigned_counts['test']])

    # Compute how many we still need for each split
    remaining_needed = np.maximum(np.round(target_counts - current_counts), 0).astype(int)
    remaining_needed = remaining_needed / remaining_needed.sum()  # normalize to 1
    remaining_ratios = remaining_needed.tolist()

    # Subset of unassigned
    mask = asap_data['split'] == 'none'
    unassigned = asap_data[mask].copy()

    # Shuffle for randomness
    unassigned = unassigned.sample(frac=1, random_state=RD_SEED).reset_index(drop=True)

    # Assign according to remaining_ratios
    n = len(unassigned)
    n_train = int(remaining_ratios[0] * n)
    n_val = int(remaining_ratios[1] * n)

    unassigned.loc[:n_train, 'split'] = 'train'
    unassigned.loc[n_train:n_train + n_val, 'split'] = 'validation'
    unassigned.loc[n_train + n_val:, 'split'] = 'test'

    # Merge back
    asap_data.loc[mask, 'split'] = unassigned['split'].values
    return asap_data


def integrate_maestro_asap(mstro, asap, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
    """
       Integrate MAESTRO and ASAP metadata with consistent splits.

       Steps:
       1. Find ASAP performances that reference MAESTRO performances (via maestro_midi_performance paths).
       2. Match these to MAESTRO split labels.
       3. If a piece in MAESTRO is split train/val/test, assign the same split to ASAP's piece.
       4. Remove duplicate ASAP entries with identical maestro_midi_performance.
       5. Assign random splits to ASAP entries not linked to any MAESTRO piece.

       Args:
           maestro_data (pd.DataFrame): Parsed MAESTRO metadata (columns: ['split','midi_path','composer','piece_name','duration'])
           asap_data (pd.DataFrame): Parsed ASAP metadata (columns: ['composer','piece_name','midi_path','maestro_midi_performance'])
           train_val_test_split (tuple): e.g., (0.8, 0.1, 0.1)
           seed (int): random seed for reproducibility
           check_duplicate: compare whether there are overlapping MIDI performance by hashing
                            !this is very time-consuming!

       Returns:
           pd.DataFrame: Combined ASAP data with new column 'split'
       """

    def normalize_path(p):
        if pd.isna(p):
            return None
        return str(Path(p)).replace("\\", "/").split("{maestro}/")[-1]  # relative to /maestro/

    maestro_dir_name = 'maestro-v3.0.0'
    assert f'/{maestro_dir_name}/' in mstro['midi_path'][0]
    asap["maestro_midi_performance_norm"] = asap["maestro_midi_performance"].apply(normalize_path)
    mstro["midi_path_norm"] = mstro["midi_path"].apply(
        lambda x: str(Path(x)).replace("\\", "/").split(f'/{maestro_dir_name}/')[-1])

    # Create mapping from MAESTRO midi_path_norm → (piece_name, split)
    maestro_map = (
        mstro[["midi_path_norm", "piece_name", "split"]]
        .drop_duplicates(subset="midi_path_norm")
        .set_index("midi_path_norm")
        .to_dict(orient="index")
    )

    # identify ASAP Maestro overlap
    asap_piece_split_map = {}
    for _, row in asap.iterrows():
        norm_path = row["maestro_midi_performance_norm"]
        if norm_path in maestro_map:
            m_split = maestro_map[norm_path]["split"]
            asap_piece_name = row["piece_name"]
            asap_piece_split_map[asap_piece_name] = m_split

    # Assign splits
    splits = []
    for _, row in asap.iterrows():
        norm_path = row["maestro_midi_performance_norm"]
        if norm_path in maestro_map:
            # Direct match
            splits.append(maestro_map[norm_path]["split"])
        elif row["piece_name"] in asap_piece_split_map:
            # Piece-level propagation (only if that piece overlaps)
            splits.append(asap_piece_split_map[row["piece_name"]])
        else:
            # No match found
            splits.append("none")
    asap["split"] = splits

    # drop duplicates
    mstro_paths = set(mstro['midi_path_norm'].dropna())
    asap = asap[~asap['maestro_midi_performance_norm'].isin(mstro_paths)].reset_index(drop=True)

    # train val test split on the rest of ASAP data
    asap = assign_remaining_splits(asap, target_ratios=train_val_test_split)

    # Finally, merge the two dataset
    mstro_sel = mstro.rename(columns={'midi_path': 'midi_path',
                                      'piece_name': 'piece_name',
                                      'composer': 'composer',
                                      'split': 'split'})  # already aligned
    asap_sel = asap.rename(columns={'midi_path': 'midi_path',
                                    'piece_name': 'piece_name',
                                    'composer': 'composer',
                                    'split': 'split'})
    mstro_sel['source'] = 'maestro'
    asap_sel['source'] = 'asap'

    mstro_sel = mstro_sel.reindex(columns=unconditional_datacols)
    asap_sel = asap_sel.reindex(columns=unconditional_datacols)
    # Keep only necessary columns
    combined_df = pd.concat([mstro_sel[unconditional_datacols], asap_sel[unconditional_datacols]], ignore_index=True)

    if check_duplicate:
        clog.info("Checking duplicates")
        overlaps = find_duplicate_midi(combined_df[combined_df['source'] == 'maestro']['midi_path'].tolist(),
                                       combined_df[combined_df['source'] == 'asap']['midi_path'].tolist())
        if len(overlaps) > 0:
            clog.warn("found", len(overlaps), 'overlapped performances in asap and maestro')
        overlapped_paths = set([f[0] for f in overlaps])
        combined_df = combined_df[~combined_df['midi_path'].isin(overlapped_paths)].reset_index(drop=True)

    return combined_df



class MIDIData:
    # Integrate different datasets into one
    def __init__(self, data_dir_dict: dict, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
        """
        :param data_dir_dict: {source_name: path}
        :param train_val_test_split:
        :param check_duplicate:
        """
        for p in data_dir_dict.values():
            assert os.path.exists(p)
        self.data_dir_dict = data_dir_dict
        self.data = None
        self.check_duplicate = check_duplicate
        self.train_val_test_split = train_val_test_split

    def _collect_data(self):
        raise NotImplementedError


class UnconditionalMIDIData(MIDIData):
    def __init__(self, data_dir_dict: dict, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=True):
        super().__init__(data_dir_dict, train_val_test_split, check_duplicate)
        self.data = self._collect_data()

    def _collect_data(self):
        # Init dataframe
        data = pd.DataFrame(columns=unconditional_datacols)

        maestro_asap = [v for k, v in self.data_dir_dict.items() if k.lower() in ['maestro', 'asap']]
        other_datasets = [k for k in self.data_dir_dict.keys() if k.lower() not in ['maestro', 'asap']]
        assert len(maestro_asap) == 0 or len(maestro_asap) == 2
        if maestro_asap:
            mstro_dir = maestro_asap[0] if 'maestro' in maestro_asap[0].lower() else maestro_asap[1]
            asap_dir = maestro_asap[0] if 'asap' in maestro_asap[0].lower() else maestro_asap[1]
            mstro_data = parse_maestro_data(mstro_dir)
            asap_data = parse_asap_data(asap_dir)
            integrated_dataset = integrate_maestro_asap(mstro_data, asap_data, self.train_val_test_split,
                                                        self.check_duplicate)
            data = pd.concat([data, integrated_dataset[unconditional_datacols]], ignore_index=True)
        for source in other_datasets:
            sub_data = parse_any_data(self.data_dir_dict[source], source, self.train_val_test_split)
            sub_data = sub_data.reindex(columns=unconditional_datacols)
            data = pd.concat([data, sub_data[unconditional_datacols]], ignore_index=True)
        data['pid'] = range(len(data))
        return data


class ChordMIDIData(MIDIData):
    def __init__(self, data_dir_dict, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
        super().__init__(data_dir_dict, train_val_test_split, check_duplicate)
        self.data = self._collect_data()

    def _collect_data(self):
        # Build chord-conditioned only using POP909.

        data = pd.DataFrame(columns=unconditional_datacols)

        pop_keys = [k for k in self.data_dir_dict.keys() if k.lower() == "pop909"]
        if not pop_keys:
            raise ValueError("ChordMIDIData requires a 'pop909' entry in data_dir_dict")

        pop_key = pop_keys[0]
        pop_dir = self.data_dir_dict[pop_key]

        pop_data = parse_pop909_dataset(pop_dir, self.train_val_test_split)

        pop_data = pop_data.reindex(columns=unconditional_datacols)
        data = pd.concat([data, pop_data], ignore_index=True)

        data["pid"] = range(len(data))
        return data


class ScoreMIDIData(MIDIData):
    def __init__(self, data_dir_dict, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
        super().__init__(data_dir_dict, train_val_test_split, check_duplicate)
        self.data = self._collect_data()

    def _collect_data(self):
        raise NotImplementedError


class BaseFeatureExtractor:
    def __init__(self):
        pass

    def __call__(self, row_data) -> (np.ndarray, np.ndarray):
        """
        Return features, labels

        :param row_data: pd Dataframe row entry. Search `unconditional_datacols` for col names
        :return:
        """
        raise NotImplementedError


class UnconditionalFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()

    def __call__(self, row_data):
        note_sequence = get_note_sequence_from_midi(row_data['midi_path'])
        return midi_to_tap(note_sequence)

class ChordFeatureExtractor(BaseFeatureExtractor):
    """
    - Chord representation:
        * Parse chord label into (root_pc, quality_id)
            root_pc: 0-11, C=0, C#=1, ..., B=11
            quality_id: fixed vocabulary order:
                ["maj", "min", "dim", "aug", "sus", "maj7", "min7", "7", "other"]
        * Structured integer chord_id:
                chord_id = quality_id * 12 + root_pc
          Special "no chord" (N) ID:
                N_ID = NUM_ROOTS * NUM_QUALITIES

    - Returned features:
        features_with_chord.shape = (N, 5)
    """

    def __init__(self):
        super().__init__()


    def _normalize_quality(self, raw_qual: str) -> str:
        if raw_qual is None or raw_qual == "":
            return "maj"

        # Drop inversion like 'maj7/5'
        raw = raw_qual.split("/")[0].lower()

        if raw in ("maj",):
            return "maj"
        if raw in ("min", "m"):
            return "min"
        if "dim" in raw:
            return "dim"
        if "aug" in raw:
            return "aug"
        if raw.startswith("sus"):
            return "sus"
        if "maj7" in raw or "maj9" in raw:
            return "maj7"
        if raw in ("min7", "m7"):
            return "min7"
        if raw.endswith("7") or raw in ("9", "11", "13"):
            # not sure here
            return "7"
        return "other"

    def _parse_chord_label(self, chord_label: str) -> int:
        if chord_label is None:
            return ChordConstants.N_ID

        chord_label = chord_label.strip()
        if chord_label == "" or chord_label.upper() == "N":
            return ChordConstants.N_ID
        if ":" in chord_label:
            root_str, qual_str = chord_label.split(":", 1)
        else:
            # If no colon, assume major
            root_str, qual_str = chord_label, "maj"

        root_str = root_str.strip()
        qual_str = qual_str.strip()

        root_pc = ChordConstants.NOTE_TO_PC.get(root_str)
        if root_pc is None:
            return ChordConstants.N_ID

        qual_norm = self._normalize_quality(qual_str)
        qual_id = ChordConstants.QUALITY_TO_ID[qual_norm]

        chord_id = qual_id * ChordConstants.NUM_ROOTS + root_pc
        return chord_id

    def _load_chord_segments(self, chord_path: str):
        """
        Each line: start_time end_time chord_name
        """
        segments = []
        with open(chord_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                except ValueError:
                    continue
                chord_name = parts[2]
                segments.append((start, end, chord_name))

        segments.sort(key=lambda x: x[0])
        return segments

    def _align_chords_to_onsets(self, onsets: np.ndarray, segments):
        """
        For each note onset t, find which chord segment [start, end)
        it falls into, and return the corresponding chord_id.
        If no segment covers t, assign N_ID.
        """
        n = len(onsets)
        chord_ids = np.full(n, ChordConstants.N_ID, dtype=np.int32)

        if not segments:
            return chord_ids

        seg_idx = 0
        n_seg = len(segments)

        for i, t in enumerate(onsets):

            while seg_idx < n_seg - 1 and t >= segments[seg_idx][1]:
                seg_idx += 1

            start, end, label = segments[seg_idx]
            if start <= t < end:
                chord_ids[i] = self._parse_chord_label(label)
            else:
                chord_ids[i] = ChordConstants.N_ID

        return chord_ids


    def __call__(self, row_data):
        """
        Returns:
            features_with_chord: (N, 5) = [tap_features(4 dims), chord_id]
            labels:              (N,)   = pitch targets
        """

        notes = get_note_sequence_from_midi(row_data["midi_path"])
        features, labels = midi_to_tap(notes)
        onsets = notes[:, 1]   # onset time in seconds

        chord_path = row_data.get("annot_file", None)
        if isinstance(chord_path, str) and os.path.exists(chord_path):
            segments = self._load_chord_segments(chord_path)
            chord_ids = self._align_chords_to_onsets(onsets, segments)
        else:
            clog.warning(f"No chord annotation file found for MIDI: {row_data.get('midi_path', 'unknown')}")
            chord_ids = np.full(len(onsets), ChordConstants.N_ID, dtype=np.int32)

        # Append chord_id as a new feature column
        chord_ids = chord_ids.astype(float).reshape(-1, 1)
        features_with_chord = np.concatenate([features, chord_ids], axis=1)

        return features_with_chord, labels

data_class_map = {
    'unconditional': UnconditionalMIDIData,
    'chord': ChordMIDIData
}

feature_extractor_map = {
    'unconditional': UnconditionalFeatureExtractor(),
    'chord': ChordFeatureExtractor()
}


class FeaturePreparation:
    def __init__(self, feature, save_dir, data_dir_dict, train_val_test_split=(0.8, 0.1, 0.1), workers=1,
                 check_duplicate=False):
        assert feature in data_class_map.keys()
        self.check_duplicate = True
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.train_val_test_split = train_val_test_split
        self.data_dir_dict = data_dir_dict
        self.feature = feature
        self.workers = workers
        self.metadata = None
        self.check_duplicate = check_duplicate

    def extract_meta(self):
        meta_dataset = data_class_map[self.feature](self.data_dir_dict, self.train_val_test_split,
                                                    check_duplicate=self.check_duplicate)
        meta_dataset.data.to_csv(self.save_dir / f'{self.feature}-meta.csv', index=False)

    def load_meta(self):
        self.metadata = pd.read_csv(self.save_dir / f'{self.feature}-meta.csv')

    def print_statistics(self):
        # Reference: PM2S github
        clog.info('Printing dataset statistics')

        # =========== number of performances ===========
        clog.debug('Get number of performances')

        n_perfms_train = len(self.metadata[self.metadata['split'] == 'train'])
        n_perfms_valid = len(self.metadata[self.metadata['split'] == 'validation'])
        n_perfms_test = len(self.metadata[self.metadata['split'] == 'test'])
        n_perfms = n_perfms_train + n_perfms_valid + n_perfms_test

        # ========== duration & number of notes ==========
        clog.debug('Get duration & number of notes')

        flawed_rows_pid = []

        def cache_duration_n_notes(row):
            midi_data = pm.PrettyMIDI(row['midi_path'])
            if len(midi_data.instruments) == 0:
                clog.error("Empty midi path", row['midi_path'])
                flawed_rows_pid.append(row['pid'])
            duration = midi_data.get_end_time()
            n_notes = np.sum([len(midi_data.instruments[i].notes) for i in range(len(midi_data.instruments))])

            cache_file = Path(self.save_dir, 'temp', str(row['pid']) + '.pkl')
            pickle.dump((duration, n_notes), open(str(cache_file), 'wb'))

        Path(self.save_dir, 'temp').mkdir(parents=True, exist_ok=True)
        rows = [row for _, row in self.metadata.iterrows()]
        for rw in tqdm(rows):
            cache_duration_n_notes(rw)
        # pool = Pool(processes=self.workers)
        # pool.map(cache_duration_n_notes, rows)

        duration_train, duration_valid, duration_test = 0, 0, 0
        n_notes_train, n_notes_valid, n_notes_test = 0, 0, 0

        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            cache_file = Path(self.save_dir, 'temp', str(row['pid']) + '.pkl')
            duration, n_notes = pickle.load(open(str(cache_file), 'rb'))

            if row['split'] == 'train':
                duration_train += duration
                n_notes_train += n_notes
            elif row['split'] == 'validation':
                duration_valid += duration
                n_notes_valid += n_notes
            elif row['split'] == 'test':
                duration_test += duration
                n_notes_test += n_notes

        shutil.rmtree(str(Path(self.save_dir, 'temp')))

        if len(flawed_rows_pid) > 0:
            self.metadata = self.metadata[~self.metadata['pid'].isin(flawed_rows_pid)]

        duration_all = duration_train + duration_valid + duration_test
        n_notes_all = n_notes_train + n_notes_valid + n_notes_test

        # ======== print dataset statistics ==========
        print('\n\t=================== Dataset Statistics ====================')
        print('\t\t\t\t\tTrain\t\tValid\t\tTest\t\tAll')
        print('\tn_perfms:\t\t{}\t\t{}\t\t\t{}\t\t\t{}'.format(n_perfms_train, n_perfms_valid, n_perfms_test, n_perfms))
        print('\tduration (h):\t{:.1f}\t\t{:.1f}\t\t{:.1f}\t\t{:.1f}'.format(duration_train / 3600,
                                                                             duration_valid / 3600,
                                                                             duration_test / 3600, duration_all / 3600))
        print('\tn_notes (k):\t{:.1f}\t\t{:.1f}\t\t{:.1f}\t\t{:.1f}\n'.format(n_notes_train / 1000,
                                                                              n_notes_valid / 1000, n_notes_test / 1000,
                                                                              n_notes_all / 1000))

    def prepare_features(self):
        clog.info("prepare features")

        def prepare_one_midi(row_data):
            features, labels = feature_extractor_map[self.feature](row_data)
            return features, labels

        def prepare_split(split):

            meta = self.metadata[self.metadata['split'] == split]
            data = []
            for _, row in tqdm(meta.iterrows(), total=len(meta)):
                dt = prepare_one_midi(row)
                if dt is not None:
                    data.append(dt)
            pickle.dump(data, open(f'{self.save_dir}/{self.feature}-{split}.pkl', 'wb'))

        prepare_split('train')
        prepare_split('validation')
        prepare_split('test')


        if self.feature == 'chord':
            extractor = feature_extractor_map[self.feature]
            if hasattr(extractor, "chord2id"):
                vocab_path = self.save_dir / f"{self.feature}-chord_vocab.pkl"
                with open(vocab_path, "wb") as f:
                    pickle.dump(extractor.chord2id, f)
                clog.info(f"Saved chord vocabulary to {vocab_path}")

        clog.info("\x1B[34m[Info]\033[0m saved all feature")


def extract_unconditional_feature():
    datasets = {
        'maestro': dotenv_config['MAESTRO_PATH'],
        'asap': dotenv_config['ASAP_PATH'],
        'pop909': dotenv_config['POP909_PATH'],
        'hannds': dotenv_config['HANNDS_PATH']
    }

    fp = FeaturePreparation(feature=dotenv_config['FEATURE_TYPE'],
                            save_dir=dotenv_config['OUTPUT_DIR'],
                            data_dir_dict=datasets,
                            check_duplicate=True)
    # fp.extract_meta()
    fp.load_meta()
    fp.print_statistics()
    # would be good to stats by source
    # fp.prepare_features()

    print('hi')

def extract_chord_feature():
    datasets = {
        'pop909': dotenv_config['POP909_PATH']
    }

    fp = FeaturePreparation(
        feature=dotenv_config['FEATURE_TYPE'],
        save_dir=dotenv_config['DATA_DIR'],
        data_dir_dict=datasets,
        check_duplicate=False
    )

    fp.extract_meta()

    fp.load_meta()

    fp.print_statistics()

    fp.prepare_features()


def main():
    # extract_unconditional_feature()
    extract_chord_feature()


if __name__ == "__main__":
    main()
