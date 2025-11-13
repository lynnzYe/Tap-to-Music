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
from pathos.multiprocessing import ProcessingPool as Pool

from ttm.config import RD_SEED, MAX_PIANO_PITCH, MIN_PIANO_PITCH
from ttm.data_preparation.utils import find_duplicate_midi, get_note_sequence_from_midi

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

random.seed(RD_SEED)
unconditional_datacols = ['source', 'composer', 'piece_name', 'midi_path', 'split']


def parse_any_data(data_dir, train_val_test_split=(0.8, 0.1, 0.1)):
    """Recursively collect all MIDI file paths and split into train/val/test."""

    midi_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                midi_paths.append(os.path.join(root, f))

    if not midi_paths:
        print("Warning: No MIDI files found in", data_dir)
        return pd.DataFrame(columns=["path", "split"])

    # Shuffle for reproducible split
    random.seed(RD_SEED)
    random.shuffle(midi_paths)

    n_total = len(midi_paths)
    n_train = int(n_total * train_val_test_split[0])
    n_val = int(n_total * train_val_test_split[1])

    splits = (["train"] * n_train +
              ["val"] * n_val +
              ["test"] * (n_total - n_train - n_val))

    df = pd.DataFrame({"midi_path": midi_paths, "split": splits})
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
    mstro_sel['source'] = 'maestro'

    asap_sel = asap.rename(columns={'midi_path': 'midi_path',
                                    'piece_name': 'piece_name',
                                    'composer': 'composer',
                                    'split': 'split'})
    asap_sel['source'] = 'asap'

    # Keep only necessary columns
    combined_df = pd.concat([mstro_sel[unconditional_datacols], asap_sel[unconditional_datacols]], ignore_index=True)

    if check_duplicate:
        print("\x1B[34m[Info]\033[0m Checking duplicates")
        overlaps = find_duplicate_midi(combined_df[combined_df['source'] == 'maestro']['midi_path'].tolist(),
                                       combined_df[combined_df['source'] == 'asap']['midi_path'].tolist())
        if len(overlaps) > 0:
            print("\x1B[33m[Warning]\033[0m found", len(overlaps), 'overlapped performances in asap and maestro')
        overlapped_paths = set([f[0] for f in overlaps])
        combined_df = combined_df[~combined_df['midi_path'].isin(overlapped_paths)].reset_index(drop=True)

    return combined_df


def parse_pop909_dataset(data_dir):
    pass


class MIDIData:
    # Integrate different datasets into one
    def __init__(self, data_dir_list, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
        for p in data_dir_list:
            assert os.path.exists(p)
        self.data_dir_list = data_dir_list
        self.data = None
        self.check_duplicate = check_duplicate
        self.train_val_test_split = train_val_test_split

    def _collect_data(self):
        raise NotImplementedError


class UnconditionalMIDIData(MIDIData):
    def __init__(self, data_dir_list, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=True):
        super().__init__(data_dir_list, train_val_test_split, check_duplicate)
        self.data = self._collect_data()

    def _collect_data(self):
        # Init dataframe
        data = pd.DataFrame(columns=unconditional_datacols)

        maestro_asap = [p for p in self.data_dir_list if 'maestro' in p.lower() or 'asap' in p.lower()]
        other_datasets = [p for p in self.data_dir_list if p not in maestro_asap]
        assert len(maestro_asap) == 0 or len(maestro_asap) == 2
        if maestro_asap:
            mstro_dir = maestro_asap[0] if 'maestro' in maestro_asap[0].lower() else maestro_asap[1]
            asap_dir = maestro_asap[0] if 'asap' in maestro_asap[0].lower() else maestro_asap[1]
            mstro_data = parse_maestro_data(mstro_dir)
            asap_data = parse_asap_data(asap_dir)
            integrated_dataset = integrate_maestro_asap(mstro_data, asap_data, self.train_val_test_split,
                                                        self.check_duplicate)
            data = pd.concat([data, integrated_dataset[unconditional_datacols]], ignore_index=True)
        for p in other_datasets:
            sub_data = parse_any_data(p, self.train_val_test_split)
            sub_data = sub_data.reindex(columns=unconditional_datacols)
            data = pd.concat([data, sub_data[unconditional_datacols]], ignore_index=True)
        data['pid'] = range(len(data))
        return data


class ChordMIDIData(MIDIData):
    def __init__(self, data_dir_list, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
        super().__init__(data_dir_list, train_val_test_split, check_duplicate)
        self.data = self._collect_data()

    def _collect_data(self):
        raise NotImplementedError


class ScoreMIDIData(MIDIData):
    def __init__(self, data_dir_list, train_val_test_split=(0.8, 0.1, 0.1), check_duplicate=False):
        super().__init__(data_dir_list, train_val_test_split, check_duplicate)
        self.data = self._collect_data()

    def _collect_data(self):
        raise NotImplementedError


data_class = {
    'unconditional': UnconditionalMIDIData,
    'chord': ChordMIDIData
}


def midi_to_tap(note_sequence):
    """
    Convert a note sequence to sequence of tapping events
    :param note_sequence:
    :return:
    """
    pass


def extract_unconditional_feature(row):
    note_sequence = get_note_sequence_from_midi(row['midi_path'])
    assert np.max(note_sequence[:, 0]) <= MAX_PIANO_PITCH and np.min(note_sequence[:, 0]) >= MIN_PIANO_PITCH


class FeaturePreparation:
    def __init__(self, feature, save_dir, data_dir_list, train_val_test_split=(0.8, 0.1, 0.1), workers=8,
                 check_duplicate=True):
        assert feature in data_class.keys()
        self.check_duplicate = True
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.train_val_test_split = train_val_test_split
        self.data_dir = data_dir_list
        self.feature = feature
        self.workers = workers
        self.metadata = None

    def extract_meta(self):
        meta_dataset = data_class[self.feature](self.data_dir, self.train_val_test_split)
        meta_dataset.data.to_csv(self.save_dir / f'{self.feature}-meta.csv', index=False)

    def load_meta(self):
        self.metadata = pd.read_csv(self.save_dir / f'{self.feature}-meta.csv')

    def print_statistics(self):
        print('INFO: Printing dataset statistics')

        # =========== number of performances ===========
        print('INFO: Get number of performances')

        n_perfms_train = len(self.metadata[self.metadata['split'] == 'train'])
        n_perfms_valid = len(self.metadata[self.metadata['split'] == 'validation'])
        n_perfms_test = len(self.metadata[self.metadata['split'] == 'test'])
        n_perfms = n_perfms_train + n_perfms_valid + n_perfms_test

        # ========== duration & number of notes ==========
        print('INFO: Get duration & number of notes')

        def cache_duration_n_notes(row):

            # print('INFO: Get duration & number of notes for performance {}'.format(row['performance_id']))
            midi_data = pm.PrettyMIDI(row['midi_path'])
            duration = midi_data.get_end_time()
            n_notes = np.sum([len(midi_data.instruments[i].notes) for i in range(len(midi_data.instruments))])

            cache_file = Path(self.save_dir, 'temp', str(row['pid']) + '.pkl')
            pickle.dump((duration, n_notes), open(str(cache_file), 'wb'))

        Path(self.save_dir, 'temp').mkdir(parents=True, exist_ok=True)
        rows = [row for _, row in self.metadata.iterrows()]
        pool = Pool(processes=self.workers)
        pool.map(cache_duration_n_notes, rows)

        duration_train, duration_valid, duration_test = 0, 0, 0
        n_notes_train, n_notes_valid, n_notes_test = 0, 0, 0

        for _, row in self.metadata.iterrows():
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


def check_mstr_asap_data():
    mstr_path = '/Users/kurono/Documents/code/data/maestro-v3.0.0'
    asap_path = '/Users/kurono/Documents/code/data/acpas/asap'

    fp = FeaturePreparation(feature='unconditional',
                            save_dir='/Users/kurono/Desktop/10701 final/tap_the_music/output',
                            data_dir_list=[mstr_path, asap_path])
    # fp.extract_meta()
    fp.load_meta()
    fp.print_statistics()

    print('hi')


def main():
    noteseq = get_note_sequence_from_midi('/Users/kurono/Documents/code/data/acpas/asap/Bach/Fugue/bwv_846/Shi05M.mid')
    # check_mstr_asap_data()
    print("Hello, world!")


if __name__ == "__main__":
    main()
