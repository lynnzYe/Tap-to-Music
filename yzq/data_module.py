"""
Author: YZQ
Created on: 2025/11/27
Brief: PyTorch Lightning DataModules for POP909 datasets
       Extends pl.LightningDataModule

       Two data modules available:
       - HandDataModule: For 5-feature hand data (pitch, dt, dur, vel, hand)
       - WindowDataModule: For 5-feature window data (pitch, dt, dur, vel, window_avg)

       Reference: https://github.com/lynnzYe/Tap-to-Music/blob/9bf3016803e6038b302265dabbd68414c9a71b0a/ttm/data_preparation/data_module.py
"""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler

from ttm.config import config
from yzq.dataset import HandDataset, WindowDataset


# Default batch size and num_workers if not in config
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4


class HandDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for hand-labeled POP909 data.
    
    Uses HandDataset which loads pop909hand-*.pkl files with 5 features:
    [pitch, dt, dur, vel, hand]
    """
    
    def __init__(self, data_dir, feature_type='pop909hand', batch_size=None, num_workers=None):
        """
        Args:
            data_dir: Path to folder containing pop909hand-*.pkl files
            feature_type: Prefix for pkl files (default: 'pop909hand')
            batch_size: Batch size (uses config if None)
            num_workers: Number of workers (uses config if None)
        """
        super().__init__()
        self.data_dir = data_dir
        self.feature_type = feature_type
        
        # Get config values or use defaults
        cfg = config.get(feature_type, config.get('unconditional', {}))
        self.batch_size = batch_size or cfg.get('batch_size', DEFAULT_BATCH_SIZE)
        self.num_workers = num_workers or cfg.get('num_workers', DEFAULT_NUM_WORKERS)
        
        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Set up datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = HandDataset(self.data_dir, 'train', self.feature_type)
            self.val_dataset = HandDataset(self.data_dir, 'validation', self.feature_type)
        
        if stage == 'test' or stage is None:
            self.test_dataset = HandDataset(self.data_dir, 'test', self.feature_type)

    def _get_dataset(self, split):
        """Get dataset for a specific split."""
        assert split in ['train', 'validation', 'test']
        return HandDataset(self.data_dir, split, self.feature_type)

    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            self.train_dataset = self._get_dataset('train')
        
        sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.val_dataset = self._get_dataset('validation')
        
        sampler = RandomSampler(self.val_dataset)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader with batch_size=1 for variable-length sequences."""
        if self.test_dataset is None:
            self.test_dataset = self._get_dataset('test')
        
        # Use batch_size=1 for test to handle variable-length sequences
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )


class WindowDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for window-averaged POP909 data.
    
    Uses WindowDataset which loads pop909window-*.pkl files with 5 features:
    [pitch, dt, dur, vel, window_avg]
    """
    
    def __init__(self, data_dir, feature_type='pop909window', batch_size=None, 
                 num_workers=None, window_size=8):
        """
        Args:
            data_dir: Path to folder containing pop909window-*.pkl files
            feature_type: Prefix for pkl files (default: 'pop909window')
            batch_size: Batch size (uses config if None)
            num_workers: Number of workers (uses config if None)
            window_size: Window size for cluster augmentation
        """
        super().__init__()
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.window_size = window_size
        
        # Get config values or use defaults
        cfg = config.get(feature_type, config.get('unconditional', {}))
        self.batch_size = batch_size or cfg.get('batch_size', DEFAULT_BATCH_SIZE)
        self.num_workers = num_workers or cfg.get('num_workers', DEFAULT_NUM_WORKERS)
        
        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Set up datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = WindowDataset(
                self.data_dir, 'train', self.feature_type, self.window_size
            )
            self.val_dataset = WindowDataset(
                self.data_dir, 'validation', self.feature_type, self.window_size
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = WindowDataset(
                self.data_dir, 'test', self.feature_type, self.window_size
            )

    def _get_dataset(self, split):
        """Get dataset for a specific split."""
        assert split in ['train', 'validation', 'test']
        return WindowDataset(self.data_dir, split, self.feature_type, self.window_size)

    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            self.train_dataset = self._get_dataset('train')
        
        sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.val_dataset = self._get_dataset('validation')
        
        sampler = RandomSampler(self.val_dataset)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader with batch_size=1 for variable-length sequences."""
        if self.test_dataset is None:
            self.test_dataset = self._get_dataset('test')
        
        # Use batch_size=1 for test to handle variable-length sequences
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )


def main():
    """Test the data modules."""
    from pathlib import Path
    
    # Get the yzq output directory
    yzq_dir = Path(__file__).resolve().parent
    features_dir = yzq_dir / "output"
    
    if not features_dir.exists():
        print(f"Features directory not found at {features_dir}")
        print("Run data_preparation.py first to generate pkl files.")
        return
    
    # Test HandDataModule if files exist
    hand_pkl = features_dir / "pop909hand-train.pkl"
    if hand_pkl.exists():
        print("\n" + "=" * 60)
        print("Testing HandDataModule")
        print("=" * 60)
        
        dm = HandDataModule(
            data_dir=str(features_dir),
            feature_type='pop909hand',
            batch_size=4,
            num_workers=0  # Use 0 for testing
        )
        dm.setup('fit')
        
        train_loader = dm.train_dataloader()
        print(f"Train batches: {len(train_loader)}")
        
        # Fetch one batch
        for batch in train_loader:
            note_seq, labels = batch
            print(f"Batch note_seq shape: {note_seq.shape}")
            print(f"Batch labels shape: {labels.shape}")
            
            # Verify pitch range
            max_pitch = torch.max(note_seq[:, :, 0])
            if max_pitch > 88:
                print(f"WARNING: Max pitch {max_pitch} exceeds 88!")
            else:
                print(f"Pitch range OK (max={max_pitch})")
            break
    else:
        print(f"Hand dataset not found at {hand_pkl}")
    
    # Test WindowDataModule if files exist
    window_pkl = features_dir / "pop909window-train.pkl"
    if window_pkl.exists():
        print("\n" + "=" * 60)
        print("Testing WindowDataModule")
        print("=" * 60)
        
        dm = WindowDataModule(
            data_dir=str(features_dir),
            feature_type='pop909window',
            batch_size=4,
            num_workers=0,
            window_size=8
        )
        dm.setup('fit')
        
        train_loader = dm.train_dataloader()
        print(f"Train batches: {len(train_loader)}")
        
        # Fetch one batch
        for batch in train_loader:
            note_seq, labels = batch
            print(f"Batch note_seq shape: {note_seq.shape}")
            print(f"Batch labels shape: {labels.shape}")
            
            # Verify pitch range
            max_pitch = torch.max(note_seq[:, :, 0])
            if max_pitch > 88:
                print(f"WARNING: Max pitch {max_pitch} exceeds 88!")
            else:
                print(f"Pitch range OK (max={max_pitch})")
            break
    else:
        print(f"Window dataset not found at {window_pkl}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

