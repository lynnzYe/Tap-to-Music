"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: 
"""
import pytorch_lightning as pl
import torch

from ttm.config import config
from ttm.data_preparation.dataset import UnconditionalDataset


class UCTapDataModule(pl.LightningModule):
    def __init__(self, data_dir, feature='unconditional'):
        super().__init__()
        self.data_dir = data_dir
        self.feature = feature

    def _get_dataset(self, split):
        assert split in ['train', 'validation', 'test']
        if self.feature == 'beat':
            dataset = UnconditionalDataset(self.data_dir, split, self.feature)
        else:
            raise ValueError('Unknown feature: {}'.format(self.feature))
        return dataset

    def train_dataloader(self):
        dataset = self._get_dataset(split='train')
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=config[self.feature]['batch_size'],
            sampler=sampler,
            num_workers=config[self.feature]['num_workers'],
            drop_last=True,
            persistent_workers=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = self._get_dataset(split='validation')
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=config[self.feature]['batch_size'],
            sampler=sampler,
            num_workers=config[self.feature]['num_workers'],
            drop_last=True,
            persistent_workers=True
        )
        return dataloader


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
