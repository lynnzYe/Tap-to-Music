"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: 
"""
import pytorch_lightning as pl
import torch

from ttm.config import config, dotenv_config
from ttm.data_preparation.dataset import ChordDataset, UnconditionalDataset


# UnconditionalTapDataModule
class UCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, feature='unconditional'):
        super().__init__()
        assert feature in ['unconditional', 'chord'], f"Unsupported feature type: {feature}"
        self.data_dir = data_dir
        self.feature = feature

    def _get_dataset(self, split):
        assert split in ['train', 'validation', 'test']
        if self.feature == 'unconditional':
            dataset_cls = UnconditionalDataset
        elif self.feature == 'chord':
            dataset_cls = ChordDataset
        else:
            raise ValueError(f"Unsupported feature type: {self.feature}")

        dataset = dataset_cls(self.data_dir, split, self.feature)
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

    def test_dataloader(self):
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
    # Check data module
    data = UCDataModule(dotenv_config['DATA_DIR'], feature=dotenv_config['FEATURE_TYPE'])
    tr = data.train_dataloader()
    for i, batch in enumerate(tr):
        if torch.max(batch[0][:, :, 0]) > 88:
            raise Exception('chord pitch index out of range')
        
        print(batch[0].shape, batch[1].shape)
        break

    print("Hello, world!")

if __name__ == "__main__":
    main()
