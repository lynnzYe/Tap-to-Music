from functools import partial
from pathlib import Path

import torch

from ttm.data_preparation.dataset import UnconditionalDataset, ChordDataset
from ttm.data_preparation.utils import ChordConstants


def collate_fn_pad(batch, max_length=128):
    features_list, labels_list = zip(*batch)
    
    max_len = min(max(f.shape[0] for f in features_list), max_length)
    
    padded_features = []
    padded_labels = []
    
    for feat, label in zip(features_list, labels_list):
        if feat.shape[0] > max_length:
            feat = feat[:max_length]
            label = label[:max_length]
        
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            if feat.shape[1] == 4:  # unconditional
                pad_row = torch.tensor([[88, 0, 0, 0]], dtype=feat.dtype)
            elif feat.shape[1] == 5:  # chord
                pad_row = torch.tensor([[88, 0, 0, 0, ChordConstants.N_ID]], dtype=feat.dtype)
            else:
                pad_row = torch.zeros((1, feat.shape[1]), dtype=feat.dtype)
                pad_row[0, 0] = 88
            
            feat_padded = torch.cat([feat, pad_row.repeat(pad_len, 1)], dim=0)
            label_padded = torch.cat([label, torch.full((pad_len,), 88, dtype=label.dtype)], dim=0)
        else:
            feat_padded = feat
            label_padded = label
        
        padded_features.append(feat_padded)
        padded_labels.append(label_padded)
    
    features_batch = torch.stack(padded_features, dim=0)
    labels_batch = torch.stack(padded_labels, dim=0)
    
    return features_batch, labels_batch


def create_dataloader_from_pickle(data_dir, feature_type, batch_size=64, num_workers=1, max_length=128, split='test'):
    data_dir = Path(data_dir)
    
    feature_folder = str(data_dir)
    pickle_file = data_dir / f'{feature_type}-{split}.pkl'
    
    if feature_type == 'unconditional':
        dataset = UnconditionalDataset(
            feature_folder=feature_folder,
            split=split,
            feature_type='unconditional'
        )
    elif feature_type == 'chord':
        dataset = ChordDataset(
            feature_folder=feature_folder,
            split=split,
            feature_type='chord'
        )
    
    dataset.max_length = max_length
    
    collate_fn = partial(collate_fn_pad, max_length=max_length)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader

