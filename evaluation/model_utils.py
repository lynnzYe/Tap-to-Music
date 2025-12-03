import inspect
from collections import OrderedDict
from pathlib import Path

import torch

from ttm.model.chord_model import ChordTapLSTM
from ttm.model.uc_model import UCTapLSTM


def parse_model_config(config_dict, model_type):
    sig = inspect.signature(model_type.__init__)
    allowed_keys = [p for p in sig.parameters if p != "self"]
    filtered_config = {k: v for k, v in config_dict.items() if k in allowed_keys}
    return filtered_config


def load_model_from_checkpoint(checkpoint_path, feature_type, model_config_dict=None, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if model_config_dict is None and "hyper_parameters" in checkpoint:
            model_config_dict = checkpoint["hyper_parameters"].get("config", {})
    else:
        state_dict = checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v
    
    if model_config_dict is None:
        model_config_dict = {
            'n_pitches': 88,
            'pitch_emb_dim': 32,
            'dt_dim': 1,
            'vel_dim': 1,
            'dur_dim': 1,
            'hidden': 128,
            'layers': 2,
            'dropout': 0.15,
            'linear_dim': 128,
        }
        if feature_type == 'chord':
            model_config_dict.update({
                'num_chords': 109,
                'chord_emb_dim': 32,
                'film_hidden_dim': 64,
                'use_next_chord': False,
            })
    
    if feature_type == 'unconditional':
        params = parse_model_config(model_config_dict, UCTapLSTM)
        model = UCTapLSTM(**params)
    elif feature_type == 'chord':
        params = parse_model_config(model_config_dict, ChordTapLSTM)
        model = ChordTapLSTM(**params)
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

