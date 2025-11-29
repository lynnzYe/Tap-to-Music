"""
Author: Lynn Ye
Created on: 2025/11/14
Brief: 
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ttm.config import model_config, MIN_PIANO_PITCH, dotenv_config
from ttm.data_preparation.utils import midi_to_tap, get_note_sequence_from_midi, ChordConstants
from ttm.data_preparation.feature_preparation import ChordFeatureExtractor
from ttm.model.chord_model import ChordTapLSTM
from ttm.model.uc_model import UCTapLSTM
from ttm.module.uc_module import parse_model_config
from ttm.utils import note_seq_to_midi, clog


def infer_taps(midi_path, model, device, temperature=1.0):
    ns = get_note_sequence_from_midi(midi_path)
    taps, _ = midi_to_tap(ns)
    taps[0, 0] = 88
    taps = torch.tensor(taps, device=device)
    prev_pitch = -1
    (h, c) = None, None

    predicted_pitches = []
    for i in tqdm(range(len(ns))):
        if prev_pitch > 0:
            taps[i, 0] = prev_pitch
        pitch_logits, (h, c) = model(taps[i, ...].unsqueeze(0), None if h is None else (h, c))
        pitch_prob = F.softmax(pitch_logits / temperature, dim=-1)

        # Choose pitch with temperature
        pitch = torch.multinomial(pitch_prob.squeeze(), num_samples=1)
        prev_pitch = pitch.item()
        predicted_pitches.append(prev_pitch)
    ns[:, 0] = np.array(predicted_pitches) + MIN_PIANO_PITCH

    mid = note_seq_to_midi(ns[1:, :])
    return mid

def infer_taps_chord(midi_path, chord_annot_path, model, device, temperature=1.0):
    ns = get_note_sequence_from_midi(midi_path)
    taps, _ = midi_to_tap(ns)
    onsets = ns[:, 1]
    extractor = ChordFeatureExtractor()

    if chord_annot_path:
        segments = extractor._load_chord_segments(chord_annot_path)
        chord_ids = extractor._align_chords_to_onsets(onsets, segments)
    else:
        clog.warning("No chord annotation found or path invalid; using 'N' for all chords.")
        chord_ids = np.full(len(onsets), ChordConstants.N_ID, dtype=np.int32)

    feats = np.concatenate([taps, chord_ids.reshape(-1, 1)], axis=1)
    feats[0, 0] = 88

    x = torch.tensor(feats, device=device, dtype=torch.float32)
    prev_pitch = -1
    (h, c) = None, None

    predicted_pitches = []
    for i in tqdm(range(len(ns))):
        if prev_pitch > 0:
            x[i, 0] = prev_pitch

        step_input = x[i, ...].unsqueeze(0)
        pitch_logits, (h, c) = model(step_input, hx=None if h is None else (h, c))
        pitch_prob = F.softmax(pitch_logits / temperature, dim=-1)

        pitch = torch.multinomial(pitch_prob.squeeze(0), num_samples=1)
        prev_pitch = pitch.item()
        predicted_pitches.append(prev_pitch)

    ns[:, 0] = np.array(predicted_pitches) + MIN_PIANO_PITCH
    mid = note_seq_to_midi(ns[1:, :])
    return mid

def get_state_dict(args):
    checkpoint = torch.load(args.model_state_dict_path, map_location=torch.device(args.device))
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")  # adjust prefix as needed
        new_state_dict[new_key] = v
    return new_state_dict


def load_model(args):
    if args.feature == 'unconditional':
        params = parse_model_config(model_config[args.feature], UCTapLSTM)
        model = UCTapLSTM(**params)
    elif args.feature == 'chord':
        params = parse_model_config(model_config[args.feature], ChordTapLSTM)
        model = ChordTapLSTM(**params)
    else:
        raise ValueError("feature not yet supported")
    model.load_state_dict(get_state_dict(args))
    model = model.to(args.device)
    model.eval()
    return model


def parse_args(required=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, required=required)
    parser.add_argument('--model_state_dict_path', type=str, required=required)
    parser.add_argument('--midi_path', type=str, required=required,
                        help='input midi path. onsets, velocities and durations are used to hint the model')
    parser.add_argument('--output_dir', type=str, required=required, help='dir to save synthesized MIDI')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--temperature', type=float, default=1.0, help='larger = more random')
    parser.add_argument('--midi_name', type=str, default='', help='name of the MIDI file to be saved')
    parser.add_argument('--chord_path', type=str, default='',
                        help='Path to chord annotation file (POP909-style chord_midi.txt)')

    args = parser.parse_args()

    return args


def syn_taps(args):
    model = load_model(args)
    if args.feature == 'unconditional':
        mid = infer_taps(args.midi_path, model, device=args.device, temperature=args.temperature)
        out_name = args.midi_name or 'syn_uc.mid'
    elif args.feature == 'chord':
        mid = infer_taps_chord(args.midi_path, args.chord_annot_path, model,
                               device=args.device, temperature=args.temperature)
        out_name = args.midi_name or 'syn_chord.mid'
    else:
        raise ValueError(f"feature {args.feature} not supported for synthesis")

    path = f'{args.output_dir}/{out_name}'
    mid.save(path)
    clog.info("synthesized MIDI saved at", path)


def main():
    args = parse_args()
    syn_taps(args)


def debug_main():
    args = parse_args(required=False)
    args.midi_path = dotenv_config['MIDI_PATH']
    args.model_state_dict_path = dotenv_config['CKPT_PATH']
    args.feature = dotenv_config['FEATURE_TYPE']
    args.temperature = 1.1
    args.output_dir = dotenv_config['OUTPUT_DIR']
    syn_taps(args)



if __name__ == "__main__":
    # main()
    debug_main()
