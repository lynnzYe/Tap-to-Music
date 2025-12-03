import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ttm.config import MIN_PIANO_PITCH
from ttm.data_preparation.utils import get_note_sequence_from_midi, midi_to_tap
from ttm.model.chord_model import ChordTapLSTM
from ttm.model.uc_model import UCTapLSTM
from ttm.utils import note_seq_to_midi
from evaluation.model_utils import load_model_from_checkpoint


def predict_midi_file(
    checkpoint_path,
    midi_input_path,
    midi_output_path,
    feature_type='unconditional',
    device='cpu',
    temperature=1.0,
):
    print(f"Loading model from {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, feature_type, model_config_dict, device)
    
    print(f"Loading MIDI file: {midi_input_path}")
    ns = get_note_sequence_from_midi(midi_input_path)
    
    taps, _ = midi_to_tap(ns)
    
    taps_tensor = torch.tensor(taps, device=device, dtype=torch.float32)
    
    taps_tensor[0, 0] = 88
    prev_pitch = -1
    (h, c) = None, None
    
    predicted_pitches = []
    
    print("Generating predictions...")
    for i in tqdm(range(len(ns))):
        if prev_pitch >= 0:
            taps_tensor[i, 0] = prev_pitch
        
        step_input = taps_tensor[i, ...].unsqueeze(0)
        
        pitch_logits, (h, c) = model(step_input, None if h is None else (h, c))
        
        pitch_prob = F.softmax(pitch_logits / temperature, dim=-1)
        pitch = torch.multinomial(pitch_prob.squeeze(), num_samples=1)
        prev_pitch = pitch.item()
        
        predicted_pitches.append(prev_pitch)
    
    ns[:, 0] = np.array(predicted_pitches) + MIN_PIANO_PITCH
    
    print(f"Saving predictions to {midi_output_path}")
    mid = note_seq_to_midi(ns[1:, :])
    mid.save(midi_output_path)


def main():
    parser = argparse.ArgumentParser(description='Predict on a single MIDI file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--input_midi', type=str, required=True,
                       help='Path to input MIDI file')
    parser.add_argument('--output_midi', type=str, required=True,
                       help='Path to save output MIDI file')
    parser.add_argument('--feature', type=str, default='unconditional',
                       choices=['unconditional', 'chord'],
                       help='Feature type')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (default: 1.0)')
    
    args = parser.parse_args()
    
    predict_midi_file(
        checkpoint_path=args.checkpoint,
        midi_input_path=args.input_midi,
        midi_output_path=args.output_midi,
        feature_type=args.feature,
        device=args.device,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

