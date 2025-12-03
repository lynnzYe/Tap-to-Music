import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

MIN_PIANO_PITCH = 21


def compute_pitch_statistics(sequences, name=''):
    all_pitches = np.array([p for seq in sequences for p in seq])
    all_pitches = all_pitches[all_pitches != 88]
    
    if len(all_pitches) == 0:
        return {}
    
    midi_pitches = all_pitches + MIN_PIANO_PITCH
    pitch_classes = midi_pitches % 12
    pitch_heights = midi_pitches - MIN_PIANO_PITCH
    
    pc_dist = Counter(pitch_classes)
    height_dist = Counter(pitch_heights)
    total = len(all_pitches)
    
    return {
        f'{name}_pitch_class_dist': {int(k): v / total for k, v in pc_dist.items()},
        f'{name}_pitch_height_dist': {int(k): v / total for k, v in height_dist.items()},
        f'{name}_mean_pitch': float(midi_pitches.mean()),
        f'{name}_std_pitch': float(midi_pitches.std()),
        f'{name}_pitch_range': [int(midi_pitches.min()), int(midi_pitches.max())],
    }


def plot_pitch_histograms(gt_stats, gen_stats, output_dir, name=''):
    os.makedirs(output_dir, exist_ok=True)
    
    gt_pc_key = None
    gen_pc_key = None
    gt_height_key = None
    gen_height_key = None
    
    for key in gt_stats:
        if key.endswith('pitch_class_dist'):
            gt_pc_key = key
        elif key.endswith('pitch_height_dist'):
            gt_height_key = key
    
    for key in gen_stats:
        if key.endswith('pitch_class_dist'):
            gen_pc_key = key
        elif key.endswith('pitch_height_dist'):
            gen_height_key = key
    
    if gt_pc_key is None or gen_pc_key is None or gt_height_key is None or gen_height_key is None:
        return
    
    gt_pc = gt_stats[gt_pc_key]
    gen_pc = gen_stats[gen_pc_key]
    gt_height = gt_stats[gt_height_key]
    gen_height = gen_stats[gen_height_key]
    
    if not gt_pc or not gen_pc or not gt_height or not gen_height:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    pc_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    x = np.arange(12)
    width = 0.35
    
    gt_pc_values = [gt_pc.get(i, 0) for i in range(12)]
    gen_pc_values = [gen_pc.get(i, 0) for i in range(12)]
    
    ax1.bar(x - width/2, gt_pc_values, width, label='Ground Truth', alpha=0.7)
    ax1.bar(x + width/2, gen_pc_values, width, label='Generated', alpha=0.7)
    ax1.set_xlabel('Pitch Class')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Pitch Class Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pc_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    max_height = max(max(gt_height.keys(), default=0), max(gen_height.keys(), default=0)) + 1
    heights = np.arange(max_height)
    
    gt_height_values = [gt_height.get(i, 0) for i in heights]
    gen_height_values = [gen_height.get(i, 0) for i in heights]
    
    ax2.plot(heights, gt_height_values, label='Ground Truth', alpha=0.7, linewidth=2)
    ax2.plot(heights, gen_height_values, label='Generated', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Pitch Height (MIDI - 21)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Pitch Height Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pitch_histograms_{name}.png'), dpi=150)
    plt.close()

