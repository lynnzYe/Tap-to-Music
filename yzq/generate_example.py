"""
Generation example for Hand and Window FiLM models.
Generates music from a random MIDI file and outputs both input and generated results
as MIDI and WAV files.

Author: YZQ
Created on: 2025/12/02
Brief: Generation example for Hand and Window FiLM models
       Generates music from a random MIDI file and outputs both input and generated results
       as MIDI and WAV files.

       Reference: https://github.com/lynnzYe/Tap-to-Music/blob/9bf3016803e6038b302265dabbd68414c9a71b0a/ttm/generate_example.py
"""
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import pretty_midi

from ttm.config import config, model_config, MIN_PIANO_PITCH, MAX_PIANO_PITCH
from ttm.data_preparation.utils import get_note_sequence_from_midi, midi_to_tap
from yzq.model import HandFiLMModule, WindowFiLMModule

# Seed for reproducibility
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_random_midi(pop909_dir):
    """Load a random MIDI file from POP909 dataset."""
    pop909_dir = Path(pop909_dir)
    pop909_main = pop909_dir / "POP909"
    
    midi_files = []
    if pop909_main.exists():
        for song_dir in sorted(pop909_main.iterdir()):
            if song_dir.is_dir():
                for midi_file in song_dir.glob("*.mid"):
                    midi_files.append(midi_file)
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {pop909_dir}")
    
    selected = random.choice(midi_files)
    print(f"Selected MIDI: {selected}")
    return selected


def extract_timing_features(midi_path):
    """
    Extract timing features from MIDI file.
    Returns: features (N, 4), original_labels (N,), note_sequence
    """
    note_sequence = get_note_sequence_from_midi(str(midi_path))
    features, labels = midi_to_tap(note_sequence)
    return features, labels, note_sequence


def compute_window_avg_single(pitches, window_size=8):
    """Compute causal window average for a single sequence of pitches."""
    window_avg = np.zeros(len(pitches), dtype=float)
    for i in range(len(pitches)):
        if i == 0:
            window_avg[i] = pitches[0]
        else:
            start_idx = max(0, i - window_size)
            window_avg[i] = np.mean(pitches[start_idx:i])
    return window_avg


def generate_with_window_model(model, features, original_labels, window_size=8, 
                                temperature=1.0, top_k=None, device='cpu'):
    """
    Generate pitches using the WindowFiLM model autoregressively.
    
    Args:
        model: WindowFiLMModule
        features: (N, 4) numpy array [pitch, log_dt, log_dur, vel]
        original_labels: (N,) original pitches for comparison
        window_size: window size for averaging
        temperature: sampling temperature
        top_k: if set, use top-k sampling
        device: torch device
    
    Returns:
        generated_pitches: (N,) numpy array of generated pitch indices
    """
    model.eval()
    model.to(device)
    
    N = len(features)
    generated_pitches = np.zeros(N, dtype=int)
    
    # Normalize features to model input format
    features_normalized = features.copy()
    features_normalized[:, 0] -= MIN_PIANO_PITCH  # Normalize pitch to 0-88
    
    with torch.no_grad():
        # Initialize with first prediction (using pad token)
        hx = None
        
        for i in range(N):
            # Build input: [pitch, dt, dur, vel, window_avg]
            if i == 0:
                # First step: use pad token (88) as pitch input
                curr_pitch = 88  # pad token
                window_avg = 0.0
            else:
                curr_pitch = generated_pitches[i-1]  # Use previously generated pitch
                # Compute window average on generated pitches (shifted back to original scale)
                generated_so_far = generated_pitches[:i] + MIN_PIANO_PITCH
                window_avg = compute_window_avg_single(generated_so_far, window_size)[-1]
            
            # Prepare input tensor (batch=1, seq=1, features=5)
            input_features = np.array([[
                curr_pitch,  # pitch (normalized)
                features_normalized[i, 1],  # log_dt
                features_normalized[i, 2],  # log_dur
                features_normalized[i, 3],  # vel
                window_avg  # window_avg
            ]], dtype=np.float32)
            
            x = torch.tensor(input_features).unsqueeze(0).to(device)  # (1, 1, 5)
            
            # Forward pass
            logits, hx = model(x, hx=hx)  # logits: (1, 1, 89)
            logits = logits[0, 0, :88]  # Remove pad class, get (88,)
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample from distribution
            if top_k is not None:
                # Top-k sampling
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = torch.softmax(topk_logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                pitch_idx = topk_indices[idx].item()
            else:
                # Greedy or full softmax sampling
                probs = torch.softmax(logits, dim=-1)
                pitch_idx = torch.multinomial(probs, 1).item()
            
            generated_pitches[i] = pitch_idx
    
    return generated_pitches


def generate_with_hand_model(model, features, hand_labels, original_labels,
                              temperature=1.0, top_k=None, device='cpu'):
    """
    Generate pitches using the HandFiLM model autoregressively.
    
    Args:
        model: HandFiLMModule
        features: (N, 4) numpy array [pitch, log_dt, log_dur, vel]
        hand_labels: (N,) hand labels (0=left, 1=right)
        original_labels: (N,) original pitches
        temperature: sampling temperature
        top_k: if set, use top-k sampling
        device: torch device
    
    Returns:
        generated_pitches: (N,) numpy array of generated pitch indices
    """
    model.eval()
    model.to(device)
    
    N = len(features)
    generated_pitches = np.zeros(N, dtype=int)
    
    # Normalize features
    features_normalized = features.copy()
    features_normalized[:, 0] -= MIN_PIANO_PITCH
    
    with torch.no_grad():
        hx = None
        
        for i in range(N):
            if i == 0:
                curr_pitch = 88  # pad token
                curr_hand = hand_labels[0] if len(hand_labels) > 0 else 0
            else:
                curr_pitch = generated_pitches[i-1]
                curr_hand = hand_labels[i]
            
            # Prepare input (batch=1, seq=1, features=5)
            input_features = np.array([[
                curr_pitch,
                features_normalized[i, 1],
                features_normalized[i, 2],
                features_normalized[i, 3],
                curr_hand
            ]], dtype=np.float32)
            
            x = torch.tensor(input_features).unsqueeze(0).to(device)
            
            # Forward pass
            logits, hx = model(x, hx=hx)
            logits = logits[0, 0, :88]
            
            # Apply temperature and sample
            logits = logits / temperature
            
            if top_k is not None:
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = torch.softmax(topk_logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                pitch_idx = topk_indices[idx].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                pitch_idx = torch.multinomial(probs, 1).item()
            
            generated_pitches[i] = pitch_idx
    
    return generated_pitches


def pitches_to_midi(pitches, note_sequence, output_path):
    """
    Convert generated pitches back to MIDI file using original timing.
    
    Args:
        pitches: (N,) array of pitch indices (0-87, need to add MIN_PIANO_PITCH)
        note_sequence: Original note sequence for timing info
        output_path: Output MIDI file path
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")
    
    for i, (pitch_idx, orig_note) in enumerate(zip(pitches, note_sequence)):
        pitch = int(pitch_idx) + MIN_PIANO_PITCH  # Convert back to MIDI pitch
        pitch = np.clip(pitch, 21, 108)  # Ensure valid piano range
        
        note = pretty_midi.Note(
            velocity=int(orig_note[3]),  # Use original velocity
            pitch=pitch,
            start=float(orig_note[1]),  # Use original onset
            end=float(orig_note[1] + orig_note[2])  # onset + duration
        )
        piano.notes.append(note)
    
    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"Saved MIDI: {output_path}")


def midi_to_wav(midi_path, wav_path, sf2_path=None):
    """
    Convert MIDI to WAV using FluidSynth.
    
    Args:
        midi_path: Input MIDI file
        wav_path: Output WAV file
        sf2_path: Path to SoundFont file (optional, uses default if None)
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Try to synthesize with FluidSynth
        # Note: This requires FluidSynth and a SoundFont installed
        if sf2_path:
            audio = pm.fluidsynth(fs=44100, sf2_path=sf2_path)
        else:
            audio = pm.fluidsynth(fs=44100)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save as WAV
        import scipy.io.wavfile as wav
        wav.write(str(wav_path), 44100, (audio * 32767).astype(np.int16))
        print(f"Saved WAV: {wav_path}")
        return True
        
    except Exception as e:
        print(f"Could not convert to WAV: {e}")
        print("Make sure FluidSynth is installed: brew install fluid-synth")
        print("And a SoundFont is available")
        return False


def load_hand_labels(midi_path, features, data_dir='yzq/output'):
    """
    Load or compute hand labels for a MIDI file.
    For simplicity, we'll use a heuristic based on pitch (higher = right hand).
    """
    # Simple heuristic: notes above middle C (60) are right hand, below are left
    # This is a rough approximation when we don't have the trained hand model
    pitches = features[:, 0]  # Original pitches before normalization
    hand_labels = (pitches >= 60).astype(int)  # 1 = right, 0 = left
    return hand_labels


def main():
    parser = argparse.ArgumentParser(description='Generate music examples with trained models')
    
    parser.add_argument('--pop909_dir', type=str, default='POP909-Dataset',
                        help='Path to POP909 dataset')
    parser.add_argument('--output_dir', type=str, default='yzq/generated_examples',
                        help='Output directory for generated files')
    parser.add_argument('--window_checkpoint', type=str, 
                        default='yzq/checkpoints/window_sizes/window_size_8/ws8-last.ckpt',
                        help='Path to Window model checkpoint')
    parser.add_argument('--hand_checkpoint', type=str,
                        default='yzq/checkpoints/hand_film_v3/hand_film-last.ckpt',
                        help='Path to Hand model checkpoint')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for window model')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-k sampling (None for full distribution)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--sf2_path', type=str, default=None,
                        help='Path to SoundFont file for WAV synthesis')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--midi_path', type=str, default=None,
                        help='Specific MIDI file to use (random if not specified)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load random MIDI file
    if args.midi_path:
        midi_path = Path(args.midi_path)
    else:
        midi_path = load_random_midi(args.pop909_dir)
    
    # Extract features
    print("\n" + "="*60)
    print("Extracting features from MIDI...")
    print("="*60)
    features, original_labels, note_sequence = extract_timing_features(midi_path)
    print(f"Number of notes: {len(note_sequence)}")
    
    # Copy original MIDI to output
    original_midi_path = output_dir / f"original_{midi_path.stem}.mid"
    import shutil
    shutil.copy(midi_path, original_midi_path)
    print(f"Copied original MIDI: {original_midi_path}")
    
    # Convert original to WAV
    original_wav_path = output_dir / f"original_{midi_path.stem}.wav"
    midi_to_wav(original_midi_path, original_wav_path, args.sf2_path)
    
    # ========== Window Model Generation ==========
    print("\n" + "="*60)
    print("Generating with Window FiLM Model...")
    print("="*60)
    
    if Path(args.window_checkpoint).exists():
        try:
            # Load model config
            m_config = {
                **model_config.get('unconditional', {}),
                'learning_rate': 3e-4,
                'window_emb_dim': 32,
                'film_hidden_dim': 128,
                'multi_layer_film': True,
                'window_dropout': 0.1,
                'label_smoothing': 0.1,
                'window_mean': 63.0,
                'window_std': 5.0,
            }
            
            window_model = WindowFiLMModule.load_from_checkpoint(
                args.window_checkpoint,
                m_config=m_config,
                map_location=args.device
            )
            
            # Generate
            window_pitches = generate_with_window_model(
                window_model, features, original_labels,
                window_size=args.window_size,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device
            )
            
            # Save generated MIDI
            window_midi_path = output_dir / f"window_generated_{midi_path.stem}.mid"
            pitches_to_midi(window_pitches, note_sequence, window_midi_path)
            
            # Convert to WAV
            window_wav_path = output_dir / f"window_generated_{midi_path.stem}.wav"
            midi_to_wav(window_midi_path, window_wav_path, args.sf2_path)
            
            # Print comparison
            print(f"\nWindow Model Results:")
            print(f"  Original pitches (first 20): {original_labels[:20].astype(int)}")
            print(f"  Generated pitches (first 20): {window_pitches[:20] + MIN_PIANO_PITCH}")
            accuracy = np.mean(window_pitches == (original_labels - MIN_PIANO_PITCH))
            print(f"  Reconstruction accuracy: {accuracy*100:.2f}%")
            
        except Exception as e:
            print(f"Window model generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Window checkpoint not found: {args.window_checkpoint}")
    
    # ========== Hand Model Generation ==========
    print("\n" + "="*60)
    print("Generating with Hand FiLM Model...")
    print("="*60)
    
    if Path(args.hand_checkpoint).exists():
        try:
            # Load hand labels (using heuristic)
            hand_labels = load_hand_labels(midi_path, features)
            
            m_config = {
                **model_config.get('unconditional', {}),
                'learning_rate': 3e-4,
                'num_hands': 2,
                'hand_emb_dim': 32,
                'film_hidden_dim': 128,
                'multi_layer_film': True,
                'hand_dropout': 0.1,
                'label_smoothing': 0.1,
            }
            
            hand_model = HandFiLMModule.load_from_checkpoint(
                args.hand_checkpoint,
                m_config=m_config,
                map_location=args.device
            )
            
            # Generate
            hand_pitches = generate_with_hand_model(
                hand_model, features, hand_labels, original_labels,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device
            )
            
            # Save generated MIDI
            hand_midi_path = output_dir / f"hand_generated_{midi_path.stem}.mid"
            pitches_to_midi(hand_pitches, note_sequence, hand_midi_path)
            
            # Convert to WAV
            hand_wav_path = output_dir / f"hand_generated_{midi_path.stem}.wav"
            midi_to_wav(hand_midi_path, hand_wav_path, args.sf2_path)
            
            # Print comparison
            print(f"\nHand Model Results:")
            print(f"  Original pitches (first 20): {original_labels[:20].astype(int)}")
            print(f"  Generated pitches (first 20): {hand_pitches[:20] + MIN_PIANO_PITCH}")
            accuracy = np.mean(hand_pitches == (original_labels - MIN_PIANO_PITCH))
            print(f"  Reconstruction accuracy: {accuracy*100:.2f}%")
            
        except Exception as e:
            print(f"Hand model generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Hand checkpoint not found: {args.hand_checkpoint}")
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"\nOutput files in: {output_dir}")
    print(f"  - original_{midi_path.stem}.mid")
    print(f"  - original_{midi_path.stem}.wav")
    if Path(args.window_checkpoint).exists():
        print(f"  - window_generated_{midi_path.stem}.mid")
        print(f"  - window_generated_{midi_path.stem}.wav")
    if Path(args.hand_checkpoint).exists():
        print(f"  - hand_generated_{midi_path.stem}.mid")
        print(f"  - hand_generated_{midi_path.stem}.wav")


if __name__ == "__main__":
    main()

