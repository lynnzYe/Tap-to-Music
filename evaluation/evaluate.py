
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.data_utils import create_dataloader_from_pickle
from evaluation.evaluation_metrics import (
    evaluate_ancestral_sampling,
    evaluate_teacher_forcing,
)
from evaluation.model_utils import load_model_from_checkpoint
from evaluation.visualization import compute_pitch_statistics, plot_pitch_histograms


def evaluate_model(
    checkpoint_path,
    dataset_pickle_path,
    output_dir,
    feature_type='unconditional',
    device='cpu',
    temperatures=[1.0, 1.2, 1.5],
    split='test'
):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, feature_type, None, device)
    
    print(f"Loading dataset from {dataset_pickle_path}")
    dataloader = create_dataloader_from_pickle(
        dataset_pickle_path, feature_type, 64, 1, split=split
    )
    
    results = {}
    
    print("Evaluating with teacher forcing...")
    tf_results = evaluate_teacher_forcing(
        model, dataloader, device, feature_type, 
        return_sequences=True,
        return_features=False
    )
    
    tf_metrics = {k: v for k, v in tf_results.items() 
                 if k not in ['predicted_sequences', 'ground_truth_sequences']}
    results['teacher_forcing'] = tf_metrics
    
    print(f"Teacher Forcing - CE: {tf_metrics['cross_entropy']:.4f}, "
          f"PPL: {tf_metrics['perplexity']:.4f}, "
          f"Top-1: {tf_metrics['top1_acc']:.4f}, "
          f"Top-5: {tf_metrics['top5_acc']:.4f}")
    
    if 'predicted_sequences' in tf_results:
        print("Computing pitch statistics for teacher forcing...")
        pred_stats = compute_pitch_statistics(
            tf_results['predicted_sequences'],
            name='predicted_tf'
        )
        gt_stats = compute_pitch_statistics(
            tf_results['ground_truth_sequences'],
            name='ground_truth_tf'
        )
        
        if pred_stats and gt_stats:
            results['teacher_forcing']['predicted_stats'] = pred_stats
            results['teacher_forcing']['ground_truth_stats'] = gt_stats
            plot_pitch_histograms(
                gt_stats, pred_stats, output_dir,
                name='teacher_forcing'
            )
            print("Teacher forcing histograms saved.")
    
    ancestral_results = {}
    for temp in temperatures:
        print(f"Evaluating with ancestral sampling (temperature={temp})...")
        as_results = evaluate_ancestral_sampling(
            model, dataloader, device, feature_type, temperature=temp,
            return_features=False
        )
        ancestral_results[f'temp_{temp}'] = {
            'accuracy': as_results['ancestral_accuracy']
        }
        
        gen_stats = compute_pitch_statistics(
            as_results['generated_sequences'],
            name=f'generated_temp{temp}'
        )
        gt_stats = compute_pitch_statistics(
            as_results['ground_truth_sequences'],
            name='ground_truth'
        )
        ancestral_results[f'temp_{temp}']['generated_stats'] = gen_stats
        ancestral_results[f'temp_{temp}']['ground_truth_stats'] = gt_stats
        
        if gen_stats and gt_stats:
            plot_pitch_histograms(
                gt_stats, gen_stats, output_dir,
                name=f'temp{temp}'
            )
    
    results['ancestral_sampling'] = ancestral_results
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj.tolist()
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=default_serializer)
    
    print(f"\nEvaluation results saved to {results_path}")
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("\nTeacher Forcing:")
    for k, v in results['teacher_forcing'].items():
        print(f"  {k}: {v:.4f}")
    print("\nAncestral Sampling:")
    for temp_key, temp_results in results['ancestral_sampling'].items():
        print(f"  {temp_key}:")
        for k, v in temp_results.items():
            print(f"    {k}: {v:.4f}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluation for Tap-to-Music models')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to directory containing pickle files (e.g., /path/to/data/)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'],
                       help='Dataset split to use (will look for {feature}-{split}.pkl)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--feature', type=str, required=True, choices=['unconditional', 'chord'],
                       help='Feature type: unconditional or chord')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1.0, 1.2, 1.5],
                       help='Temperatures for ancestral sampling')
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset_pickle_path=args.dataset,
        output_dir=args.output_dir,
        feature_type=args.feature,
        device=args.device,
        temperatures=args.temperatures,
        split=args.split
    )


if __name__ == "__main__":
    main()
