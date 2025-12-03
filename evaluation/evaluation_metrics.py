import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate_teacher_forcing(model, dataloader, device, feature='unconditional', return_sequences=False, return_features=False):
    model.eval()
    all_logits = []
    all_labels = []
    predicted_sequences = []
    ground_truth_sequences = []
    feature_sequences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Teacher forcing eval"):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            pitch_logits, _ = model(features)
            all_logits.append(pitch_logits.cpu())
            all_labels.append(labels.cpu())
            
            if return_sequences or return_features:
                top1_predictions = torch.argmax(pitch_logits, dim=-1).cpu()
                pad_mask = labels != 88
                for b in range(labels.shape[0]):
                    mask = pad_mask[b].bool()
                    if return_sequences:
                        pred_seq = top1_predictions[b][mask].numpy().tolist()
                        gt_seq = labels[b][mask].numpy().tolist()
                        predicted_sequences.append(pred_seq)
                        ground_truth_sequences.append(gt_seq)
                    if return_features:
                        feat_seq = features[b][mask].cpu().numpy()
                        feature_sequences.append(feat_seq)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    pad_mask = all_labels != 88
    B, L, C = all_logits.shape
    logits_flat = all_logits.reshape(B * L, C)
    labels_flat = all_labels.reshape(B * L).long()
    
    ce = F.cross_entropy(logits_flat, labels_flat, reduction='mean', ignore_index=88)
    ppl = torch.exp(ce).item()
    ce = ce.item()
    
    top_k_accs = {}
    for k in [1, 3, 5, 10]:
        top_k = torch.topk(all_logits, k, dim=-1).indices
        hits = (top_k == all_labels.unsqueeze(-1)).any(dim=-1).float() * pad_mask.float()
        mask = pad_mask.bool()
        hits_masked = hits[mask]
        acc = hits_masked.sum().item() / (mask.sum().item() + 1e-8)
        top_k_accs[f'top{k}_acc'] = acc
    
    results = {
        'cross_entropy': ce,
        'perplexity': ppl,
        **top_k_accs
    }
    
    if return_sequences:
        results['predicted_sequences'] = predicted_sequences
        results['ground_truth_sequences'] = ground_truth_sequences
    if return_features:
        results['feature_sequences'] = feature_sequences
    
    return results


def evaluate_ancestral_sampling(model, dataloader, device, feature='unconditional', temperature=1.0, max_length=None, return_features=False):
    model.eval()
    all_generated = []
    all_ground_truth = []
    feature_sequences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Ancestral sampling eval (temp={temperature})"):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            B, L, _ = features.shape
            if max_length:
                L = min(L, max_length, L)
            
            generated_sequences = []
            ground_truth_sequences = []
            batch_feature_sequences = []
            
            for b in range(B):
                hx = None
                generated = []
                gt_sequence = []
                
                if return_features:
                    pad_mask = labels[b] != 88
                    orig_features = features[b].clone()
                    feat_seq = orig_features[pad_mask].cpu().numpy()
                    batch_feature_sequences.append(feat_seq)
                
                first_input = features[b, 0:1, :].unsqueeze(0)
                pitch_logits, hx = model(first_input, hx=hx)
                
                pitch_prob = F.softmax(pitch_logits / temperature, dim=-1)
                first_pitch = torch.multinomial(pitch_prob.squeeze(0), num_samples=1).item()
                generated.append(first_pitch)
                gt_sequence.append(labels[b, 0].item())
                
                prev_pitch = first_pitch
                for t in range(1, L):
                    step_input = features[b, t:t+1, :].clone()
                    step_input[0, 0] = prev_pitch
                    step_input = step_input.unsqueeze(0)
                    
                    pitch_logits, hx = model(step_input, hx=hx)
                    pitch_prob = F.softmax(pitch_logits / temperature, dim=-1)
                    pitch = torch.multinomial(pitch_prob.squeeze(0), num_samples=1).item()
                    
                    generated.append(pitch)
                    gt_sequence.append(labels[b, t].item())
                    prev_pitch = pitch
                
                generated_sequences.append(generated)
                ground_truth_sequences.append(gt_sequence)
            
            all_generated.extend(generated_sequences)
            all_ground_truth.extend(ground_truth_sequences)
            if return_features:
                feature_sequences.extend(batch_feature_sequences)
    
    generated_array = np.array([seq for seq in all_generated])
    gt_array = np.array([seq for seq in all_ground_truth])
    matches = (generated_array == gt_array).astype(float)
    accuracy = matches.mean()
    
    results = {
        'ancestral_accuracy': accuracy,
        'generated_sequences': all_generated,
        'ground_truth_sequences': all_ground_truth
    }
    
    if return_features:
        results['feature_sequences'] = feature_sequences
    
    return results

