
## LIPSTICK/utils/metrics.py


"""Evaluation metrics for LIPSTICK."""

import torch
import numpy as np


def compute_kpa(pred_key, true_key, threshold=0.5):
    """Compute Key Prediction Accuracy (KPA).
    
    Args:
        pred_key: Predicted key tensor [batch_size, key_size]
        true_key: True key tensor [batch_size, key_size]
        threshold: Threshold for binarizing predictions
    
    Returns:
        KPA as percentage
    """
    # Binarize predictions
    pred_binary = (pred_key > threshold).float()
    
    # Compute accuracy
    correct = (pred_binary == true_key).float()
    kpa = correct.mean().item() * 100
    
    return kpa


def compute_kpr(pred_key, circuit_func, input_patterns, true_key, threshold=0.5):
    """Compute Key Precision Rate (KPR).
    
    Args:
        pred_key: Predicted key tensor [batch_size, key_size]
        circuit_func: Function that evaluates circuit with given key
        input_patterns: Test input patterns
        true_key: True key for reference
        threshold: Threshold for binarizing predictions
    
    Returns:
        KPR as percentage
    """
    # Binarize predictions
    pred_binary = (pred_key > threshold).float()
    
    # Evaluate circuit with predicted key
    correct_outputs = 0
    total_patterns = len(input_patterns)
    
    for pattern in input_patterns:
        pred_output = circuit_func(pattern, pred_binary)
        true_output = circuit_func(pattern, true_key)
        
        if torch.all(pred_output == true_output):
            correct_outputs += 1
    
    kpr = (correct_outputs / total_patterns) * 100
    return kpr


def compute_hamming_distance(key1, key2):
    """Compute Hamming distance between two keys.
    
    Args:
        key1: First key tensor
        key2: Second key tensor
    
    Returns:
        Hamming distance
    """
    return torch.sum(key1 != key2).item()


def compute_error_rate(pred_key, circuit_func, input_patterns, true_circuit_func, threshold=0.5):
    """Compute Error Rate (ER).
    
    Args:
        pred_key: Predicted key
        circuit_func: Locked circuit function
        input_patterns: Test patterns
        true_circuit_func: Original circuit function
        threshold: Binarization threshold
    
    Returns:
        Error rate as float in [0, 1]
    """
    pred_binary = (pred_key > threshold).float()
    
    errors = 0
    for pattern in input_patterns:
        pred_output = circuit_func(pattern, pred_binary)
        true_output = true_circuit_func(pattern)
        
        if not torch.all(pred_output == true_output):
            errors += 1
    
    return errors / len(input_patterns)