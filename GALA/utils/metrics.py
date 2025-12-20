"""Metrics (generic placeholders)."""

import torch

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item() * 100.0
