## tests/test_lipstick.py

"""Unit tests for LIPSTICK."""

import pytest
import torch
from torch_geometric.data import Data

import sys
sys.path.append('../LIPSTICK')
from models.lipstick_model import LIPSTICKModel
from utils.metrics import compute_kpa, compute_hamming_distance


def test_lipstick_model():
    """Test LIPSTICK model creation and forward pass."""
    model = LIPSTICKModel(
        num_node_features=14,
        hidden_dim=64,
        key_size=64,
        num_layers=5,
        dropout=0.5
    )
    
    # Create dummy data
    x = torch.randn(100, 14)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)
    
    # Forward pass
    output = model(x, edge_index, batch)
    
    assert output.shape == (1, 64)
    assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output


def test_kpa_metric():
    """Test KPA computation."""
    pred_key = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    true_key = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    
    kpa = compute_kpa(pred_key, true_key, threshold=0.5)
    
    assert kpa == 100.0  # All bits correct


def test_hamming_distance():
    """Test Hamming distance computation."""
    key1 = torch.tensor([0, 1, 0, 1])
    key2 = torch.tensor([0, 0, 0, 1])
    
    hd = compute_hamming_distance(key1, key2)
    
    assert hd == 1  # One bit different


if __name__ == '__main__':
    pytest.main([__file__])
