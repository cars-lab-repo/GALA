## tests/test_gala.py


"""Unit tests for GALA."""

import pytest
import torch

import sys
sys.path.append('../GALA')
from models.gala_model import GALAModel
from models.gala_subgraph import GALASubgraphModel


def test_gala_graph_model():
    """Test GALA graph-level model."""
    model = GALAModel(
        num_node_features=14,
        hidden_dim=64,
        key_size=64,
        num_layers=5,
        dropout=0.5,
        use_power_area=True
    )
    
    # Create dummy data (with power/area features)
    x = torch.randn(100, 17)  # 14 + 3 for power/area
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)
    
    # Forward pass
    output = model(x, edge_index, batch)
    
    assert output.shape == (1, 64)
    assert torch.all((output >= 0) & (output <= 1))


def test_gala_subgraph_model():
    """Test GALA subgraph-level model."""
    model = GALASubgraphModel(
        num_node_features=14,
        hidden_dim=64,
        num_classes=2,
        num_layers=5,
        dropout=0.5,
        use_power_area=True
    )
    
    x = torch.randn(50, 17)
    edge_index = torch.randint(0, 50, (2, 100))
    batch = torch.zeros(50, dtype=torch.long)
    
    # Forward pass
    output = model(x, edge_index, batch)
    
    assert output.shape == (1, 2)  # Binary classification


def test_gala_multitask():
    """Test GALA multi-task outputs."""
    model = GALAModel(
        num_node_features=14,
        hidden_dim=64,
        key_size=64,
        num_layers=5,
        use_power_area=True
    )
    
    x = torch.randn(100, 17)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)
    
    outputs = model(x, edge_index, batch, return_all=True)
    
    assert 'key' in outputs
    assert 'lock_type' in outputs
    assert 'er' in outputs
    assert 'power' in outputs
    assert 'area' in outputs


if __name__ == '__main__':
    pytest.main([__file__])
