## GALA/models/gala_subgraph.py


"""GALA subgraph-level attack (enhanced OMLA)."""

import torch
import torch.nn as nn
from .gin_model import GIN


class GALASubgraphModel(nn.Module):
    """GALA subgraph-level model (OMLA with power/area features)."""
    
    def __init__(self, num_node_features, hidden_dim, num_classes=2, num_layers=5, 
                 dropout=0.5, use_power_area=True):
        super(GALASubgraphModel, self).__init__()
        
        self.use_power_area = use_power_area
        
        # Adjust input features
        input_features = num_node_features + (3 if use_power_area else 0)
        
        # Base GIN for subgraph classification
        self.gin = GIN(
            num_features=input_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,  # Binary classification for key bit
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x, edge_index, batch):
        """Forward pass for subgraph classification.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
        
        Returns:
            Key bit prediction [batch_size, 2] (logits for 0/1)
        """
        return self.gin(x, edge_index, batch)
