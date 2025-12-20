## LIPSTICK/models/lipstick_model.py

"""LIPSTICK model with corruptibility awareness."""

import torch
import torch.nn as nn
from .gin_model import GIN


class LIPSTICKModel(nn.Module):
    """LIPSTICK: Corruptibility-aware GNN attack model."""
    
    def __init__(self, num_node_features, hidden_dim, key_size, num_layers=5, 
                 dropout=0.5, num_lock_types=7):
        super(LIPSTICKModel, self).__init__()
        
        self.key_size = key_size
        self.num_lock_types = num_lock_types
        
        # Base GIN network
        self.gin = GIN(
            num_features=num_node_features,
            hidden_dim=hidden_dim,
            num_classes=key_size,  # Predict key bits directly
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Additional head for lock type classification (auxiliary task)
        self.lock_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_lock_types)
        )
        
        # ER prediction head (auxiliary task for corruptibility awareness)
        self.er_predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # ER is in [0, 1]
        )
    
    def forward(self, x, edge_index, batch, return_all=False):
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            return_all: If True, return all predictions
        
        Returns:
            Key predictions (and optionally lock type and ER predictions)
        """
        # Get graph embeddings
        embeddings = self.gin.get_embeddings(x, edge_index, batch)
        
        # Key prediction (sigmoid for binary key bits)
        key_logits = self.gin.fc1(embeddings)
        key_logits = torch.nn.functional.leaky_relu(key_logits, negative_slope=0.01)
        key_logits = torch.nn.functional.dropout(key_logits, p=self.gin.dropout, training=self.training)
        key_pred = torch.sigmoid(self.gin.fc2(key_logits))
        
        if not return_all:
            return key_pred
        
        # Additional predictions
        lock_type = self.lock_classifier(embeddings)
        er_pred = self.er_predictor(embeddings)
        
        return key_pred, lock_type, er_pred
