## GALA/models/gala_model.py


"""GALA model with power and area features."""

import torch
import torch.nn as nn
from .gin_model import GIN


class GALAModel(nn.Module):
    """GALA: GNN attack with behavioral and functional features."""
    
    def __init__(self, num_node_features, hidden_dim, key_size, num_layers=5, 
                 dropout=0.5, num_lock_types=7, use_power_area=True):
        super(GALAModel, self).__init__()
        
        self.key_size = key_size
        self.num_lock_types = num_lock_types
        self.use_power_area = use_power_area
        
        # Adjust input features if using power/area
        input_features = num_node_features + (3 if use_power_area else 0)  # +3 for power, area features
        
        # Base GIN network
        self.gin = GIN(
            num_features=input_features,
            hidden_dim=hidden_dim,
            num_classes=key_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Additional head for lock type classification
        self.lock_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_lock_types)
        )
        
        # ER prediction head
        self.er_predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Power prediction head (auxiliary task)
        if use_power_area:
            self.power_predictor = nn.Sequential(
                nn.Linear(hidden_dim * num_layers, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            
            self.area_predictor = nn.Sequential(
                nn.Linear(hidden_dim * num_layers, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, x, edge_index, batch, return_all=False):
        """Forward pass.
        
        Args:
            x: Node features (with power/area if use_power_area=True)
            edge_index: Edge connectivity
            batch: Batch assignment
            return_all: If True, return all predictions
        
        Returns:
            Key predictions (and optionally all auxiliary predictions)
        """
        # Get graph embeddings
        embeddings = self.gin.get_embeddings(x, edge_index, batch)
        
        # Key prediction
        key_logits = self.gin.fc1(embeddings)
        key_logits = torch.nn.functional.leaky_relu(key_logits, negative_slope=0.01)
        key_logits = torch.nn.functional.dropout(key_logits, p=self.gin.dropout, training=self.training)
        key_pred = torch.sigmoid(self.gin.fc2(key_logits))
        
        if not return_all:
            return key_pred
        
        # Additional predictions
        lock_type = self.lock_classifier(embeddings)
        er_pred = self.er_predictor(embeddings)
        
        outputs = {
            'key': key_pred,
            'lock_type': lock_type,
            'er': er_pred
        }
        
        if self.use_power_area:
            power_pred = self.power_predictor(embeddings)
            area_pred = self.area_predictor(embeddings)
            outputs['power'] = power_pred
            outputs['area'] = area_pred
        
        return outputs
