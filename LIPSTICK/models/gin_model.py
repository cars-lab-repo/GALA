## LIPSTICK/models/gin_model.py

"""Graph Isomorphism Network (GIN) implementation for LIPSTICK."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(nn.Module):
    """Graph Isomorphism Network as described in Xu et al. 2019."""
    
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=5, dropout=0.5):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.fc1 = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        """Forward pass through GIN.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Output predictions [batch_size, num_classes]
        """
        # Store layer representations for readout
        layer_outputs = []
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Aggregate for this layer
            pooled = global_add_pool(x, batch)
            layer_outputs.append(pooled)
        
        # Concatenate all layer outputs (jump knowledge)
        graph_repr = torch.cat(layer_outputs, dim=1)
        
        # Final classification
        x = self.fc1(graph_repr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def get_embeddings(self, x, edge_index, batch):
        """Get graph embeddings without final classification."""
        layer_outputs = []
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.01)
            pooled = global_add_pool(x, batch)
            layer_outputs.append(pooled)
        
        return torch.cat(layer_outputs, dim=1)
