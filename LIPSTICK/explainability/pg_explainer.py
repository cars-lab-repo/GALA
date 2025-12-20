## LIPSTICK/explainability/pg_explainer.py


"""PGExplainer implementation for LIPSTICK and GALA."""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class PGExplainer(nn.Module):
    """Parameterized Graph Explainer for GNN models."""
    
    def __init__(self, model, num_layers, device, explain_graph=True):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.device = device
        self.explain_graph = explain_graph
        
        # Explanation network (learns to predict edge importance)
        self.elayers = nn.ModuleList()
        for _ in range(num_layers):
            self.elayers.append(
                nn.Sequential(
                    nn.Linear(model.gin.convs[0].nn[0].in_features * 2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            )
    
    def forward(self, x, edge_index, batch=None):
        """Generate explanations.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment (for graph-level)
        
        Returns:
            Edge importance scores
        """
        # Get node embeddings from model
        edge_imp = torch.zeros(edge_index.shape[1]).to(self.device)
        
        for i in range(self.num_layers):
            # Get source and target node features
            row, col = edge_index
            edge_feat = torch.cat([x[row], x[col]], dim=1)
            
            # Predict edge importance
            imp = self.elayers[i](edge_feat).squeeze()
            edge_imp += imp
        
        return edge_imp / self.num_layers
    
    def train_explainer(self, data_loader, num_epochs=30, lr=0.001):
        """Train the explainer network.
        
        Args:
            data_loader: DataLoader with graphs
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        self.model.eval()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for data in data_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # Get model predictions
                with torch.no_grad():
                    pred = self.model(data.x, data.edge_index, data.batch)
                
                # Get edge importances
                edge_imp = self.forward(data.x, data.edge_index, data.batch)
                
                # Create masked graph
                mask = edge_imp > 0.5
                masked_edge_index = data.edge_index[:, mask]
                
                # Predict with masked graph
                masked_pred = self.model(data.x, masked_edge_index, data.batch)
                
                # Loss: maintain prediction with important edges
                pred_loss = nn.MSELoss()(masked_pred, pred)
                
                # Sparsity loss: encourage few important edges
                sparsity_loss = edge_imp.mean()
                
                # Total loss
                loss = pred_loss + 0.1 * sparsity_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Explainer Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}')
    
    def explain_graph(self, data, threshold=0.5):
        """Explain a single graph.
        
        Args:
            data: PyG Data object
            threshold: Threshold for important edges
        
        Returns:
            Dictionary with explanation
        """
        self.eval()
        with torch.no_grad():
            edge_imp = self.forward(data.x, data.edge_index, data.batch)
        
        # Get important edges
        important_edges = edge_imp > threshold
        important_edge_index = data.edge_index[:, important_edges]
        important_scores = edge_imp[important_edges]
        
        return {
            'edge_importance': edge_imp.cpu().numpy(),
            'important_edges': important_edge_index.cpu().numpy(),
            'important_scores': important_scores.cpu().numpy(),
            'threshold': threshold
        }
    
    def visualize(self, data, explanation, save_path=None):
        """Visualize explanation on graph.
        
        Args:
            data: PyG Data object
            explanation: Explanation dictionary
            save_path: Path to save visualization
        """
        # Convert to NetworkX
        G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            G.add_edge(src, dst, weight=explanation['edge_importance'][i])
        
        # Layout
        pos = nx.spring_layout(G, seed=42)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray')
        
        # Draw edges with importance coloring
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for coloring
        vmin, vmax = min(weights), max(weights)
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=weights,
            width=2.0,
            edge_cmap=plt.cm.YlOrRd,
            edge_vmin=vmin,
            edge_vmax=vmax
        )
        
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd), 
                    label='Edge Importance')
        plt.title('Graph Explanation (warmer = more important)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Saved visualization to {save_path}')
        else:
            plt.show()
        
        plt.close()
