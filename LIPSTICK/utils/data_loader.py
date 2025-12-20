
## LIPSTICK/utils/data_loader.py


"""Data loading utilities for LIPSTICK."""

import os
import torch
import pickle
import numpy as np
from torch_geometric.data import Data, Dataset
from .graph_utils import netlist_to_graph


class LIPSTICKDataset(Dataset):
    """Dataset for LIPSTICK model."""
    
    def __init__(self, root, split='train', key_size=64, transform=None, pre_transform=None):
        self.split = split
        self.key_size = key_size
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']
    
    def process(self):
        """Process raw netlists into graph data."""
        data_list = []
        
        # Load processed data if available
        processed_file = os.path.join(self.processed_dir, f'{self.split}_data.pkl')
        if os.path.exists(processed_file):
            with open(processed_file, 'rb') as f:
                data_list = pickle.load(f)
        else:
            # Process netlists
            netlists_dir = os.path.join(self.root, 'netlists', self.split)
            for netlist_file in os.listdir(netlists_dir):
                if not netlist_file.endswith('.v'):
                    continue
                
                # Parse netlist metadata from filename
                # Format: benchmark_locktype_keyvalue_er.v
                parts = netlist_file.replace('.v', '').split('_')
                benchmark = parts[0]
                lock_type = parts[1]
                key_str = parts[2]
                er = float(parts[3]) if len(parts) > 3 else 0.0
                
                # Convert to graph
                graph_data = netlist_to_graph(os.path.join(netlists_dir, netlist_file))
                
                # Add labels
                key_bits = [int(b) for b in key_str]
                graph_data.key = torch.tensor(key_bits, dtype=torch.float32)
                graph_data.er = torch.tensor([er], dtype=torch.float32)
                graph_data.lock_type = torch.tensor([self._lock_type_to_id(lock_type)], dtype=torch.long)
                
                data_list.append(graph_data)
            
            # Save processed data
            with open(processed_file, 'wb') as f:
                pickle.dump(data_list, f)
        
        torch.save(data_list, self.processed_paths[0])
    
    def len(self):
        data = torch.load(self.processed_paths[0])
        return len(data)
    
    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data[idx]
    
    @property
    def num_node_features(self):
        """Return number of node features."""
        data = self.get(0)
        return data.x.shape[1]
    
    def _lock_type_to_id(self, lock_type):
        """Convert lock type string to integer ID."""
        lock_types = {
            'xor': 0,
            'mux': 1,
            'lut': 2,
            'sar': 3,
            'antisat': 4,
            'ble': 5,
            'unsail': 6
        }
        return lock_types.get(lock_type.lower(), 0)
