
## GALA/utils/data_loader.py


"""Data loading utilities for GALA."""

import os
import torch
import pickle
import numpy as np
from torch_geometric.data import Data, Dataset
from .graph_utils import netlist_to_graph_with_features, extract_subgraphs


class GALADataset(Dataset):
    """Dataset for GALA graph-level model."""
    
    def __init__(self, root, split='train', key_size=64, use_power_area=True, 
                 transform=None, pre_transform=None):
        self.split = split
        self.key_size = key_size
        self.use_power_area = use_power_area
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        suffix = '_power_area' if self.use_power_area else ''
        return [f'{self.split}_data{suffix}.pt']
    
    def process(self):
        """Process raw netlists into graph data with power/area features."""
        data_list = []
        
        processed_file = os.path.join(self.processed_dir, f'{self.split}_data.pkl')
        if os.path.exists(processed_file):
            with open(processed_file, 'rb') as f:
                data_list = pickle.load(f)
        else:
            netlists_dir = os.path.join(self.root, 'netlists', self.split)
            features_dir = os.path.join(self.root, 'features', self.split) if self.use_power_area else None
            
            for netlist_file in os.listdir(netlists_dir):
                if not netlist_file.endswith('.v'):
                    continue
                
                # Parse metadata
                parts = netlist_file.replace('.v', '').split('_')
                benchmark = parts[0]
                lock_type = parts[1]
                key_str = parts[2]
                er = float(parts[3]) if len(parts) > 3 else 0.0
                
                # Convert to graph with features
                netlist_path = os.path.join(netlists_dir, netlist_file)
                
                if self.use_power_area:
                    feature_file = os.path.join(features_dir, netlist_file.replace('.v', '.feat'))
                    graph_data = netlist_to_graph_with_features(netlist_path, feature_file)
                    
                    # Add circuit-level power and area labels
                    if os.path.exists(feature_file):
                        with open(feature_file, 'r') as f:
                            lines = f.readlines()
                            total_power = float(lines[0].split(':')[1].strip())
                            total_area = float(lines[1].split(':')[1].strip())
                        graph_data.power = torch.tensor([total_power], dtype=torch.float32)
                        graph_data.area = torch.tensor([total_area], dtype=torch.float32)
                else:
                    from .graph_utils import netlist_to_graph
                    graph_data = netlist_to_graph(netlist_path)
                
                # Add labels
                key_bits = [int(b) for b in key_str]
                graph_data.key = torch.tensor(key_bits, dtype=torch.float32)
                graph_data.er = torch.tensor([er], dtype=torch.float32)
                graph_data.lock_type = torch.tensor([self._lock_type_to_id(lock_type)], dtype=torch.long)
                
                data_list.append(graph_data)
            
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
        data = self.get(0)
        return data.x.shape[1]
    
    def _lock_type_to_id(self, lock_type):
        lock_types = {
            'xor': 0, 'mux': 1, 'lut': 2, 'sar': 3,
            'antisat': 4, 'ble': 5, 'unsail': 6
        }
        return lock_types.get(lock_type.lower(), 0)


class GALASubgraphDataset(Dataset):
    """Dataset for GALA subgraph-level model (enhanced OMLA)."""
    
    def __init__(self, root, split='train', h_hops=2, use_power_area=True,
                 transform=None, pre_transform=None):
        self.split = split
        self.h_hops = h_hops
        self.use_power_area = use_power_area
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        suffix = '_power_area' if self.use_power_area else ''
        return [f'{self.split}_subgraph_data{suffix}.pt']
    
    def process(self):
        """Process netlists into subgraphs centered on key gates."""
        data_list = []
        
        processed_file = os.path.join(self.processed_dir, f'{self.split}_subgraph_data.pkl')
        if os.path.exists(processed_file):
            with open(processed_file, 'rb') as f:
                data_list = pickle.load(f)
        else:
            netlists_dir = os.path.join(self.root, 'netlists', self.split)
            features_dir = os.path.join(self.root, 'features', self.split) if self.use_power_area else None
            
            for netlist_file in os.listdir(netlists_dir):
                if not netlist_file.endswith('.v'):
                    continue
                
                # Parse metadata
                parts = netlist_file.replace('.v', '').split('_')
                key_str = parts[2]
                key_bits = [int(b) for b in key_str]
                
                netlist_path = os.path.join(netlists_dir, netlist_file)
                feature_file = os.path.join(features_dir, netlist_file.replace('.v', '.feat')) if self.use_power_area else None
                
                # Extract subgraphs for each key gate
                subgraphs = extract_subgraphs(netlist_path, self.h_hops, feature_file)
                
                for i, subgraph in enumerate(subgraphs):
                    if i < len(key_bits):
                        subgraph.y = torch.tensor([key_bits[i]], dtype=torch.long)
                        data_list.append(subgraph)
            
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
        data = self.get(0)
        return data.x.shape[1]
