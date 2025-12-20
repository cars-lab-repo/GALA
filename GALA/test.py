## GALA/test.py


"""Testing script for GALA model."""

import os
import yaml
import torch
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.gala_model import GALAModel
from models.gala_subgraph import GALASubgraphModel
from utils.data_loader import GALADataset, GALASubgraphDataset
from utils.metrics import compute_kpa


def test_graph_model(model, loader, device):
    """Test graph-level model."""
    model.eval()
    
    all_kpa = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc='Testing'):
            data = data.to(device)
            key_pred = model(data.x, data.edge_index, data.batch)
            
            kpa = compute_kpa(key_pred, data.key)
            all_kpa.append(kpa)
            
            all_predictions.append(key_pred.cpu())
            all_targets.append(data.key.cpu())
    
    mean_kpa = np.mean(all_kpa)
    std_kpa = np.std(all_kpa)
    
    return {
        'mean_kpa': mean_kpa,
        'std_kpa': std_kpa,
        'predictions': torch.cat(all_predictions, dim=0),
        'targets': torch.cat(all_targets, dim=0)
    }


def test_subgraph_model(model, loader, device):
    """Test subgraph-level model."""
    model.eval()
    
    all_acc = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc='Testing'):
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            
            pred = logits.argmax(dim=1)
            acc = (pred == data.y).float().mean().item() * 100
            all_acc.append(acc)
    
    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }


def main():
    parser = argparse.ArgumentParser(description='Test GALA model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, default='graph', choices=['graph', 'subgraph'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Testing {args.model}-level GALA model')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create dataset
    if args.model == 'graph':
        test_dataset = GALADataset(
            root=config['data']['root'],
            split=args.split,
            key_size=config['data']['key_size'],
            use_power_area=config['model']['use_power_area']
        )
    else:
        test_dataset = GALASubgraphDataset(
            root=config['data']['root'],
            split=args.split,
            use_power_area=config['model']['use_power_area']
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    if args.model == 'graph':
        model = GALAModel(
            num_node_features=test_dataset.num_node_features,
            hidden_dim=config['model']['hidden_dim'],
            key_size=config['data']['key_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            use_power_area=config['model']['use_power_area']
        ).to(device)
    else:
        model = GALASubgraphModel(
            num_node_features=test_dataset.num_node_features,
            hidden_dim=config['model']['hidden_dim'],
            num_classes=2,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            use_power_area=config['model']['use_power_area']
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print('Testing model...')
    if args.model == 'graph':
        results = test_graph_model(model, test_loader, device)
        print(f'\nTest Results:')
        print(f'  Mean KPA: {results["mean_kpa"]:.2f}% ± {results["std_kpa"]:.2f}%')
    else:
        results = test_subgraph_model(model, test_loader, device)
        print(f'\nTest Results:')
        print(f'  Mean Accuracy: {results["mean_accuracy"]:.2f}% ± {results["std_accuracy"]:.2f}%')
    
    print(f'  Checkpoint epoch: {checkpoint["epoch"]}')


if __name__ == '__main__':
    main()
