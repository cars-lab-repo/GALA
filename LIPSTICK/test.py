
## LIPSTICK/test.py


"""Testing script for LIPSTICK model."""

import os
import yaml
import torch
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.lipstick_model import LIPSTICKModel
from utils.data_loader import LIPSTICKDataset
from utils.metrics import compute_kpa, compute_kpr


def test_model(model, loader, device):
    """Test the model and compute metrics."""
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
    
    # Aggregate results
    mean_kpa = np.mean(all_kpa)
    std_kpa = np.std(all_kpa)
    
    return {
        'mean_kpa': mean_kpa,
        'std_kpa': std_kpa,
        'predictions': torch.cat(all_predictions, dim=0),
        'targets': torch.cat(all_targets, dim=0)
    }


def main():
    parser = argparse.ArgumentParser(description='Test LIPSTICK model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create dataset
    test_dataset = LIPSTICKDataset(
        root=config['data']['root'],
        split=args.split,
        key_size=config['data']['key_size']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = LIPSTICKModel(
        num_node_features=test_dataset.num_node_features,
        hidden_dim=config['model']['hidden_dim'],
        key_size=config['data']['key_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print('Testing model...')
    results = test_model(model, test_loader, device)
    
    print(f'\nTest Results:')
    print(f'  Mean KPA: {results["mean_kpa"]:.2f}% ± {results["std_kpa"]:.2f}%')
    print(f'  Checkpoint epoch: {checkpoint["epoch"]}')
    print(f'  Checkpoint val KPA: {checkpoint["val_kpa"]:.2f}%')


if __name__ == '__main__':
    main()
