
## GALA/train.py


"""Training script for GALA model."""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from models.gala_model import GALAModel
from models.gala_subgraph import GALASubgraphModel
from utils.data_loader import GALADataset, GALASubgraphDataset
from utils.metrics import compute_kpa, compute_kpr


def train_epoch_graph(model, loader, optimizer, criterion, device):
    """Train graph-level model for one epoch."""
    model.train()
    total_loss = 0
    total_kpa = 0
    num_batches = 0
    
    for data in tqdm(loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        
        outputs = model(data.x, data.edge_index, data.batch, return_all=True)
        
        # Multi-task loss
        key_loss = criterion(outputs['key'], data.key)
        lock_loss = nn.CrossEntropyLoss()(outputs['lock_type'], data.lock_type)
        er_loss = nn.MSELoss()(outputs['er'].squeeze(), data.er)
        
        loss = key_loss + 0.1 * lock_loss + 0.5 * er_loss
        
        # Add power and area losses if available
        if 'power' in outputs:
            power_loss = nn.MSELoss()(outputs['power'].squeeze(), data.power)
            area_loss = nn.MSELoss()(outputs['area'].squeeze(), data.area)
            loss = loss + 0.3 * power_loss + 0.3 * area_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        kpa = compute_kpa(outputs['key'], data.key)
        total_kpa += kpa
        num_batches += 1
    
    return total_loss / num_batches, total_kpa / num_batches


def train_epoch_subgraph(model, loader, optimizer, criterion, device):
    """Train subgraph-level model for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for data in tqdm(loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index, data.batch)
        loss = criterion(logits, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        acc = (pred == data.y).float().mean().item() * 100
        total_acc += acc
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def validate_graph(model, loader, criterion, device):
    """Validate graph-level model."""
    model.eval()
    total_loss = 0
    total_kpa = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc='Validation'):
            data = data.to(device)
            key_pred = model(data.x, data.edge_index, data.batch)
            loss = criterion(key_pred, data.key)
            
            total_loss += loss.item()
            kpa = compute_kpa(key_pred, data.key)
            total_kpa += kpa
            num_batches += 1
    
    return total_loss / num_batches, total_kpa / num_batches


def validate_subgraph(model, loader, criterion, device):
    """Validate subgraph-level model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc='Validation'):
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            acc = (pred == data.y).float().mean().item() * 100
            total_acc += acc
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train GALA model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, default='graph', choices=['graph', 'subgraph'])
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training {args.model}-level GALA model')
    
    # Create dataset
    if args.model == 'graph':
        train_dataset = GALADataset(
            root=config['data']['root'],
            split='train',
            key_size=config['data']['key_size'],
            use_power_area=config['model']['use_power_area']
        )
        val_dataset = GALADataset(
            root=config['data']['root'],
            split='val',
            key_size=config['data']['key_size'],
            use_power_area=config['model']['use_power_area']
        )
    else:
        train_dataset = GALASubgraphDataset(
            root=config['data']['root'],
            split='train',
            use_power_area=config['model']['use_power_area']
        )
        val_dataset = GALASubgraphDataset(
            root=config['data']['root'],
            split='val',
            use_power_area=config['model']['use_power_area']
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    if args.model == 'graph':
        model = GALAModel(
            num_node_features=train_dataset.num_node_features,
            hidden_dim=config['model']['hidden_dim'],
            key_size=config['data']['key_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            use_power_area=config['model']['use_power_area']
        ).to(device)
    else:
        model = GALASubgraphModel(
            num_node_features=train_dataset.num_node_features,
            hidden_dim=config['model']['hidden_dim'],
            num_classes=2,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            use_power_area=config['model']['use_power_area']
        ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < 100:
            return 1.0
        else:
            return 0.1
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Loss criterion
    if args.model == 'graph':
        criterion = nn.BCELoss()
        train_fn = train_epoch_graph
        val_fn = validate_graph
    else:
        criterion = nn.CrossEntropyLoss()
        train_fn = train_epoch_subgraph
        val_fn = validate_subgraph
    
    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config['training']['log_dir'], args.model))
    
    # Training loop
    best_val_metric = 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        train_loss, train_metric = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = val_fn(model, val_loader, criterion, device)
        
        scheduler.step()
        
        metric_name = 'KPA' if args.model == 'graph' else 'Accuracy'
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar(f'{metric_name}/train', train_metric, epoch)
        writer.add_scalar(f'{metric_name}/val', val_metric, epoch)
        
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.2f}%')
        
        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'val_{metric_name.lower()}': val_metric,
                'config': config,
                'model_type': args.model
            }, os.path.join(config['training']['checkpoint_dir'], f'best_{args.model}_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if train_loss > 1.0:
            print(f'Loss explosion at epoch {epoch+1}')
            break
    
    writer.close()
    print(f'Training completed. Best validation {metric_name}: {best_val_metric:.2f}%')


if __name__ == '__main__':
    main()
