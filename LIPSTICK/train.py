
## LIPSTICK/train.py


"""Training script for LIPSTICK model."""

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

from models.lipstick_model import LIPSTICKModel
from utils.data_loader import LIPSTICKDataset
from utils.metrics import compute_kpa, compute_kpr


def train_epoch(model, loader, optimizer, criterion, device, use_er=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_kpa = 0
    num_batches = 0
    
    for data in tqdm(loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        
        if use_er:
            key_pred, lock_pred, er_pred = model(data.x, data.edge_index, data.batch, return_all=True)
            
            # Multi-task loss
            key_loss = criterion(key_pred, data.key)
            lock_loss = nn.CrossEntropyLoss()(lock_pred, data.lock_type)
            er_loss = nn.MSELoss()(er_pred.squeeze(), data.er)
            
            loss = key_loss + 0.1 * lock_loss + 0.5 * er_loss
        else:
            key_pred = model(data.x, data.edge_index, data.batch)
            loss = criterion(key_pred, data.key)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        kpa = compute_kpa(key_pred, data.key)
        total_kpa += kpa
        num_batches += 1
    
    return total_loss / num_batches, total_kpa / num_batches


def validate(model, loader, criterion, device):
    """Validate the model."""
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


def main():
    parser = argparse.ArgumentParser(description='Train LIPSTICK model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    train_dataset = LIPSTICKDataset(
        root=config['data']['root'],
        split='train',
        key_size=config['data']['key_size']
    )
    val_dataset = LIPSTICKDataset(
        root=config['data']['root'],
        split='val',
        key_size=config['data']['key_size']
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
    model = LIPSTICKModel(
        num_node_features=train_dataset.num_node_features,
        hidden_dim=config['model']['hidden_dim'],
        key_size=config['data']['key_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Optimizer with scheduled learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler: 0.1 for first 100 epochs, then 0.01
    def lr_lambda(epoch):
        if epoch < 100:
            return 1.0
        else:
            return 0.1
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Loss criterion
    criterion = nn.BCELoss()
    
    # Tensorboard
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # Training loop
    best_val_kpa = 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        train_loss, train_kpa = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_er=config['model'].get('use_er', True)
        )
        val_loss, val_kpa = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('KPA/train', train_kpa, epoch)
        writer.add_scalar('KPA/val', val_kpa, epoch)
        
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train KPA: {train_kpa:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val KPA: {val_kpa:.2f}%')
        
        # Save best model
        if val_kpa > best_val_kpa:
            best_val_kpa = val_kpa
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kpa': val_kpa,
                'config': config
            }, os.path.join(config['training']['checkpoint_dir'], 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        # Check for loss explosion
        if train_loss > 1.0:
            print(f'Loss explosion detected at epoch {epoch+1}. Stopping.')
            break
    
    writer.close()
    print(f'Training completed. Best validation KPA: {best_val_kpa:.2f}%')


if __name__ == '__main__':
    main()
