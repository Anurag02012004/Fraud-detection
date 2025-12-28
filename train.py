"""
Training script for Advanced GNN Fraud Detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import os

from src.data_loader import FraudDatasetGenerator, create_homogeneous_graph
from src.model import AdvancedGCN, AdvancedGAT, EnsembleGNN
from src.utils import calculate_metrics, set_seed, plot_confusion_matrix, plot_roc_curve


def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Calculate metrics on training set
    with torch.no_grad():
        train_preds = out[data.train_mask].argmax(dim=1).cpu().numpy()
        train_probas = torch.exp(out[data.train_mask, 1]).cpu().numpy()
        train_labels = data.y[data.train_mask].cpu().numpy()
    
    metrics = calculate_metrics(train_labels, train_preds, train_probas)
    return loss.item(), metrics


def evaluate(model, data, criterion, device, mask):
    """Evaluate model"""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask])
        
        preds = out[mask].argmax(dim=1).cpu().numpy()
        probas = torch.exp(out[mask, 1]).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
    
    metrics = calculate_metrics(labels, preds, probas)
    return loss.item(), metrics, labels, preds, probas


def main():
    parser = argparse.ArgumentParser(description='Train GNN for Fraud Detection')
    parser.add_argument('--model', type=str, default='gat', choices=['gcn', 'gat', 'ensemble'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='models')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Generate data
    print('Generating synthetic fraud dataset...')
    generator = FraudDatasetGenerator(
        n_users=1000,
        n_accounts=1500,
        n_transactions=10000,
        fraud_ratio=0.1,
        seed=args.seed
    )
    
    hetero_data, transactions_df = generator.generate_transaction_graph()
    
    # Convert to homogeneous graph
    data = create_homogeneous_graph(hetero_data)
    
    # Split data
    train_idx, val_idx, test_idx = generator.get_train_test_split(hetero_data)
    
    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    data = data.to(device)
    
    # Initialize model
    input_dim = data.x.size(1)
    num_classes = 2
    
    if args.model == 'gcn':
        model = AdvancedGCN(input_dim, hidden_dims=[128, 64, 32], num_classes=num_classes).to(device)
    elif args.model == 'gat':
        model = AdvancedGAT(input_dim, hidden_dims=[128, 64, 32], num_classes=num_classes).to(device)
    else:
        model = EnsembleGNN(input_dim, num_classes=num_classes).to(device)
    
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Train nodes: {data.train_mask.sum().item()}, Val nodes: {data.val_mask.sum().item()}, Test nodes: {data.test_mask.sum().item()}')
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_aucs = []
    
    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(model, data, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics, _, _, _ = evaluate(model, data, criterion, device, data.val_mask)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_metrics['auc_roc'])
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train AUC: {train_metrics["auc_roc"]:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val AUC: {val_metrics["auc_roc"]:.4f}')
        
        # Save best model
        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc = val_metrics['auc_roc']
            best_epoch = epoch
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_auc': best_val_auc,
            }, os.path.join(args.save_dir, f'best_{args.model}.pth'))
    
    print(f'\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch}')
    
    # Load best model and test
    checkpoint = torch.load(os.path.join(args.save_dir, f'best_{args.model}.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('\nEvaluating on test set...')
    test_loss, test_metrics, test_labels, test_preds, test_probas = evaluate(model, data, criterion, device, data.test_mask)
    
    print('\nTest Results:')
    print(f'AUC-ROC: {test_metrics["auc_roc"]:.4f}')
    print(f'Precision: {test_metrics["precision"]:.4f}')
    print(f'Recall: {test_metrics["recall"]:.4f}')
    print(f'F1-Score: {test_metrics["f1"]:.4f}')
    print(f'PR-AUC: {test_metrics["pr_auc"]:.4f}')
    
    # Save plots
    os.makedirs('plots', exist_ok=True)
    plot_confusion_matrix(np.array(test_labels), np.array(test_preds), 
                         save_path='plots/confusion_matrix.png')
    plot_roc_curve(np.array(test_labels), np.array(test_probas), 
                   save_path='plots/roc_curve.png')
    
    print('\nTraining complete!')


if __name__ == '__main__':
    main()

