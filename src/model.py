"""
Advanced GNN architectures for fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import HeteroConv, Linear
from torch_geometric.data import HeteroData
from typing import Optional, Dict


class AdvancedGCN(nn.Module):
    """
    Advanced Graph Convolutional Network with multiple layers and skip connections
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.3):
        super(AdvancedGCN, self).__init__()
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if prev_dim != hidden_dim:
                self.skip_connections.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.skip_connections.append(nn.Identity())
            prev_dim = hidden_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch=None):
        # GCN layers with skip connections
        for i, (conv, bn, skip) in enumerate(zip(self.convs, self.batch_norms, self.skip_connections)):
            # Skip connection
            residual = skip(x)
            
            # GCN convolution
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual if dimensions match
            if residual.shape == x.shape:
                x = x + residual
        
        # Classification - apply to each node
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class AdvancedGAT(nn.Module):
    """
    Advanced Graph Attention Network with multi-head attention
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, 
                 heads=[8, 4, 1], dropout=0.3):
        super(AdvancedGAT, self).__init__()
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        prev_dim = input_dim
        for i, (hidden_dim, num_heads) in enumerate(zip(hidden_dims, heads)):
            self.convs.append(
                GATConv(prev_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=(i < len(hidden_dims)-1))
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads if i < len(hidden_dims)-1 else hidden_dim))
            prev_dim = hidden_dim * num_heads if i < len(hidden_dims)-1 else hidden_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch=None):
        # GAT layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification - apply to each node
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class HeteroGNN(nn.Module):
    """
    Heterogeneous GNN for modeling users, accounts, and transactions
    """
    def __init__(self, metadata, hidden_dim=64, num_layers=3, num_classes=2, dropout=0.3):
        super(HeteroGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial embeddings
        self.user_lin = Linear(-1, hidden_dim)
        self.account_lin = Linear(-1, hidden_dim)
        self.transaction_lin = Linear(-1, hidden_dim)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('user', 'owns', 'account'): GCNConv(-1, hidden_dim),
                ('account', 'initiates', 'transaction'): GCNConv(-1, hidden_dim),
                ('account', 'receives', 'transaction'): GCNConv(-1, hidden_dim),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Final classifier for transactions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x_dict, edge_index_dict):
        # Initial linear transformations
        x_dict['user'] = self.user_lin(x_dict['user'])
        x_dict['account'] = self.account_lin(x_dict['account'])
        x_dict['transaction'] = self.transaction_lin(x_dict['transaction'])
        
        # Heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Classify transactions
        x = self.classifier(x_dict['transaction'])
        return F.log_softmax(x, dim=1)


class EnsembleGNN(nn.Module):
    """
    Ensemble of multiple GNN models for improved performance
    """
    def __init__(self, input_dim, num_classes=2):
        super(EnsembleGNN, self).__init__()
        
        self.gcn = AdvancedGCN(input_dim, hidden_dims=[128, 64, 32], num_classes=num_classes)
        self.gat = AdvancedGAT(input_dim, hidden_dims=[128, 64, 32], num_classes=num_classes)
        
        # Ensemble weights
        self.ensemble_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, x, edge_index, batch=None):
        out_gcn = self.gcn(x, edge_index, batch)
        out_gat = self.gat(x, edge_index, batch)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weight, dim=0)
        out = weights[0] * out_gcn + weights[1] * out_gat
        
        return out

