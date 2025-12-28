"""
Advanced data generation and preprocessing for fraud detection
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import networkx as nx
from typing import Tuple, List, Dict
import random


class FraudDatasetGenerator:
    """
    Generate synthetic financial transaction graph data with fraud patterns
    """
    
    def __init__(self, n_users=1000, n_accounts=1500, n_transactions=10000, fraud_ratio=0.1, seed=42):
        self.n_users = n_users
        self.n_accounts = n_accounts
        self.n_transactions = n_transactions
        self.fraud_ratio = fraud_ratio
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_transaction_graph(self) -> Tuple[HeteroData, pd.DataFrame]:
        """
        Generate heterogeneous graph with users, accounts, and transactions
        """
        # Generate users
        users = pd.DataFrame({
            'user_id': range(self.n_users),
            'age': np.random.normal(35, 12, self.n_users).astype(int),
            'account_age_days': np.random.lognormal(6, 1, self.n_users).astype(int),
            'total_transactions': np.random.poisson(50, self.n_users),
            'avg_transaction_amount': np.random.lognormal(5, 1, self.n_users),
        })
        users['age'] = np.clip(users['age'], 18, 80)
        
        # Generate accounts
        accounts = pd.DataFrame({
            'account_id': range(self.n_accounts),
            'user_id': np.random.choice(self.n_users, self.n_accounts),
            'balance': np.random.lognormal(8, 2, self.n_accounts),
            'credit_limit': np.random.lognormal(7.5, 1.5, self.n_accounts),
        })
        
        # Generate transactions
        n_fraud = int(self.n_transactions * self.fraud_ratio)
        n_legitimate = self.n_transactions - n_fraud
        
        # Legitimate transactions
        legitimate = pd.DataFrame({
            'transaction_id': range(n_legitimate),
            'from_account': np.random.choice(self.n_accounts, n_legitimate),
            'to_account': np.random.choice(self.n_accounts, n_legitimate),
            'amount': np.random.lognormal(5, 1.2, n_legitimate),
            'timestamp': np.random.uniform(0, 30*24*3600, n_legitimate),  # 30 days
            'is_fraud': 0
        })
        
        # Fraud transactions (with suspicious patterns)
        fraud = pd.DataFrame({
            'transaction_id': range(n_legitimate, self.n_transactions),
            'from_account': np.random.choice(self.n_accounts, n_fraud),
            'to_account': np.random.choice(self.n_accounts, n_fraud),
            'amount': np.random.lognormal(7, 1.5, n_fraud),  # Higher amounts
            'timestamp': np.random.uniform(0, 30*24*3600, n_fraud),
            'is_fraud': 1
        })
        
        transactions = pd.concat([legitimate, fraud]).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Add temporal features
        transactions['hour'] = (transactions['timestamp'] / 3600) % 24
        transactions['day_of_week'] = ((transactions['timestamp'] / (24*3600)) % 7).astype(int)
        
        # Add account features to transactions
        transactions = transactions.merge(
            accounts[['account_id', 'balance', 'credit_limit']],
            left_on='from_account', right_on='account_id', suffixes=('', '_from')
        )
        transactions = transactions.merge(
            accounts[['account_id', 'balance', 'credit_limit']],
            left_on='to_account', right_on='account_id', suffixes=('_from', '_to')
        )
        
        # Calculate additional features
        transactions['amount_to_balance_ratio'] = transactions['amount'] / (transactions['balance_from'] + 1e-6)
        transactions['amount_to_limit_ratio'] = transactions['amount'] / (transactions['credit_limit_from'] + 1e-6)
        transactions['balance_diff'] = transactions['balance_to'] - transactions['balance_from']
        
        return self._build_hetero_graph(users, accounts, transactions)
    
    def _build_hetero_graph(self, users: pd.DataFrame, accounts: pd.DataFrame, 
                           transactions: pd.DataFrame) -> Tuple[HeteroData, pd.DataFrame]:
        """
        Build PyTorch Geometric HeteroData graph
        """
        data = HeteroData()
        
        # Node features
        # User node features
        user_features = torch.tensor(
            users[['age', 'account_age_days', 'total_transactions', 'avg_transaction_amount']].values,
            dtype=torch.float
        )
        user_features = (user_features - user_features.mean(0)) / (user_features.std(0) + 1e-6)
        data['user'].x = user_features
        data['user'].num_nodes = len(users)
        
        # Account node features
        account_features = torch.tensor(
            accounts[['balance', 'credit_limit']].values,
            dtype=torch.float
        )
        account_features = (account_features - account_features.mean(0)) / (account_features.std(0) + 1e-6)
        data['account'].x = account_features
        data['account'].num_nodes = len(accounts)
        
        # Transaction node features
        tx_features = ['amount', 'hour', 'day_of_week', 'amount_to_balance_ratio', 
                      'amount_to_limit_ratio', 'balance_diff']
        transaction_features = torch.tensor(
            transactions[tx_features].values,
            dtype=torch.float
        )
        transaction_features = (transaction_features - transaction_features.mean(0)) / (transaction_features.std(0) + 1e-6)
        data['transaction'].x = transaction_features
        data['transaction'].num_nodes = len(transactions)
        
        # Labels
        data['transaction'].y = torch.tensor(transactions['is_fraud'].values, dtype=torch.long)
        
        # Edges
        # User -> Account (owns)
        user_account_edges = []
        for idx, row in accounts.iterrows():
            user_account_edges.append([row['user_id'], idx])
        if user_account_edges:
            edge_index_ua = torch.tensor(user_account_edges, dtype=torch.long).t().contiguous()
            data['user', 'owns', 'account'].edge_index = edge_index_ua
        
        # Account -> Transaction (initiates)
        account_transaction_from = []
        for idx, row in transactions.iterrows():
            account_transaction_from.append([row['from_account'], idx])
        edge_index_at_from = torch.tensor(account_transaction_from, dtype=torch.long).t().contiguous()
        data['account', 'initiates', 'transaction'].edge_index = edge_index_at_from
        
        # Account -> Transaction (receives)
        account_transaction_to = []
        for idx, row in transactions.iterrows():
            account_transaction_to.append([row['to_account'], idx])
        edge_index_at_to = torch.tensor(account_transaction_to, dtype=torch.long).t().contiguous()
        data['account', 'receives', 'transaction'].edge_index = edge_index_at_to
        
        return data, transactions
    
    def get_train_test_split(self, data: HeteroData, test_size=0.2, val_size=0.1):
        """
        Split transaction nodes into train/val/test sets
        """
        n_transactions = data['transaction'].num_nodes
        indices = torch.randperm(n_transactions)
        
        n_test = int(n_transactions * test_size)
        n_val = int(n_transactions * val_size)
        n_train = n_transactions - n_test - n_val
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        return train_idx, val_idx, test_idx


def create_homogeneous_graph(data: HeteroData) -> Data:
    """
    Convert heterogeneous graph to homogeneous for simpler models
    Creates a graph where transactions are nodes and edges connect related transactions
    """
    # Use transaction nodes as primary nodes
    transaction_features = data['transaction'].x
    transaction_labels = data['transaction'].y
    
    # Create edges between transactions sharing same accounts
    edge_list = []
    
    # Get account-transaction mappings
    from_account_tx = {}
    to_account_tx = {}
    
    at_from = data['account', 'initiates', 'transaction'].edge_index
    for i in range(at_from.size(1)):
        acc, tx = at_from[0, i].item(), at_from[1, i].item()
        if acc not in from_account_tx:
            from_account_tx[acc] = []
        from_account_tx[acc].append(tx)
    
    at_to = data['account', 'receives', 'transaction'].edge_index
    for i in range(at_to.size(1)):
        acc, tx = at_to[0, i].item(), at_to[1, i].item()
        if acc not in to_account_tx:
            to_account_tx[acc] = []
        to_account_tx[acc].append(tx)
    
    # Connect transactions sharing accounts
    all_accounts = set(from_account_tx.keys()) | set(to_account_tx.keys())
    for acc in all_accounts:
        txs = set()
        if acc in from_account_tx:
            txs.update(from_account_tx[acc])
        if acc in to_account_tx:
            txs.update(to_account_tx[acc])
        
        txs = list(txs)
        # Connect all transactions involving this account
        for i in range(len(txs)):
            for j in range(i+1, len(txs)):
                edge_list.append([txs[i], txs[j]])
                edge_list.append([txs[j], txs[i]])  # Undirected
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return Data(x=transaction_features, edge_index=edge_index, y=transaction_labels)

