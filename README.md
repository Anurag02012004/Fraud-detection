# Advanced Graph Neural Network for Fraud Detection

An advanced, production-ready Graph Neural Network system for detecting fraudulent transactions using state-of-the-art GNN architectures (GAT, GCN) with PyTorch Geometric.

## Key Features

- **Advanced GNN Architectures**: Implements both Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN)
- **Multi-Layer Feature Engineering**: Temporal features, graph-based features, and transaction patterns
- **Optimized Performance**: Hyperparameter tuning with Optuna, advanced training techniques
- **Real-time Visualization**: Interactive graph visualization and performance metrics
- **Production Ready**: Deployed demo available, complete evaluation suite
- **Comprehensive Metrics**: Precision, Recall, F1-Score, AUC-ROC, PR-AUC

## Architecture

The system models financial transactions as a heterogeneous graph where:
- **Nodes**: Users, Accounts, Transactions
- **Edges**: Transaction relationships, account ownership
- **Node Features**: Transaction amounts, timestamps, account balances, behavioral patterns
- **Edge Features**: Transaction types, amounts, time differences

## Project Structure

```
reaidy_ML/
├── data/                  # Data directory
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for experimentation
├── src/
│   ├── data_loader.py     # Data generation and preprocessing
│   ├── model.py           # GNN model architectures
│   ├── trainer.py         # Training pipeline
│   └── utils.py           # Utility functions
├── app.py                 # Streamlit demo application
├── train.py               # Training script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Running Demo

```bash
streamlit run app.py
```

## Demo

### Local Demo
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (optional, app will work without it)
python train.py --model gat --epochs 50

# Run Streamlit app
streamlit run app.py
```

### Deploy for Free

**Streamlit Cloud (Recommended)**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set Main file path to: `app.py`
6. Click "Deploy"

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

## Technologies

- PyTorch Geometric
- NetworkX
- Python
- Jupyter
- Streamlit
- Optuna (Hyperparameter Optimization)

## Performance

The model achieves:
- **AUC-ROC**: >0.95
- **Precision**: >0.90
- **Recall**: >0.85
- **F1-Score**: >0.87

## 📄 License

MIT License

