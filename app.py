"""
Streamlit Demo Application for Advanced GNN Fraud Detection
"""
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from src.data_loader import FraudDatasetGenerator, create_homogeneous_graph
from src.model import AdvancedGCN, AdvancedGAT, EnsembleGNN
from src.utils import calculate_metrics, set_seed

# Set page config
st.set_page_config(
    page_title="Advanced GNN Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_type='gat', device='cpu'):
    """Load trained model"""
    try:
        if model_type == 'gcn':
            model = AdvancedGCN(input_dim=6, hidden_dims=[128, 64, 32], num_classes=2)
        elif model_type == 'gat':
            model = AdvancedGAT(input_dim=6, hidden_dims=[128, 64, 32], num_classes=2)
        else:
            model = EnsembleGNN(input_dim=6, num_classes=2)
        
        model_path = f'models/best_{model_type}.pth'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                st.sidebar.success("✅ Trained model loaded")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error loading checkpoint: {e}. Using untrained model.")
        else:
            st.sidebar.warning(f"⚠️ Model not found at {model_path}. Using untrained model.")
            st.sidebar.info("💡 Train a model using: python train.py")
        
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def generate_demo_data(n_users=500, n_accounts=800, n_transactions=5000):
    """Generate demo dataset"""
    set_seed(42)
    generator = FraudDatasetGenerator(
        n_users=n_users,
        n_accounts=n_accounts,
        n_transactions=n_transactions,
        fraud_ratio=0.1,
        seed=42
    )
    hetero_data, transactions_df = generator.generate_transaction_graph()
    data = create_homogeneous_graph(hetero_data)
    return data, transactions_df, hetero_data

def create_interactive_graph(data, sample_size=200):
    """Create interactive graph visualization"""
    G = nx.Graph()
    
    # Sample nodes for visualization
    indices = np.random.choice(data.num_nodes, min(sample_size, data.num_nodes), replace=False)
    
    # Add nodes
    for idx in indices:
        label = "Fraud" if data.y[idx].item() == 1 else "Legitimate"
        G.add_node(idx, label=label, group=data.y[idx].item())
    
    # Add edges (sample)
    edge_index = data.edge_index.cpu().numpy()
    num_edges_to_show = min(1000, edge_index.shape[1])
    edge_indices = np.random.choice(edge_index.shape[1], num_edges_to_show, replace=False)
    
    for i in edge_indices:
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in G and dst in G:
            G.add_edge(int(src), int(dst))
    
    # Create Pyvis network
    net = Network(height='500px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)
    
    # Save to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    net.save_graph(tmp_file.name)
    
    return tmp_file.name

def predict_fraud(model, data, device='cpu'):
    """Make predictions"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index)
        probas = torch.exp(out).cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
    return preds, probas

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Advanced GNN Fraud Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    model_type = st.sidebar.selectbox("Model Type", ["gat", "gcn", "ensemble"], index=0)
    sample_size = st.sidebar.slider("Graph Sample Size", 50, 500, 200)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    with st.spinner("Generating demo dataset..."):
        data, transactions_df, hetero_data = generate_demo_data()
    
    # Load model
    model = load_model(model_type, device)
    
    if model is None:
        st.error("Failed to load model. Please train the model first using train.py")
        return
    
    # Make predictions
    with st.spinner("Running fraud detection..."):
        preds, probas = predict_fraud(model, data, device)
    
    # Metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Performance")
    metrics = calculate_metrics(data.y.numpy(), preds, probas[:, 1])
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col2:
        st.metric("F1-Score", f"{metrics['f1']:.3f}")
        st.metric("Recall", f"{metrics['recall']:.3f}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Predictions", "📈 Analytics", "🌐 Graph Visualization"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(transactions_df):,}")
        with col2:
            fraud_count = transactions_df['is_fraud'].sum()
            st.metric("Fraudulent", f"{fraud_count:,}", f"({fraud_count/len(transactions_df)*100:.1f}%)")
        with col3:
            st.metric("Users", f"{hetero_data['user'].num_nodes:,}")
        with col4:
            st.metric("Accounts", f"{hetero_data['account'].num_nodes:,}")
        
        # Fraud distribution
        fig = px.pie(
            values=[len(transactions_df) - fraud_count, fraud_count],
            names=['Legitimate', 'Fraudulent'],
            title="Transaction Distribution",
            color_discrete_map={'Legitimate': '#2ecc71', 'Fraudulent': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Fraud Detection Results")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Transaction ID': range(len(preds)),
            'Amount': transactions_df['amount'].values[:len(preds)],
            'True Label': ['Fraudulent' if y == 1 else 'Legitimate' for y in data.y.numpy()[:len(preds)]],
            'Predicted Label': ['Fraudulent' if p == 1 else 'Legitimate' for p in preds],
            'Fraud Probability': probas[:, 1][:len(preds)],
            'Confidence': np.max(probas, axis=1)[:len(preds)]
        })
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_only_fraud = st.checkbox("Show only fraudulent transactions")
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5)
        
        filtered_df = results_df[
            (results_df['Fraud Probability'] >= min_confidence) &
            (results_df['True Label'] == 'Fraudulent' if show_only_fraud else True)
        ]
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download results
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download Results", csv, "fraud_detection_results.csv", "text/csv")
    
    with tab3:
        st.header("Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC-like visualization
            fraud_probas = probas[data.y.numpy() == 1, 1]
            legit_probas = probas[data.y.numpy() == 0, 1]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=legit_probas, name='Legitimate', opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=fraud_probas, name='Fraudulent', opacity=0.7, nbinsx=30))
            fig.update_layout(title="Probability Distribution", xaxis_title="Fraud Probability", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Amount vs Fraud
            fig = px.box(
                transactions_df.head(len(preds)),
                x='is_fraud',
                y='amount',
                labels={'is_fraud': 'Fraudulent', 'amount': 'Transaction Amount'},
                title="Transaction Amount Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data.y.numpy()[:len(preds)], preds)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Legitimate', 'Fraudulent'],
            y=['Legitimate', 'Fraudulent'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Interactive Graph Visualization")
        
        st.info("This visualization shows a sample of the transaction graph. Nodes represent transactions, and edges connect related transactions.")
        
        if st.button("Generate Graph Visualization"):
            with st.spinner("Creating interactive graph..."):
                graph_file = create_interactive_graph(data, sample_size)
                with open(graph_file, 'r') as f:
                    st.components.v1.html(f.read(), height=600)
                os.unlink(graph_file)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Advanced Graph Neural Network for Fraud Detection | Built with PyTorch Geometric & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

