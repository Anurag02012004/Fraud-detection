"""
Setup script for Advanced GNN Fraud Detection
"""
from setuptools import setup, find_packages

setup(
    name="gnn-fraud-detection",
    version="1.0.0",
    description="Advanced Graph Neural Network for Fraud Detection",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "streamlit>=1.24.0",
        "pyvis>=0.3.0",
    ],
)


