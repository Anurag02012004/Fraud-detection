"""
Quick start script to train a model and run the demo
"""
import subprocess
import sys
import os

def main():
    print("🚀 Advanced GNN Fraud Detection - Quick Start")
    print("=" * 50)
    
    # Step 1: Install dependencies
    print("\n📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    
    # Step 2: Train model
    print("\n🎯 Training GAT model (this may take a few minutes)...")
    subprocess.run([sys.executable, "train.py", "--model", "gat", "--epochs", "50"])
    
    # Step 3: Run app
    print("\n✨ Starting Streamlit app...")
    print("The app will open in your browser automatically.")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()


