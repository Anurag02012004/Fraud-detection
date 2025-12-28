"""
Script to prepare the project for deployment
Trains a model and ensures all necessary files are in place
"""
import os
import subprocess
import sys

def main():
    print("Preparing project for deployment...")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Train model if it doesn't exist
    model_path = 'models/best_gat.pth'
    if not os.path.exists(model_path):
        print("\nTraining GAT model (required for demo)...")
        print("This may take a few minutes. Please be patient...")
        subprocess.run([
            sys.executable, "train.py", 
            "--model", "gat", 
            "--epochs", "50"
        ])
    else:
        print(f"\nModel already exists at {model_path}")
    
    print("\nProject is ready for deployment!")
    print("\nNext steps:")
    print("1. Push to GitHub: git push")
    print("2. Deploy on Streamlit Cloud: https://share.streamlit.io")
    print("3. Set main file to: app.py")

if __name__ == "__main__":
    main()


