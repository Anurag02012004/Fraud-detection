# Getting Started Guide

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
```bash
python train.py --model gat --epochs 50
```
This will:
- Generate synthetic fraud dataset
- Train a GAT model
- Save the best model to `models/best_gat.pth`
- Generate evaluation plots

### 3. Run the Demo
```bash
streamlit run app.py
```

The app will open in your browser automatically!

## For Deployment

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Advanced GNN Fraud Detection"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy"

3. **Optional: Pre-train Model**
   For faster first load, you can train the model locally and commit it:
   ```bash
   python train.py --model gat --epochs 50
   git add models/
   git commit -m "Add trained model"
   git push
   ```

### Option 2: Local Development

Simply run:
```bash
streamlit run app.py
```

The app will work even without a pre-trained model (though predictions will be from an untrained model).

## Project Structure

```
reaidy_ML/
├── app.py                    # Streamlit demo app (main entry point)
├── train.py                  # Training script
├── src/
│   ├── model.py             # GNN architectures (GAT, GCN, Ensemble)
│   ├── data_loader.py       # Data generation & preprocessing
│   └── utils.py             # Evaluation utilities
├── models/                   # Saved model checkpoints
├── notebooks/                # Jupyter notebooks
└── requirements.txt          # Python dependencies
```

## Next Steps

- **Experiment**: Modify model architectures in `src/model.py`
- **Customize**: Adjust data generation in `src/data_loader.py`
- **Extend**: Add new features or visualization in `app.py`
- **Deploy**: Follow deployment guide to share your demo!

## Need Help?

- Check `README.md` for overview
- See `DEPLOYMENT.md` for deployment details
- Read `PROJECT_SUMMARY.md` for technical details


