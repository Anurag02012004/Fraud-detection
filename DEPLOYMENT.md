# Deployment Guide

## Streamlit Cloud Deployment (Free)

### Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set Main file path to: `app.py`
   - Click "Deploy"

3. **First Time Setup**
   - The app will need to train a model on first load
   - This may take a few minutes
   - Alternatively, train locally and upload the model files:
     ```bash
     python train.py --model gat --epochs 50
     git add models/
     git commit -m "Add trained models"
     git push
     ```

## Alternative Free Deployment Options

### 1. Hugging Face Spaces
- Create a new Space with Streamlit SDK
- Upload your files
- Add requirements.txt
- Space will auto-deploy

### 2. Render
- Create a new Web Service
- Connect GitHub repo
- Set build command: `pip install -r requirements.txt`
- Set start command: `streamlit run app.py --server.port $PORT`

### 3. Railway
- Connect GitHub repo
- Auto-detects Python
- Add environment variables if needed

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py --model gat --epochs 100

# Run app
streamlit run app.py
```


