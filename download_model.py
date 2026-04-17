#!/usr/bin/env python3
"""
Download the pre-trained MRI classifier model
Run this during Render build process
"""
import os
import gdown

print("📥 Downloading pre-trained model...")

# Model ID from Google Drive (you need to update this with your own)
# This is a placeholder - you would upload the model to Google Drive and get its ID
model_url = "https://drive.google.com/uc?id=1kHm8eqy2x-L9pY1z0aB-m3nO4p5qR6s7"

model_path = "best_mri_classifier.h5"

# Check if model already exists
if os.path.exists(model_path):
    print(f"✅ Model already exists at {model_path}")
else:
    try:
        print(f"⏳ Downloading from {model_url[:50]}...")
        gdown.download(model_url, model_path, quiet=False)
        print(f"✅ Model downloaded to {model_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not download model: {e}")
        print("💡 Tip: Upload model to Google Drive and update MODEL_URL in download_model.py")
