# Speech Emotion Recognition - Quick Start Guide

This notebook demonstrates how to use the speech emotion recognition system.

## 1. Setup and Configuration

```python
# Import required libraries
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from src.utils import set_seed, get_device
from src.data import create_synthetic_dataset, EmotionDataModule
from src.models import create_model
from src.features import AudioFeatureExtractor
from src.train import train_model
from src.eval import evaluate_model

# Load configuration
config = OmegaConf.load("configs/default.yaml")
print(f"Model type: {config.model.type}")
print(f"Feature type: {config.features.type}")
print(f"Sample rate: {config.features.sample_rate} Hz")

# Set random seed for reproducibility
set_seed(config.training.seed)

# Get device
device = get_device()
print(f"Using device: {device}")
```

## 2. Create Synthetic Dataset

```python
# Create synthetic dataset
print("Creating synthetic dataset...")
synthetic_df = create_synthetic_dataset(
    output_dir=config.data.data_dir,
    num_samples_per_emotion=50,  # Smaller for demo
    sample_rate=config.features.sample_rate,
    duration=2.0
)

print(f"Created dataset with {len(synthetic_df)} samples")
print(f"Emotions: {synthetic_df['emotion'].unique()}")
print(f"Split distribution:")
print(synthetic_df['split'].value_counts())
```

## 3. Visualize Dataset

```python
# Plot emotion distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=synthetic_df, x='emotion', hue='split')
plt.title('Emotion Distribution by Split')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot duration distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=synthetic_df, x='duration', bins=20)
plt.title('Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.tight_layout()
plt.show()
```

## 4. Feature Extraction Demo

```python
# Load a sample audio file
import librosa
sample_file = synthetic_df.iloc[0]['file_path']
sample_path = os.path.join(config.data.data_dir, sample_file)

# Load audio
audio, sr = librosa.load(sample_path, sr=config.features.sample_rate)

print(f"Audio shape: {audio.shape}")
print(f"Duration: {len(audio) / sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")

# Extract features
feature_extractor = AudioFeatureExtractor(config.features)

# Extract different types of features
mfcc = feature_extractor.extract_features(audio, "mfcc")
log_mel = feature_extractor.extract_features(audio, "log_mel")
combined = feature_extractor.extract_features(audio, "combined")

print(f"MFCC shape: {mfcc.shape}")
print(f"Log-mel shape: {log_mel.shape}")
print(f"Combined shape: {combined.shape}")
```

## 5. Model Training

```python
# Create model
model = create_model(config.model)
print(f"Model type: {config.model.type}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Move model to device
model = model.to(device)
print(f"Model moved to {device}")

# Train model (quick training for demo)
print("Starting training...")

# Update config for quick training
config.training.max_epochs = 5  # Quick demo
config.training.batch_size = 16

# Train
train_model(config)
print("Training completed!")
```

## 6. Model Evaluation

```python
# Evaluate model
print("Evaluating model...")
metrics = evaluate_model(config)

print("\nEvaluation Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
print(f"Calibration Error: {metrics['calibration_error']:.4f}")
```

## 7. Inference Demo

```python
# Load trained model
from src.models import load_pretrained_model
import torch

model_path = os.path.join(config.training.checkpoint_dir, "best.pth")
trained_model = load_pretrained_model(model_path, config.model, device)

print("Model loaded successfully!")

# Test on a few samples
emotions = ['happy', 'sad', 'angry', 'neutral']

for emotion in emotions:
    # Get a sample of this emotion
    emotion_samples = synthetic_df[synthetic_df['emotion'] == emotion]
    if len(emotion_samples) > 0:
        sample_file = emotion_samples.iloc[0]['file_path']
        sample_path = os.path.join(config.data.data_dir, sample_file)
        
        # Load and process audio
        audio, sr = librosa.load(sample_path, sr=config.features.sample_rate)
        
        # Extract features
        features = feature_extractor.extract_features(audio, config.features.type)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = trained_model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        predicted_emotion = emotions[predicted_idx] if predicted_idx < len(emotions) else "unknown"
        
        print(f"True: {emotion}, Predicted: {predicted_emotion}, Confidence: {confidence:.3f}")
```

## Summary

This notebook demonstrated:

1. **Dataset Creation**: Generating synthetic emotion data
2. **Feature Extraction**: Multiple audio feature types
3. **Model Training**: Training emotion recognition models
4. **Evaluation**: Comprehensive model assessment
5. **Inference**: Real-time emotion prediction

The system is now ready for:
- Interactive demo via Streamlit
- Further model experimentation
- Integration with real datasets
- Production deployment (with proper considerations)

**Remember**: This is a research tool for educational purposes only!
