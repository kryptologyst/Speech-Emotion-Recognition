"""Unit tests for speech emotion recognition."""

import pytest
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.models import create_model, EmotionCNN, EmotionTransformer
from src.features import AudioFeatureExtractor, AudioAugmentation
from src.data import EmotionDataset, create_synthetic_dataset
from src.utils import set_seed, get_device, EarlyStopping


class TestModels:
    """Test model architectures."""
    
    def test_cnn_model(self):
        """Test CNN model creation and forward pass."""
        config = OmegaConf.create({
            "type": "cnn",
            "num_classes": 7,
            "input_channels": 1,
            "dropout_rate": 0.3
        })
        
        model = create_model(config)
        assert isinstance(model, EmotionCNN)
        
        # Test forward pass
        batch_size = 2
        height, width = 80, 100
        x = torch.randn(batch_size, 1, height, width)
        
        output = model(x)
        assert output.shape == (batch_size, 7)
    
    def test_transformer_model(self):
        """Test Transformer model creation and forward pass."""
        config = OmegaConf.create({
            "type": "transformer",
            "num_classes": 7,
            "input_dim": 80,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dropout_rate": 0.1
        })
        
        model = create_model(config)
        assert isinstance(model, EmotionTransformer)
        
        # Test forward pass
        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 80)
        
        output = model(x)
        assert output.shape == (batch_size, 7)


class TestFeatureExtraction:
    """Test feature extraction utilities."""
    
    def test_audio_feature_extractor(self):
        """Test audio feature extraction."""
        config = OmegaConf.create({
            "sample_rate": 16000,
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 80,
            "n_mfcc": 13
        })
        
        extractor = AudioFeatureExtractor(config)
        
        # Create dummy audio
        audio = np.random.randn(16000)  # 1 second of audio
        
        # Test different feature types
        mfcc = extractor.extract_features(audio, "mfcc")
        assert mfcc.shape[0] == 13
        
        log_mel = extractor.extract_features(audio, "log_mel")
        assert log_mel.shape[0] == 80
        
        combined = extractor.extract_features(audio, "combined")
        assert combined.shape[0] > 80  # Combined features should have more dimensions
    
    def test_audio_augmentation(self):
        """Test audio augmentation."""
        config = OmegaConf.create({
            "noise_factor": 0.005,
            "speed_factor_range": [0.9, 1.1],
            "pitch_shift_range": [-2, 2],
            "time_stretch_range": [0.8, 1.2]
        })
        
        augmentation = AudioAugmentation(config)
        
        # Create dummy audio
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        # Test different augmentations
        noisy_audio = augmentation.add_noise(audio)
        assert noisy_audio.shape == audio.shape
        
        speed_changed = augmentation.speed_change(audio, sample_rate)
        assert isinstance(speed_changed, np.ndarray)


class TestDataHandling:
    """Test data handling utilities."""
    
    def test_synthetic_dataset_creation(self, tmp_path):
        """Test synthetic dataset creation."""
        output_dir = str(tmp_path / "synthetic_data")
        
        df = create_synthetic_dataset(
            output_dir=output_dir,
            num_samples_per_emotion=10,
            sample_rate=16000,
            duration=1.0
        )
        
        assert len(df) == 70  # 7 emotions * 10 samples
        assert "file_path" in df.columns
        assert "emotion" in df.columns
        assert "split" in df.columns
        
        # Check emotion distribution
        emotions = df["emotion"].unique()
        assert len(emotions) == 7
        
        # Check split distribution
        splits = df["split"].unique()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Test numpy random
        np.random.seed(42)
        val1 = np.random.randn()
        
        set_seed(42)
        val2 = np.random.randn()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Mock model
        model = torch.nn.Linear(1, 1)
        
        # Test improving loss
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.4, model)
        assert not early_stopping(0.3, model)
        
        # Test non-improving loss
        assert not early_stopping(0.3, model)  # Same loss
        assert not early_stopping(0.3, model)  # Same loss
        assert not early_stopping(0.3, model)  # Same loss
        assert early_stopping(0.3, model)  # Should trigger early stopping


if __name__ == "__main__":
    pytest.main([__file__])
