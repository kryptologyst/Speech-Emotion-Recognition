"""Data handling and dataset utilities."""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from .features import AudioFeatureExtractor, load_audio, normalize_audio, trim_silence

logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """Dataset class for emotion recognition.
    
    Args:
        data_dir: Directory containing audio files
        metadata: DataFrame with file paths and labels
        config: Configuration object
        split: Dataset split (train/val/test)
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata: pd.DataFrame,
        config: DictConfig,
        split: str = "train",
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.metadata = metadata
        self.config = config
        self.split = split
        self.augment = augment and split == "train"
        
        # Initialize feature extractor
        self.feature_extractor = AudioFeatureExtractor(config.features)
        
        # Get emotion classes
        self.emotions = sorted(metadata["emotion"].unique())
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        
        logger.info(f"Dataset {split}: {len(metadata)} samples, {len(self.emotions)} emotions")
        logger.info(f"Emotions: {self.emotions}")
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.data_dir, row["file_path"])
        
        # Load audio
        audio, sr = load_audio(file_path, self.config.features.sample_rate)
        
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Trim silence
        if self.config.features.get("trim_silence", True):
            audio = trim_silence(audio, sr)
        
        # Extract features
        features = self.feature_extractor.extract_features(
            audio, self.config.features.type
        )
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        
        # Get label
        emotion = row["emotion"]
        label = self.emotion_to_idx[emotion]
        
        return features, label
    
    def get_emotion_classes(self) -> List[str]:
        """Get list of emotion classes."""
        return self.emotions


class EmotionDataModule:
    """Data module for emotion recognition.
    
    Args:
        config: Configuration object
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.data_dir = config.data.data_dir
        self.metadata_path = config.data.metadata_path
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train/val/test splits.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        
        # Ensure required columns exist
        required_columns = ["file_path", "emotion", "split"]
        for col in required_columns:
            if col not in metadata.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Split data
        train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
        val_df = metadata[metadata["split"] == "val"].reset_index(drop=True)
        test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = EmotionDataset(
            self.data_dir, train_df, self.config, split="train", augment=True
        )
        val_dataset = EmotionDataset(
            self.data_dir, val_df, self.config, split="val", augment=False
        )
        test_dataset = EmotionDataset(
            self.data_dir, test_df, self.config, split="test", augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader


def create_synthetic_dataset(
    output_dir: str,
    num_samples_per_emotion: int = 100,
    sample_rate: int = 16000,
    duration: float = 3.0,
) -> pd.DataFrame:
    """Create a synthetic emotion dataset for testing.
    
    Args:
        output_dir: Output directory for synthetic data
        num_samples_per_emotion: Number of samples per emotion
        sample_rate: Sample rate
        duration: Duration of each sample
        
    Returns:
        Metadata DataFrame
    """
    import soundfile as sf
    from .features import AudioAugmentation
    
    emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    metadata = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for emotion in emotions:
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(num_samples_per_emotion):
            # Generate synthetic audio based on emotion
            if emotion == "happy":
                # Higher frequency, faster tempo
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
            elif emotion == "sad":
                # Lower frequency, slower tempo
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 330 * t)
            elif emotion == "angry":
                # Higher amplitude, more noise
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 330 * t) + 0.7 * np.random.normal(0, 0.1, len(t))
            elif emotion == "fear":
                # Tremolo effect
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 350 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
            elif emotion == "surprise":
                # Sudden frequency changes
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 600 * t)
            elif emotion == "disgust":
                # Lower frequency with harmonics
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 150 * t) + 0.4 * np.sin(2 * np.pi * 300 * t)
            else:  # neutral
                # Simple sine wave
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 300 * t)
            
            # Normalize and add some variation
            audio = normalize_audio(audio)
            audio += np.random.normal(0, 0.01, len(audio))  # Add small noise
            
            # Save audio file
            filename = f"{emotion}_{i:03d}.wav"
            filepath = os.path.join(emotion_dir, filename)
            sf.write(filepath, audio, sample_rate)
            
            # Add to metadata
            metadata.append({
                "file_path": os.path.join(emotion, filename),
                "emotion": emotion,
                "sample_rate": sample_rate,
                "duration": duration,
            })
    
    # Create metadata DataFrame
    df = pd.DataFrame(metadata)
    
    # Add train/val/test splits
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["emotion"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["emotion"])
    
    # Add split information
    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    
    logger.info(f"Created synthetic dataset with {len(df)} samples")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return df


def balance_dataset(metadata: pd.DataFrame, max_samples_per_class: Optional[int] = None) -> pd.DataFrame:
    """Balance dataset by limiting samples per class.
    
    Args:
        metadata: Input metadata DataFrame
        max_samples_per_class: Maximum samples per class (None for no limit)
        
    Returns:
        Balanced metadata DataFrame
    """
    if max_samples_per_class is None:
        return metadata
    
    balanced_dfs = []
    for emotion in metadata["emotion"].unique():
        emotion_df = metadata[metadata["emotion"] == emotion]
        if len(emotion_df) > max_samples_per_class:
            emotion_df = emotion_df.sample(n=max_samples_per_class, random_state=42)
        balanced_dfs.append(emotion_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Balanced dataset: {len(balanced_df)} samples")
    return balanced_df
