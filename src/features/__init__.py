"""Audio feature extraction utilities."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchaudio
import librosa
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Modern audio feature extractor with multiple feature types.
    
    Args:
        config: Configuration object containing feature extraction parameters
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_fft = config.get("n_fft", 1024)
        self.hop_length = config.get("hop_length", 512)
        self.n_mels = config.get("n_mels", 80)
        self.n_mfcc = config.get("n_mfcc", 13)
        self.f_min = config.get("f_min", 0)
        self.f_max = config.get("f_max", None)
        
    def extract_features(
        self,
        audio: np.ndarray,
        feature_type: str = "log_mel",
    ) -> np.ndarray:
        """Extract audio features.
        
        Args:
            audio: Input audio signal
            feature_type: Type of features to extract
            
        Returns:
            Extracted features
        """
        if feature_type == "mfcc":
            return self._extract_mfcc(audio)
        elif feature_type == "log_mel":
            return self._extract_log_mel(audio)
        elif feature_type == "mel_spectrogram":
            return self._extract_mel_spectrogram(audio)
        elif feature_type == "spectrogram":
            return self._extract_spectrogram(audio)
        elif feature_type == "chroma":
            return self._extract_chroma(audio)
        elif feature_type == "zero_crossing_rate":
            return self._extract_zcr(audio)
        elif feature_type == "spectral_centroid":
            return self._extract_spectral_centroid(audio)
        elif feature_type == "spectral_rolloff":
            return self._extract_spectral_rolloff(audio)
        elif feature_type == "combined":
            return self._extract_combined_features(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc
    
    def _extract_log_mel(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-mel spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        return mel_spec
    
    def _extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectrogram features."""
        stft = librosa.stft(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        magnitude = np.abs(stft)
        return magnitude
    
    def _extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return chroma
    
    def _extract_zcr(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate features."""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )
        return zcr
    
    def _extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid features."""
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return centroid
    
    def _extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff features."""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return rolloff
    
    def _extract_combined_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract combined features from multiple feature types."""
        features = []
        
        # MFCC features
        mfcc = self._extract_mfcc(audio)
        features.append(mfcc)
        
        # Log-mel features
        log_mel = self._extract_log_mel(audio)
        features.append(log_mel)
        
        # Chroma features
        chroma = self._extract_chroma(audio)
        features.append(chroma)
        
        # Spectral features
        zcr = self._extract_zcr(audio)
        centroid = self._extract_spectral_centroid(audio)
        rolloff = self._extract_spectral_rolloff(audio)
        
        spectral_features = np.vstack([zcr, centroid, rolloff])
        features.append(spectral_features)
        
        # Concatenate all features
        combined = np.vstack(features)
        return combined


class AudioAugmentation:
    """Audio augmentation utilities for training data augmentation.
    
    Args:
        config: Configuration object containing augmentation parameters
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.noise_factor = config.get("noise_factor", 0.005)
        self.speed_factor_range = config.get("speed_factor_range", (0.9, 1.1))
        self.pitch_shift_range = config.get("pitch_shift_range", (-2, 2))
        self.time_stretch_range = config.get("time_stretch_range", (0.8, 1.2))
        
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random noise to audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Audio with added noise
        """
        noise = np.random.normal(0, self.noise_factor, len(audio))
        return audio + noise
    
    def speed_change(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Change audio speed.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Speed-changed audio
        """
        speed_factor = np.random.uniform(*self.speed_factor_range)
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def pitch_shift(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Shift audio pitch.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Pitch-shifted audio
        """
        n_steps = np.random.randint(*self.pitch_shift_range)
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    
    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Stretch audio time.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Time-stretched audio
        """
        stretch_factor = np.random.uniform(*self.time_stretch_range)
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    def apply_augmentation(
        self,
        audio: np.ndarray,
        sample_rate: int,
        augmentation_type: str = "random",
    ) -> np.ndarray:
        """Apply audio augmentation.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented audio
        """
        if augmentation_type == "noise":
            return self.add_noise(audio)
        elif augmentation_type == "speed":
            return self.speed_change(audio, sample_rate)
        elif augmentation_type == "pitch":
            return self.pitch_shift(audio, sample_rate)
        elif augmentation_type == "time_stretch":
            return self.time_stretch(audio)
        elif augmentation_type == "random":
            # Randomly choose one augmentation
            augmentations = ["noise", "speed", "pitch", "time_stretch"]
            aug_type = np.random.choice(augmentations)
            return self.apply_augmentation(audio, sample_rate, aug_type)
        else:
            return audio


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample if necessary.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.
    
    Args:
        audio: Input audio signal
        
    Returns:
        Normalized audio signal
    """
    return audio / np.max(np.abs(audio))


def trim_silence(audio: np.ndarray, sample_rate: int, top_db: int = 20) -> np.ndarray:
    """Trim silence from audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        top_db: Silence threshold in dB
        
    Returns:
        Trimmed audio signal
    """
    return librosa.effects.trim(audio, top_db=top_db)[0]
