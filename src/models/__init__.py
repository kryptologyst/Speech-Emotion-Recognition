"""Modern emotion recognition models."""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class EmotionCNN(nn.Module):
    """Convolutional Neural Network for emotion recognition.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.input_channels = config.get("input_channels", 1)
        self.dropout_rate = config.get("dropout_rate", 0.3)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            self.input_channels, 32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class EmotionTransformer(nn.Module):
    """Transformer-based model for emotion recognition.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.input_dim = config.get("input_dim", 80)  # log-mel features
        self.d_model = config.get("d_model", 256)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 6)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, self.num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output logits
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Args:
        d_model: Model dimension
        dropout: Dropout rate
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class EmotionCRNN(nn.Module):
    """Convolutional Recurrent Neural Network for emotion recognition.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.input_channels = config.get("input_channels", 1)
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout_rate = config.get("dropout_rate", 0.3)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            self.input_channels, 32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits
        """
        batch_size = x.size(0)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape for RNN: (batch_size, seq_len, features)
        x = x.view(batch_size, -1, 128)
        
        # RNN
        x, _ = self.rnn(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EmotionWav2Vec2(nn.Module):
    """Wav2Vec2-based model for emotion recognition.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.model_name = config.get("model_name", "facebook/wav2vec2-base")
        self.freeze_feature_extractor = config.get("freeze_feature_extractor", True)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        # Load pre-trained Wav2Vec2 model
        self.wav2vec2 = AutoModel.from_pretrained(self.model_name)
        
        if self.freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Get hidden size from model config
        model_config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = model_config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.num_classes),
        )
        
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_values: Input audio tensor
            
        Returns:
            Output logits
        """
        # Wav2Vec2 forward pass
        outputs = self.wav2vec2(input_values)
        
        # Use mean pooling over sequence dimension
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def create_model(config: DictConfig) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        PyTorch model
    """
    model_type = config.get("type", "cnn")
    
    if model_type == "cnn":
        return EmotionCNN(config)
    elif model_type == "transformer":
        return EmotionTransformer(config)
    elif model_type == "crnn":
        return EmotionCRNN(config)
    elif model_type == "wav2vec2":
        return EmotionWav2Vec2(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(
    model_path: str,
    config: DictConfig,
    device: torch.device,
) -> nn.Module:
    """Load a pre-trained model.
    
    Args:
        model_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded pre-trained model from {model_path}")
    return model
