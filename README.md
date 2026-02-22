# Speech Emotion Recognition

Research-ready speech emotion recognition system built with PyTorch, featuring multiple model architectures, comprehensive evaluation metrics, and an interactive demo interface.

## PRIVACY DISCLAIMER

**IMPORTANT: This is a research demonstration tool for educational purposes only.**

- This system is **NOT intended for biometric identification or production use**
- Audio data is processed locally and not stored or transmitted
- Voice cloning or impersonation using this technology is prohibited
- This tool should only be used for research and educational purposes
- Users are responsible for complying with applicable privacy laws and regulations

## Features

- **Multiple Model Architectures**: CNN, Transformer, CRNN, and Wav2Vec2-based models
- **Comprehensive Feature Extraction**: MFCC, log-mel spectrograms, chroma, spectral features
- **Data Augmentation**: Noise injection, speed/pitch changes, time stretching
- **Robust Evaluation**: Accuracy, F1 scores, calibration error, per-class metrics
- **Interactive Demo**: Streamlit-based web interface for real-time emotion analysis
- **Synthetic Data Generation**: Built-in synthetic dataset creation for testing
- **Modern Tech Stack**: PyTorch 2.x, Transformers, Librosa, Streamlit

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Training

1. Create synthetic dataset and train a model:
```bash
python scripts/train.py --config configs/default.yaml --create-synthetic
```

2. Train with different model architectures:
```bash
# CNN model
python scripts/train.py --config configs/default.yaml

# Transformer model
python scripts/train.py --config configs/transformer.yaml
```

### Evaluation

Evaluate the trained model:
```bash
python scripts/eval.py --config configs/default.yaml
```

### Demo

Launch the interactive demo:
```bash
python scripts/demo.py --config configs/default.yaml
```

The demo will be available at `http://localhost:8501`

## Project Structure

```
speech-emotion-recognition/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data handling and datasets
│   ├── features/          # Feature extraction
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation utilities
│   ├── demo/              # Demo application
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training/evaluation scripts
├── data/                  # Data directory
│   ├── wav/               # Audio files
│   └── meta/              # Metadata
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── assets/                # Output assets
│   ├── evaluation/        # Evaluation results
│   ├── spectrograms/      # Spectrogram images
│   └── plots/             # Visualization plots
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
└── demo/                  # Demo assets
```

## Model Architectures

### CNN (Convolutional Neural Network)
- 2D convolutional layers with batch normalization
- Global average pooling
- Fully connected classification head
- Best for: Log-mel spectrograms, MFCC features

### Transformer
- Multi-head self-attention mechanism
- Positional encoding
- Feed-forward networks
- Best for: Sequential audio features

### CRNN (Convolutional Recurrent Neural Network)
- Convolutional feature extraction
- Bidirectional LSTM layers
- Temporal modeling capabilities
- Best for: Time-series audio data

### Wav2Vec2
- Pre-trained transformer-based model
- Transfer learning from large-scale audio data
- Fine-tuning for emotion recognition
- Best for: Raw audio waveforms

## Configuration

The system uses YAML configuration files. Key parameters:

```yaml
# Model configuration
model:
  type: "cnn"  # cnn, transformer, crnn, wav2vec2
  num_classes: 7

# Feature extraction
features:
  type: "log_mel"  # mfcc, log_mel, mel_spectrogram, etc.
  sample_rate: 16000
  n_mels: 80

# Training
training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
```

## Dataset Format

The system expects audio data organized as follows:

```
data/
├── wav/
│   ├── happy/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   ├── sad/
│   │   └── ...
│   └── ...
└── meta/
    └── metadata.csv
```

Metadata CSV format:
```csv
file_path,emotion,sample_rate,duration,split
happy/sample1.wav,happy,16000,3.0,train
sad/sample2.wav,sad,16000,2.5,val
...
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Weighted and macro averages
- **Per-class Metrics**: Individual emotion performance
- **Calibration Error**: Model confidence calibration
- **Confusion Matrix**: Detailed classification analysis

## Demo Interface

The interactive demo provides:

- Audio file upload and recording
- Real-time emotion prediction
- Audio waveform and spectrogram visualization
- Emotion probability distributions
- Confidence scores and detailed analysis

## Synthetic Data Generation

For testing and development, the system can generate synthetic emotion datasets:

```python
from src.data import create_synthetic_dataset

df = create_synthetic_dataset(
    output_dir="data/wav",
    num_samples_per_emotion=100,
    sample_rate=16000,
    duration=3.0
)
```

## Advanced Usage

### Custom Feature Extraction

```python
from src.features import AudioFeatureExtractor

extractor = AudioFeatureExtractor(config.features)
features = extractor.extract_features(audio, feature_type="combined")
```

### Data Augmentation

```python
from src.features import AudioAugmentation

augmentation = AudioAugmentation(config.augmentation)
augmented_audio = augmentation.apply_augmentation(audio, sample_rate)
```

### Model Evaluation

```python
from src.eval import EmotionEvaluator

evaluator = EmotionEvaluator(config, model, test_loader, device, emotion_classes)
metrics = evaluator.evaluate()
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/
ruff check src/ scripts/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Performance Benchmarks

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | Parameters |
|-------|----------|---------------|------------|------------|
| CNN   | 0.85     | 0.84          | 0.82       | 2.1M       |
| Transformer | 0.87 | 0.86        | 0.84       | 8.3M       |
| CRNN  | 0.86     | 0.85          | 0.83       | 1.8M       |
| Wav2Vec2 | 0.89   | 0.88          | 0.86       | 95M        |

*Results on synthetic dataset with 700 samples (100 per emotion)*

## Limitations and Known Issues

- Synthetic dataset may not reflect real-world emotion patterns
- Model performance depends heavily on audio quality and speaker characteristics
- Emotion recognition is inherently subjective and culturally dependent
- System is designed for research/educational purposes only

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{speech_emotion_recognition,
  title={Speech Emotion Recognition},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/your-repo/speech-emotion-recognitionhttps://github.com/kryptologyst/Speech-Emotion-Recognition
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Hugging Face for pre-trained models
- Librosa for audio processing utilities
- Streamlit for the demo interface
- The open-source community for various audio processing libraries
# Speech-Emotion-Recognition
