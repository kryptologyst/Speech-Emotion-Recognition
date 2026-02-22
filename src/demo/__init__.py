"""Interactive demo for speech emotion recognition."""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import streamlit as st
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import DictConfig

from ..utils import get_device, load_config
from ..models import load_pretrained_model
from ..features import AudioFeatureExtractor, normalize_audio, trim_silence

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Speech Emotion Recognition Demo",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Privacy disclaimer
PRIVACY_DISCLAIMER = """
**PRIVACY DISCLAIMER**

This is a research demonstration tool for educational purposes only. 

**IMPORTANT:**
- This system is NOT intended for biometric identification or production use
- Audio data is processed locally and not stored or transmitted
- Voice cloning or impersonation using this technology is prohibited
- This tool should only be used for research and educational purposes
- Users are responsible for complying with applicable privacy laws and regulations

By using this demo, you acknowledge and agree to these terms.
"""


class EmotionRecognitionDemo:
    """Demo class for emotion recognition."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = get_device()
        self.emotion_classes = [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ]
        
        # Initialize model
        self.model = None
        self.feature_extractor = None
        
    def load_model(self) -> None:
        """Load the pre-trained model."""
        if self.model is None:
            model_path = self.config.demo.model_path
            if os.path.exists(model_path):
                self.model = load_pretrained_model(model_path, self.config.model, self.device)
                self.feature_extractor = AudioFeatureExtractor(self.config.features)
                st.success("Model loaded successfully!")
            else:
                st.error(f"Model not found at {model_path}")
                st.info("Please train a model first using the training script.")
    
    def predict_emotion(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float, np.ndarray]:
        """Predict emotion from audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Tuple of (predicted_emotion, confidence, probabilities)
        """
        if self.model is None or self.feature_extractor is None:
            return "Model not loaded", 0.0, np.zeros(len(self.emotion_classes))
        
        # Preprocess audio
        audio = normalize_audio(audio)
        if self.config.features.trim_silence:
            audio = trim_silence(audio, sample_rate)
        
        # Extract features
        features = self.feature_extractor.extract_features(
            audio, self.config.features.type
        )
        
        # Convert to tensor
        features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(features)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        predicted_emotion = self.emotion_classes[predicted_idx]
        probabilities_np = probabilities.cpu().numpy()[0]
        
        return predicted_emotion, confidence, probabilities_np
    
    def create_audio_visualization(self, audio: np.ndarray, sample_rate: int) -> go.Figure:
        """Create audio waveform visualization.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Plotly figure
        """
        time = np.linspace(0, len(audio) / sample_rate, len(audio))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=audio,
            mode='lines',
            name='Waveform',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300,
            showlegend=False
        )
        
        return fig
    
    def create_spectrogram(self, audio: np.ndarray, sample_rate: int) -> go.Figure:
        """Create spectrogram visualization.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Plotly figure
        """
        # Compute spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        db = librosa.amplitude_to_db(magnitude)
        
        # Create time and frequency axes
        times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sample_rate)
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        fig = go.Figure(data=go.Heatmap(
            z=db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            colorbar=dict(title="dB")
        ))
        
        fig.update_layout(
            title="Spectrogram",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=400
        )
        
        return fig
    
    def create_emotion_visualization(self, probabilities: np.ndarray) -> go.Figure:
        """Create emotion probability visualization.
        
        Args:
            probabilities: Emotion probabilities
            
        Returns:
            Plotly figure
        """
        colors = ['red', 'brown', 'purple', 'green', 'gray', 'blue', 'orange']
        
        fig = go.Figure(data=[
            go.Bar(
                x=self.emotion_classes,
                y=probabilities,
                marker_color=colors,
                text=[f"{p:.3f}" for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Emotion Probabilities",
            xaxis_title="Emotion",
            yaxis_title="Probability",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        return fig


def main():
    """Main demo application."""
    # Load configuration
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Initialize demo
    demo = EmotionRecognitionDemo(config)
    
    # Sidebar
    st.sidebar.title("🎭 Speech Emotion Recognition")
    st.sidebar.markdown(PRIVACY_DISCLAIMER)
    
    # Load model
    if st.sidebar.button("Load Model"):
        demo.load_model()
    
    # Main content
    st.title("Speech Emotion Recognition Demo")
    st.markdown("Upload an audio file or record your voice to analyze emotions.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    # Audio recording
    st.subheader("Or Record Audio")
    audio_bytes = st.audio(
        "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
        format="audio/wav",
        start_time=0
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            audio, sample_rate = librosa.load(tmp_path, sr=config.features.sample_rate)
            
            # Display audio info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio) / sample_rate:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sample_rate} Hz")
            with col3:
                st.metric("Channels", "Mono")
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    demo.create_audio_visualization(audio, sample_rate),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    demo.create_spectrogram(audio, sample_rate),
                    use_container_width=True
                )
            
            # Predict emotion
            if st.button("Analyze Emotion"):
                if demo.model is None:
                    st.error("Please load the model first!")
                else:
                    with st.spinner("Analyzing emotion..."):
                        predicted_emotion, confidence, probabilities = demo.predict_emotion(
                            audio, sample_rate
                        )
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Emotion", predicted_emotion.title())
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    with col2:
                        st.plotly_chart(
                            demo.create_emotion_visualization(probabilities),
                            use_container_width=True
                        )
                    
                    # Detailed probabilities
                    st.subheader("Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        "Emotion": demo.emotion_classes,
                        "Probability": probabilities
                    }).sort_values("Probability", ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
            
            # Play audio
            st.subheader("Play Audio")
            st.audio(tmp_path, format="audio/wav")
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    # Model information
    st.sidebar.subheader("Model Information")
    st.sidebar.text(f"Model Type: {config.model.type}")
    st.sidebar.text(f"Feature Type: {config.features.type}")
    st.sidebar.text(f"Sample Rate: {config.features.sample_rate} Hz")
    st.sidebar.text(f"Device: {demo.device}")
    
    # Instructions
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("""
    1. Click "Load Model" to initialize the model
    2. Upload an audio file or use the recording feature
    3. Click "Analyze Emotion" to get predictions
    4. View the results and visualizations
    """)


if __name__ == "__main__":
    main()
