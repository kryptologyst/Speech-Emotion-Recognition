#!/usr/bin/env python3
"""Training script for speech emotion recognition."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import DictConfig, OmegaConf
from src.utils import setup_logging, set_seed
from src.train import train_model
from src.data import create_synthetic_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train speech emotion recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--create-synthetic",
        action="store_true",
        help="Create synthetic dataset for training"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    config = OmegaConf.load(args.config)
    
    # Create synthetic dataset if requested
    if args.create_synthetic or config.data.get("create_synthetic", False):
        logger.info("Creating synthetic dataset...")
        synthetic_df = create_synthetic_dataset(
            output_dir=config.data.data_dir,
            num_samples_per_emotion=config.data.get("synthetic_samples_per_emotion", 100),
            sample_rate=config.features.sample_rate,
        )
        
        # Update metadata path
        config.data.metadata_path = os.path.join(config.data.data_dir, "metadata.csv")
        logger.info(f"Synthetic dataset created with {len(synthetic_df)} samples")
    
    # Create necessary directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.evaluation.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Train model
    try:
        train_model(config)
        logger.info("Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
