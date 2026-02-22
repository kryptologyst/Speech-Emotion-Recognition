#!/usr/bin/env python3
"""Evaluation script for speech emotion recognition."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import DictConfig, OmegaConf
from src.utils import setup_logging
from src.eval import evaluate_model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate speech emotion recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: best model)"
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
    
    # Create output directory
    os.makedirs(config.evaluation.output_dir, exist_ok=True)
    
    # Evaluate model
    try:
        metrics = evaluate_model(config, args.model_path)
        
        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Calibration Error: {metrics['calibration_error']:.4f}")
        
        logger.info("Evaluation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
