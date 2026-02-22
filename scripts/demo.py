#!/usr/bin/env python3
"""Demo script for speech emotion recognition."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import DictConfig, OmegaConf
from src.utils import setup_logging


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Run speech emotion recognition demo")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit demo"
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
    
    # Run Streamlit demo
    try:
        import streamlit.web.cli as stcli
        
        demo_path = Path(__file__).parent.parent / "src" / "demo" / "__init__.py"
        
        sys.argv = [
            "streamlit",
            "run",
            str(demo_path),
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ]
        
        logger.info(f"Starting demo on port {args.port}")
        stcli.main()
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
