"""Evaluation utilities for emotion recognition."""

import logging
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

from ..utils import get_device, create_confusion_matrix_plot
from ..models import create_model, load_pretrained_model
from ..data import EmotionDataModule

logger = logging.getLogger(__name__)


class EmotionEvaluator:
    """Evaluator class for emotion recognition models.
    
    Args:
        config: Configuration object
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        emotion_classes: List of emotion class names
    """
    
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        emotion_classes: List[str],
    ):
        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.emotion_classes = emotion_classes
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Generate reports
        self._generate_reports(all_targets, all_predictions)
        
        # Create visualizations
        self._create_visualizations(all_targets, all_predictions, all_probabilities)
        
        logger.info("Evaluation completed!")
        return metrics
    
    def _calculate_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average="weighted")
        recall = recall_score(targets, predictions, average="weighted")
        f1 = f1_score(targets, predictions, average="weighted")
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None)
        recall_per_class = recall_score(targets, predictions, average=None)
        f1_per_class = f1_score(targets, predictions, average=None)
        
        # Macro averages
        precision_macro = precision_score(targets, predictions, average="macro")
        recall_macro = recall_score(targets, predictions, average="macro")
        f1_macro = f1_score(targets, predictions, average="macro")
        
        # Confidence metrics
        max_probabilities = np.max(probabilities, axis=1)
        confidence_mean = np.mean(max_probabilities)
        confidence_std = np.std(max_probabilities)
        
        # Calibration metrics
        calibration_error = self._calculate_calibration_error(targets, predictions, probabilities)
        
        metrics = {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "calibration_error": calibration_error,
        }
        
        # Add per-class metrics
        for i, emotion in enumerate(self.emotion_classes):
            metrics[f"precision_{emotion}"] = precision_per_class[i]
            metrics[f"recall_{emotion}"] = recall_per_class[i]
            metrics[f"f1_{emotion}"] = f1_per_class[i]
        
        return metrics
    
    def _calculate_calibration_error(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate calibration error.
        
        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration error
        """
        max_probabilities = np.max(probabilities, axis=1)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probabilities > bin_lower) & (max_probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = max_probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _generate_reports(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Generate classification reports.
        
        Args:
            targets: True labels
            predictions: Predicted labels
        """
        # Classification report
        report = classification_report(
            targets,
            predictions,
            target_names=self.emotion_classes,
            output_dict=True,
        )
        
        # Save detailed report
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(self.config.evaluation.output_dir, "classification_report.csv")
        report_df.to_csv(report_path)
        
        logger.info(f"Classification report saved to {report_path}")
        
        # Print summary
        logger.info("Classification Report:")
        logger.info(classification_report(targets, predictions, target_names=self.emotion_classes))
    
    def _create_visualizations(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Create evaluation visualizations.
        
        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
        """
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.emotion_classes,
            yticklabels=self.emotion_classes,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        cm_path = os.path.join(self.config.evaluation.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Confidence distribution
        max_probabilities = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probabilities, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Confidence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        conf_path = os.path.join(self.config.evaluation.output_dir, "confidence_distribution.png")
        plt.savefig(conf_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Per-class accuracy
        per_class_acc = []
        for i, emotion in enumerate(self.emotion_classes):
            class_mask = targets == i
            if np.sum(class_mask) > 0:
                class_acc = (predictions[class_mask] == targets[class_mask]).mean()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.emotion_classes, per_class_acc, alpha=0.7, edgecolor="black")
        plt.xlabel("Emotion Class")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy")
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, per_class_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{acc:.3f}", ha="center", va="bottom")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        acc_path = os.path.join(self.config.evaluation.output_dir, "per_class_accuracy.png")
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Visualizations saved to {self.config.evaluation.output_dir}")
    
    def create_leaderboard(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """Create evaluation leaderboard.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Leaderboard DataFrame
        """
        # Overall metrics
        overall_metrics = {
            "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1 (Weighted)",
                      "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "Calibration Error"],
            "Value": [
                metrics["accuracy"],
                metrics["precision_weighted"],
                metrics["recall_weighted"],
                metrics["f1_weighted"],
                metrics["precision_macro"],
                metrics["recall_macro"],
                metrics["f1_macro"],
                metrics["calibration_error"],
            ]
        }
        
        overall_df = pd.DataFrame(overall_metrics)
        
        # Per-class metrics
        per_class_data = []
        for emotion in self.emotion_classes:
            per_class_data.append({
                "Emotion": emotion,
                "Precision": metrics[f"precision_{emotion}"],
                "Recall": metrics[f"recall_{emotion}"],
                "F1": metrics[f"f1_{emotion}"],
            })
        
        per_class_df = pd.DataFrame(per_class_data)
        
        # Save leaderboards
        overall_path = os.path.join(self.config.evaluation.output_dir, "overall_metrics.csv")
        per_class_path = os.path.join(self.config.evaluation.output_dir, "per_class_metrics.csv")
        
        overall_df.to_csv(overall_path, index=False)
        per_class_df.to_csv(per_class_path, index=False)
        
        logger.info(f"Leaderboards saved to {self.config.evaluation.output_dir}")
        
        return overall_df, per_class_df


def evaluate_model(config: DictConfig, model_path: Optional[str] = None) -> Dict[str, float]:
    """Evaluate emotion recognition model.
    
    Args:
        config: Configuration object
        model_path: Path to model checkpoint (None for best model)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data module
    data_module = EmotionDataModule(config)
    train_df, val_df, test_df = data_module.prepare_data()
    _, _, test_loader = data_module.create_dataloaders(train_df, val_df, test_df)
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config.training.checkpoint_dir, "best.pth")
    
    model = load_pretrained_model(model_path, config.model, device)
    
    # Get emotion classes from dataset
    emotion_classes = sorted(test_df["emotion"].unique())
    
    # Create evaluator
    evaluator = EmotionEvaluator(config, model, test_loader, device, emotion_classes)
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Create leaderboard
    evaluator.create_leaderboard(metrics)
    
    # Print summary
    logger.info("Evaluation Summary:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"Calibration Error: {metrics['calibration_error']:.4f}")
    
    return metrics
