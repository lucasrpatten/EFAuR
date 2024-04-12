"""
Contains Various Machine Learning Metric and Loss Functions

Author: Lucas Patten
"""

import torch
import torch.nn.functional as F


class Metrics:
    """Class containing various metric computations"""

    @staticmethod
    def accuracy(pred: torch.Tensor, labels: torch.Tensor, threshold: float = 0.8):
        """Computes acccuracy (correct predictions / total predictions)

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels
            threshold (float, optional): Threshold for binary classification.
            Defaults to 0.8. [1 if p >= threshold else 0 for p in pred]

        Returns:
            float: Accuracy Metric
        """
        device = labels.get_device()
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        correct_predictions = (predictions == labels).sum().item()

        total_predictions = labels.size(0)
        return float(correct_predictions / total_predictions)

    @staticmethod
    def precision(pred: torch.Tensor, labels: torch.Tensor, threshold: float = 0.8):
        """Computes precision (true positives / (true positives + false positives))

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels
            threshold (float, optional): Threshold for binary classification. Defaults to 0.8.

        Returns:
            float: Precision Metric
        """
        device = labels.get_device()
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        true_positives = ((predictions == 1) & (labels == 1)).sum().item()
        false_positives = ((predictions == 1) & (labels == 0)).sum().item()

        precision_score = true_positives / (true_positives + false_positives + 1e-9)
        return precision_score

    @staticmethod
    def recall(pred: torch.Tensor, labels: torch.Tensor, threshold: float = 0.8):
        """Computes recall (true positives / (true positives + false negatives))

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels
            threshold (float, optional): Threshold for binary classification. Defaults to 0.8.

        Returns:
            float: Recall Metric
        """
        device = labels.get_device()
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        true_positives = ((predictions == 1) & (labels == 1)).sum().item()
        false_negatives = ((predictions == 0) & (labels == 1)).sum().item()

        recall_score = true_positives / (true_positives + false_negatives + 1e-9)
        return recall_score

    @staticmethod
    def f1_score(pred: torch.Tensor, labels: torch.Tensor):
        """Computes f1_score (2 * (precision * recall) / (precision + recall + 1e-9))

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels

        Returns:
            float: F1 Score
        """
        precision_score = Metrics.precision(pred, labels)
        recall_score = Metrics.recall(pred, labels)

        f1_score = (
            2
            * (precision_score * recall_score)
            / (precision_score + recall_score + 1e-9)
        )

        return f1_score

    @staticmethod
    def mse(pred: torch.Tensor, labels: torch.Tensor):
        """Computes Mean Squared Error Loss

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels

        Returns:
            float: Mean Squared Error
        """
        mse_score = F.mse_loss(pred, labels)
        return mse_score.item()

    @staticmethod
    def mae(pred: torch.Tensor, labels: torch.Tensor):
        """Computes Mean Absolute Error Loss

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels

        Returns:
            float: Mean Absolute Error
        """
        mae_score = F.l1_loss(pred, labels)
        return mae_score.item()

    @staticmethod
    def rmse(pred: torch.Tensor, labels: torch.Tensor):
        """Computes Root Mean Squared Error Loss

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels

        Returns:
            float: Root Mean Squared Error
        """
        mse_score = F.mse_loss(pred, labels)
        rmse_score = torch.sqrt(mse_score)
        return rmse_score.item()

    @staticmethod
    def cross_entropy(pred: torch.Tensor, labels: torch.Tensor):
        """Computes Cross Entropy Loss

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels

        Returns:
            float: Cross Entropy Loss
        """
        return F.cross_entropy(pred, labels).item()

    @staticmethod
    def binary_cross_entropy(
        pred: torch.Tensor, labels: torch.Tensor, threshold: float = 0.8
    ):
        """Computes Binary Cross Entropy

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels
            threshold (float, optional): Threshold for binary classification. Defaults to 0.8.

        Returns:
            float: Binary Cross Entropy Loss
        """
        device = labels.get_device()
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        return F.binary_cross_entropy_with_logits(predictions, labels).item()
