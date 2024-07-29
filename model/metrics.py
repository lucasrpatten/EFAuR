"""
Contains Various Machine Learning Metric and Loss Functions

Author: Lucas Patten
"""

import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """Contrastive Loss Class

    Args:
        *args: Arguments to the base class
        margin (float, optional): Margin for the contrastive loss. Defaults to 1.0.
        **kwargs: Keyword arguments to the base class
    """

    def __init__(self, *args, margin: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, targets: torch.Tensor):
        """Computes Contrastive Loss

        Args:
            output1 (torch.Tensor): The output of the first model
            output2 (torch.Tensor): The output of the second model
            targets (torch.Tensor): The target labels

        Returns:
            torch.Tensor: Contrastive Loss
        """
        return Metrics.contrastive_average(
            output1, output2, targets, self.margin
        )

class Metrics:
    """Class containing various metric computations"""

    @classmethod
    def contrastive_loss(
        cls,
        output1: torch.Tensor,
        output2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Computes Contrastive Losses

        Args:
            output1 (torch.Tensor): The output of the first model
            output2 (torch.Tensor): The output of the second model
            target (torch.Tensor): The target labels
            margin (float, optional): Margin for the contrastive loss. Defaults to 1.0.

        Returns:
            torch.Tensor: A Tensor of Contrastive Losses
        """
        # if target.size(0) != output1.size(0) or target.size(0) != output2.size(0):
        #     logging.critical(
        #         "Target and output sizes do not match. Output1: %s, Output2: %s, Target: %s",
        #         output1.size(0),
        #         output2.size(0),
        #         target.size(0),
        #     )
        #     sys.exit(1)

        euclidean_distance = F.pairwise_distance(  # pylint: disable=not-callable
            output1, output2
        )
        same_author_loss = (1 - target) * torch.pow(euclidean_distance, 2)
        different_author_loss = target * torch.pow(
            torch.clamp(margin - euclidean_distance, min=0.0), 2
        )
        return same_author_loss + different_author_loss

    @classmethod
    def binary_contrastive_tensor(
        cls,
        output1: torch.Tensor,
        output2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 1.0,
        threshold: float = 0.8,
    ):
        """Computes Contrastive Loss with threshold

        Args:
            output1 (torch.Tensor): The output of the first model
            output2 (torch.Tensor): The output of the second model
            target (torch.Tensor): The target labels
            margin (float, optional): Margin for the contrastive loss. Defaults to 1.0.
            threshold (float, optional): Threshold for binary classification. Defaults to 0.8.

        Returns:
            Tensor: Contrastive Loss
        """

        loss_contrastive = cls.contrastive_loss(
            output1, output2, target, margin
        )

        # 0 = same, 1 = different
        binary_vals = (loss_contrastive > threshold).float()

        return binary_vals

    @classmethod
    def contrastive_average(
        cls,
        output1: torch.Tensor,
        output2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Computes Contrastive Loss

        Args:
            output1 (torch.Tensor): The output of the first model
            output2 (torch.Tensor): The output of the second model
            target (torch.Tensor): The target labels
            margin (float, optional): Margin for the contrastive loss. Defaults to 1.0.

        Returns:
            Tensor: Contrastive Loss
        """
        loss_contrastive = cls.contrastive_loss(
            output1, output2, target, margin
        )
        return (loss_contrastive).mean()

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
        # 0 = same, 1 = different
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        correct_predictions = (predictions == labels).sum().item()

        total_predictions = labels.size(0)
        return float(correct_predictions / total_predictions)

    #!FIXME Why does this always return 0?
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
        # 0 = same, 1 = different
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        true_positives = ((predictions == 0.0) & (labels == 0.0)).sum().item()
        false_positives = ((predictions == 0.0) & (labels == 1.0)).sum().item()

        precision_score = true_positives / (true_positives + false_positives + 1e-9)
        return precision_score

    #!FIXME Why does this always return 0?
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
        # 0 = same, 1 = different
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        true_positives = ((predictions == 0.0) & (labels == 0.0)).sum().item()
        false_negatives = ((predictions == 1.0) & (labels == 0.0)).sum().item()

        recall_score = true_positives / (true_positives + false_negatives + 1e-9)
        return recall_score

    #!FIXME Why does this always return 0? (Because recall and precision are always 0)
    @staticmethod
    def f1_score(pred: torch.Tensor, labels: torch.Tensor, threshold: float = 0.8):
        """Computes f1_score (2 * (precision * recall) / (precision + recall + 1e-9))

        Args:
            pred (torch.Tensor): Predicted Labels
            labels (torch.Tensor): True Labels

        Returns:
            float: F1 Score
        """
        precision_score = Metrics.precision(pred, labels, threshold)
        recall_score = Metrics.recall(pred, labels, threshold)

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
        # 0 = same, 1 = different
        predictions = torch.where(
            pred >= threshold,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        return F.binary_cross_entropy_with_logits(predictions, labels).item()
