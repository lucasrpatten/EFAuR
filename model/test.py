import logging
import os
import torch
from dataset import AuthorshipPairDataset
from metrics import Metrics
from torch.utils.data import DataLoader
from models import SiameseAuthorshipModel
from torch.utils.tensorboard.writer import SummaryWriter

import typing
import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeterSingleGPU:
    """Metric Tracker for Tensorboard (Single GPU)

    Args:
        name (str): Metric name
        metric (Callable): Metric function
        writer (SummaryWriter): Tensorboard writer
    """

    def __init__(self, name: str, metric: typing.Callable, writer: SummaryWriter):
        self.name = name
        self.metric = metric
        self.writer = writer
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        """Reset the metric"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        prediction: torch.Tensor | typing.Tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        n=1,
        threshold: float | None = None,
    ):
        """Update the metric

        Args:
            prediction (torch.Tensor): Predictions from model
            labels (torch.Tensor): True labels
            n (int, optional): Number of samples. Defaults to 1.
            threshold (float, optional): Threshold for metric computation. Defaults to None.
        """
        if threshold is not None:
            self.val = self.metric(prediction, labels, threshold)
        else:
            self.val = self.metric(prediction, labels)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def reduce(self):
        """Compute the final metric value"""
        self.avg = self.sum / self.count if self.count != 0 else 0

    def write(self, epoch: int):
        """Write the metric to TensorBoard

        Args:
            epoch (int): Epoch number
        """
        self.writer.add_scalar(self.name, self.avg, epoch)


class Tester:
    """
    Tester for the EFAuR model in a single GPU environment.
    Loads a trained checkpoint and evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): Trained model to test (e.g., a SiameseAuthorshipModel)
        test_data (DataLoader): Test data loader
        checkpoint_dir (str): Directory containing model checkpoints
        checkpoint_number (str | int | None, optional): Checkpoint identifier to load.
            Defaults to None.
        log_dir (str, optional): Directory for logging test metrics.
            Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_data: DataLoader,
        checkpoint_dir: str,
        checkpoint_number: str | int,
        log_dir: str,
    ):
        # Set device to GPU if available, else CPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine")
        self.device = torch.device("cuda")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_data = test_data
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.checkpoint_number = checkpoint_number

        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Load checkpoint if provided
        if checkpoint_number is not None:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"snapshot_{checkpoint_number}.pt"
            )
            if os.path.exists(checkpoint_path):
                logging.info("Loading checkpoint from %s", checkpoint_path)
                self._load_snapshot(checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"Checkpoint {checkpoint_path} does not exist. Please provide a valid checkpoint."
                )
                logging.warning(
                    "Checkpoint %s does not exist. Testing will proceed without loading weights.",
                    checkpoint_path,
                )

        # Initialize Tensorboard logging if log_dir is provided
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

    def _load_snapshot(self, snapshot_path: str):
        """Load the model state from a snapshot."""
        snapshot = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        logging.info("Loaded checkpoint from %s", snapshot_path)

    def test(self):
        """Evaluate the model on the test dataset and log metrics."""
        self.model.eval()

        # Define test metrics; adjust thresholds and metric functions as needed
        metrics: dict[str, tuple[AverageMeterSingleGPU, float | None]] = {
            "acc80": (
                AverageMeterSingleGPU("Accuracy80/test", Metrics.accuracy, self.writer),
                0.80,
            ),
            "acc65": (
                AverageMeterSingleGPU("Accuracy65/test", Metrics.accuracy, self.writer),
                0.65,
            ),
            "acc50": (
                AverageMeterSingleGPU("Accuracy50/test", Metrics.accuracy, self.writer),
                0.50,
            ),
            "acc30": (
                AverageMeterSingleGPU("Accuracy30/test", Metrics.accuracy, self.writer),
                0.30,
            ),
            "acc25": (
                AverageMeterSingleGPU("Accuracy25/test", Metrics.accuracy, self.writer),
                0.25,
            ),
            "acc20": (
                AverageMeterSingleGPU("Accuracy20/test", Metrics.accuracy, self.writer),
                0.20,
            ),
            "acc15": (
                AverageMeterSingleGPU("Accuracy15/test", Metrics.accuracy, self.writer),
                0.15,
            ),
            "acc10": (
                AverageMeterSingleGPU("Accuracy10/test", Metrics.accuracy, self.writer),
                0.10,
            ),
            "acc05": (
                AverageMeterSingleGPU("Accuracy05/test", Metrics.accuracy, self.writer),
                0.05,
            ),
            "acc01": (
                AverageMeterSingleGPU("Accuracy01/test", Metrics.accuracy, self.writer),
                0.01,
            ),
            "mse": (AverageMeterSingleGPU("MSE/test", Metrics.mse, self.writer), None),
            "mae": (AverageMeterSingleGPU("MAE/test", Metrics.mae, self.writer), None),
            "rmse": (
                AverageMeterSingleGPU("RMSE/test", Metrics.rmse, self.writer),
                None,
            ),
            "bce80": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy/test", Metrics.binary_cross_entropy, self.writer
                ),
                0.80,
            ),
            "bce65": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy65/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.65,
            ),
            "bce50": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy50/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.50,
            ),
            "bce30": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy30/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.30,
            ),
            "bce25": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy25/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.25,
            ),
            "bce20": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy20/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.20,
            ),
            "bce15": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy15/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.15,
            ),
            "bce10": (
                AverageMeterSingleGPU(
                    "BinaryCrossEntropy10/test",
                    Metrics.binary_cross_entropy,
                    self.writer,
                ),
                0.10,
            ),
        }

        with torch.no_grad():
            i = 0
            for i, (source1, source2, targets) in enumerate(self.test_data):

                source1 = source1.to(self.device)
                source2 = source2.to(self.device)
                targets = targets.to(self.device)
                # Forward pass: obtain predictions (assumes model returns two outputs)
                pred1, pred2 = self.model(source1, source2)
                # Compute predictions using a contrastive loss function or similar metric
                preds = Metrics.contrastive_loss(pred1, pred2, targets)

                if len(preds) == 0:
                    raise ValueError("No predictions in this batch during testing")
                for _, (metric, threshold) in metrics.items():
                    # Update the metric; assumes batch size is the number of targets
                    if threshold is None:
                        metric.update(preds, targets, len(source1))
                    else:
                        metric.update(preds, targets, len(source1), threshold)
                i += 1

        # Write metrics (using 0 as a dummy epoch value)
        for name, (metric, _) in metrics.items():
            metric.reduce()
            # metric.write(self.checkpoint_number)
            logging.info("Test %s: %.4f, count: %d, sum: %.4f", name, metric.avg, metric.count, metric.sum)

        # Return a dictionary of metric values for further analysis if needed
        return {name: metric.avg for name, (metric, _) in metrics.items()}


def test(
    batch_size: int = 4,
    checkpoint_number: int = 1,
    original_batch_size: int = 20,
    activation: str = "swish",
    pooling_method: str = "attention",
    learning_rate: float | str = "5e-05",
):
    """Test the model on the test dataset."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_id = f"{original_batch_size}_{learning_rate}_{pooling_method}_{activation}"
    log_dir = f"/home/lucasrpatten/efaur/logs/{train_id}"
    checkpoint_dir = f"/home/lucasrpatten/efaur/checkpoints/{train_id}"
    test_ds = AuthorshipPairDataset("/home/lucasrpatten/efaur/dataset/test/")
    model = SiameseAuthorshipModel(pooling_method, activation)
    train_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
    )
    tester = Tester(model, train_loader, checkpoint_dir, checkpoint_number, log_dir)
    return tester.test()


if __name__ == "__main__":
    test(batch_size=2, checkpoint_number=4)
