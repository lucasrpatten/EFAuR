"""
Training Script(s) for the EFAuR model
These scripts support ddp (Dynamic Distributed Parallel) training

Author: Lucas Patten
"""

import logging
import os
import re
from typing import Callable
import typing
import torch

from torch import distributed
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import AuthorshipPairDataset
from metrics import ContrastiveLoss, Metrics
from models import SiameseAuthorshipModel


def ddp_setup():
    """Sets up Distributed Data Parallel for training"""
    distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class AverageMeter:
    """Metric Tracker for Tensorboard

    Args:
        name (str): Metric name
        metric (Callable): Metric function
        writer (SummaryWriter): Tensorboard writer
    """

    def __init__(self, name: str, metric: Callable, writer: SummaryWriter):
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
            predictions (torch.Tensor): Predictions from model
            labels (torch.Tensor): True labels
            n (int, optional): Number of samples. Defaults to 1.
        """

        if threshold is not None:
            self.val = self.metric(prediction, labels, threshold)
        else:
            self.val = self.metric(prediction, labels)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """Reduce the metric across all processes"""
        total = torch.tensor([self.sum, self.count]).cuda()
        distributed.all_reduce(total, op=distributed.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def write(self, epoch: int):
        """Write the metric to tensorboard

        Args:
            epoch (int): Epoch number
        """
        self.writer.add_scalar(self.name, self.avg, epoch)


class Trainer:
    """
    Trainer for the EFAuR model.
    Contains training loop.
    Supports checkpointing and Tensorboard logging.

    Args:
        model (torch.nn.Module): Initialized Model to train (Usually a SiameseAuthorshipModel)
        train_data (DataLoader): Training data
        val_data (DataLoader): Validation data
        optimizer (torch.optim.Optimizer): Initialized Optimizer
        save_interval (int): Number of epochs between saving checkpoints
        log_dir (str): Directory to save Tensorboard logs
        checkpoint_dir (str): Directory to save checkpoints
        checkpoint_number (str | int | None, optional): What checkpoint number to load.
        Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_interval: int,
        log_dir: str,
        checkpoint_dir: str,
        checkpoint_number: str | int | None = None,
    ):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_interval = save_interval
        self.epochs_run = 1
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        # self.criterion =

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if checkpoint_number:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"snapshot_{checkpoint_number}.pt"
            )
            if os.path.exists(checkpoint_path):
                logging.info(
                    "[GPU:%s] Loading snapshot from %s", self.gpu_id, checkpoint_path
                )
                self._load_snapshot(checkpoint_path)
            else:
                logging.warning(
                    "[GPU:%s] Snapshot %s does not exist, starting from epoch 1",
                    self.gpu_id,
                    checkpoint_path,
                )
        self.writer = SummaryWriter(log_dir)
        self.model = DDP(
            self.model, device_ids=[self.gpu_id], find_unused_parameters=True
        )

        self.batch_count = 0
        self.train_loss = 0.0
        self.train_acc = AverageMeter("Accuracy/train", Metrics.accuracy, self.writer)
        self.train_acc65 = AverageMeter(
            "Accuracy65/train", Metrics.accuracy, self.writer
        )
        self.train_acc50 = AverageMeter(
            "Accuracy50/train", Metrics.accuracy, self.writer
        )

    def _save_snapshot(self, epoch: int):
        model_state_dict = (
            self.model.module.state_dict()
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model.state_dict()
        )
        snapshot = {
            "MODEL_STATE": model_state_dict,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        snapshot_path = os.path.join(self.checkpoint_dir, f"snapshot_{epoch}.pt")
        torch.save(snapshot, snapshot_path)
        logging.info(
            "[GPU:%s] Snapshot of epoch %s saved to %s",
            self.gpu_id,
            epoch,
            snapshot_path,
        )

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"] + 1
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        logging.info("[GPU:%s] Snapshot loaded from %s", self.gpu_id, snapshot_path)

    def _run_batch(self, source1, source2, targets):
        criterion = ContrastiveLoss()
        self.optimizer.zero_grad()
        embeddings1, embeddings2 = self.model(source1, source2)
        loss = criterion(embeddings1, embeddings2, targets)
        loss.backward()
        self.optimizer.step()
        self.train_loss += loss.item()
        train_acc = Metrics.binary_contrastive_tensor(embeddings1, embeddings2, targets)
        train_acc65 = Metrics.binary_contrastive_tensor(
            embeddings1, embeddings2, targets, threshold=0.65
        )
        train_acc50 = Metrics.binary_contrastive_tensor(
            embeddings1, embeddings2, targets, threshold=0.5
        )
        self.train_acc.update(train_acc, targets, len(source1))
        self.train_acc65.update(train_acc65, targets, len(source1), 0.65)
        self.train_acc50.update(train_acc50, targets, len(source1), 0.5)
        self.batch_count += 1

    def _run_epoch(self, epoch: int):
        batch_size = len(next(iter(self.train_data))[2])
        logging.info(
            "Train: [GPU:%s] Epoch %s | Batchsize %s | Steps %s",
            self.gpu_id,
            epoch,
            batch_size,
            len(self.train_data),
        )

        self.train_data.sampler.set_epoch(epoch)  # type: ignore
        for source1, source2, targets in self.train_data:
            source1 = source1.to(self.gpu_id)
            source2 = source2.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source1, source2, targets)

        # Syncrhronize gradients processes
        distributed.barrier()

        # Sum the total loss across all GPUs
        total_loss_tensor = torch.tensor(self.train_loss).cuda()
        total_batches_tensor = torch.tensor(self.batch_count).cuda()
        distributed.all_reduce(total_loss_tensor, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(total_batches_tensor, op=distributed.ReduceOp.SUM)
        average_loss = total_loss_tensor.item() / total_batches_tensor.item()
        self.writer.add_scalar("Loss/train", average_loss, epoch)
        self.batch_count = 0
        self.train_loss = 0.0

        self.train_acc.all_reduce()
        self.train_acc65.all_reduce()
        self.train_acc50.all_reduce()

        self.train_acc.write(epoch)
        self.train_acc65.write(epoch)
        self.train_acc50.write(epoch)

        self.train_acc.reset()
        self.train_acc65.reset()
        self.train_acc50.reset()

    def _run_eval(self, epoch):  # pylint: disable=too-many-locals
        batch_size = len(next(iter(self.val_data))[2])
        logging.info(
            "Eval: [GPU:%s] Epoch %s | Batchsize %s | Steps %s",
            self.gpu_id,
            epoch,
            batch_size,
            len(self.val_data),
        )

        with torch.no_grad():
            # metric : tuple(AverageMeter, pred_type, threshold)
            metrics: dict[str, tuple[AverageMeter, float | None]] = {
                "acc100": (
                    AverageMeter("Accuracy/val", Metrics.accuracy, self.writer),
                    1.0,
                ),
                "acc80": (
                    AverageMeter("Accuracy65/val", Metrics.accuracy, self.writer),
                    0.80,
                ),
                "acc65": (
                    AverageMeter("Accuracy50/val", Metrics.accuracy, self.writer),
                    0.65,
                ),
                "acc50": (
                    AverageMeter("Accuracy/val", Metrics.accuracy, self.writer),
                    0.5,
                ),
                "acc25": (
                    AverageMeter("Accuracy25/val", Metrics.accuracy, self.writer),
                    0.25,
                ),
                "mse": (AverageMeter("MSE/val", Metrics.mse, self.writer), None),
                "mae": (AverageMeter("MAE/val", Metrics.mae, self.writer), None),
                "rmse": (AverageMeter("RMSE/val", Metrics.rmse, self.writer), None),
                "bce": (
                    AverageMeter(
                        "BinaryCrossEntropy/val",
                        Metrics.binary_cross_entropy,
                        self.writer,
                    ),
                    None,
                ),
                "bce65": (
                    AverageMeter(
                        "BinaryCrossEntropy65/val",
                        Metrics.binary_cross_entropy,
                        self.writer,
                    ),
                    0.65,
                ),
                "bce50": (
                    AverageMeter(
                        "BinaryCrossEntropy50/val",
                        Metrics.binary_cross_entropy,
                        self.writer,
                    ),
                    0.5,
                ),
            }

            for source1, source2, targets in self.val_data:
                source1 = source1.to(self.gpu_id)
                source2 = source2.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                pred1, pred2 = self.model(source1, source2)

                preds = Metrics.contrastive_loss(pred1, pred2, target=targets)
                if len(preds) == 0:
                    raise ValueError("No predictions in this batch")
                for _, (metric, threshold) in metrics.items():
                    if threshold is None:
                        metric.update(preds, targets, batch_size)
                    else:
                        metric.update(preds, targets, batch_size, threshold)

            distributed.barrier()

            for _, (metric, _) in metrics.items():
                metric.all_reduce()
                metric.write(epoch)

    def train(self, max_epochs: int):
        """Run the model train loop

        Args:
            max_epochs (int): Maximum number of epochs to train for
        """
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs + 1):
            if epoch == 1:
                self._run_eval(0)
            self._run_epoch(epoch)
            self._run_eval(epoch)
            if self.gpu_id == 0 and epoch % self.save_interval == 0:
                self._save_snapshot(epoch)


def latest_snapshot_number(checkpoint_dir: str) -> int | None:
    """Get the latest snapshot number in the checkpoint directory

    Args:
        checkpoint_dir (str): Checkpoint directory

    Returns:
        int | None: Latest snapshot number
    """
    if not os.path.exists(checkpoint_dir):
        logging.warning(
            "Checkpoint directory does not exist: %s. Starting from scratch",
            checkpoint_dir,
        )
        return None

    snapshot_pattern = re.compile(r"snapshot_(\d+)\.pt")
    files = os.listdir(checkpoint_dir)
    if not files:
        logging.warning(
            "No files exist in checkpoint directory: %s. Starting from scratch",
            checkpoint_dir,
        )
        return None
    snapshot = -1
    for filename in files:
        match = snapshot_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > snapshot:
                snapshot = epoch
    if snapshot == -1:
        logging.warning(
            "No snapshots found in directory: %s. Starting from scratch", checkpoint_dir
        )
        return None
    return snapshot


def train(
    batch_size: int = 16,
    epochs: int = 320,
    learning_rate: float = 0.0001,
    pooling_method: str = "max",
    activation_function: str = "relu",
):
    """Train the model

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        epochs (int, optional): Number of epochs. Defaults to 150.
        learning_rate (float, optional): Learning rate. Defaults to 0.0005.
        pooling_method (str, optional): Pooling method. Defaults to "max".
        activation_function (str, optional): Activation function. Defaults to "relu".
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ddp_setup()
    torch.set_grad_enabled(True)
    train_ds = AuthorshipPairDataset(
        "/home/lucasrp/nobackup/archive/gutenberg/dataset/train/"
    )
    val_ds = AuthorshipPairDataset(
        "/home/lucasrp/nobackup/archive/gutenberg/dataset/val/"
    )
    model = SiameseAuthorshipModel(pooling_method, activation_function)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_id = f"bs{batch_size}_lr{learning_rate}"
    log_dir = f"/home/lucasrp/nobackup/archive/efaur/logs/{train_id}/"
    checkpoint_dir = f"/home/lucasrp/nobackup/archive/efaur/checkpoints/{train_id}/"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_ds),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(val_ds),
    )
    latest_epoch = latest_snapshot_number(checkpoint_dir)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        1,
        log_dir,
        checkpoint_dir,
        latest_epoch,
    )
    trainer.train(epochs)
    distributed.destroy_process_group()


if __name__ == "__main__":
    train(batch_size=20, learning_rate=0.0001)
