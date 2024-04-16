"""
Training Script(s) for the EFAuR model
These scripts support ddp (Dynamic Distributed Parallel) training

Author: Lucas Patten
"""

import logging
import os
from typing import Callable
import torch
import torch.nn.functional as F

from torch import distributed
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import AuthorshipPairDataset
from metrics import Metrics
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
        predictions: torch.Tensor,
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
            self.val = self.metric(predictions, labels, threshold)
        else:
            self.val = self.metric(predictions, labels)
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

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if checkpoint_number:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"snapshot_{checkpoint_number}.pt"
            )
            if os.path.exists(checkpoint_path):
                logging.info("Loading snapshot from %s", checkpoint_path)
                self._load_snapshot(checkpoint_path)
            else:
                logging.warning(
                    "Snapshot %s does not exist, starting from epoch 1", checkpoint_path
                )
        self.writer = SummaryWriter(log_dir)
        self.model = DDP(
            self.model, device_ids=[self.gpu_id], find_unused_parameters=True
        )

        self.batch_count = 0
        self.train_loss = 0.0
        self.train_acc = AverageMeter("Accuracy/train", Metrics.accuracy, self.writer)

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
        logging.info("Snapshot of epoch %s saved to %s", epoch, snapshot_path)

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]+1
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        logging.info("Resuming training from snapshot at Epoch %s", self.epochs_run-1)

    def _run_batch(self, source1, source2, targets):
        self.optimizer.zero_grad()
        output = self.model(source1, source2)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        self.train_loss += loss.item()
        self.train_acc.update(output, targets, len(source1))
        self.batch_count += 1

    def _run_epoch(self, epoch):
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
        self.train_acc.write(epoch)
        self.train_acc.reset()

    def _run_val(self, epoch):  # pylint: disable=too-many-locals
        batch_size = len(next(iter(self.val_data))[2])
        logging.info(
            "Eval: [GPU:%s] Epoch %s | Batchsize %s | Steps %s",
            self.gpu_id,
            epoch,
            batch_size,
            len(self.val_data),
        )
        with torch.no_grad():
            loss = AverageMeter("Loss/val", Metrics.cross_entropy, self.writer)
            acc = AverageMeter("Accuracy/val", Metrics.accuracy, self.writer)
            precision = AverageMeter("Precision/val", Metrics.precision, self.writer)
            recall = AverageMeter("Recall/val", Metrics.recall, self.writer)
            f1 = AverageMeter("F1Score/val", Metrics.f1_score, self.writer)
            mse = AverageMeter("MSE/val", Metrics.mse, self.writer)
            mae = AverageMeter("MAE/val", Metrics.mae, self.writer)
            rmse = AverageMeter("RMSE/val", Metrics.rmse, self.writer)
            bce = AverageMeter(
                "BinaryCrossEntropy/val", Metrics.binary_cross_entropy, self.writer
            )
            acc65 = AverageMeter("Accuracy65/val", Metrics.accuracy, self.writer)
            precision65 = AverageMeter(
                "Precision65/val", Metrics.precision, self.writer
            )
            recall65 = AverageMeter("Recall65/val", Metrics.recall, self.writer)
            f1_65 = AverageMeter("F1Score65/val", Metrics.f1_score, self.writer)
            bce65 = AverageMeter(
                "BinaryCrossEntropy65/val", Metrics.binary_cross_entropy, self.writer
            )

            for source1, source2, targets in self.val_data:
                source1 = source1.to(self.gpu_id)
                source2 = source2.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                predictions = self.model(source1, source2)
                loss.update(predictions, targets, batch_size)
                acc.update(predictions, targets, batch_size)
                precision.update(predictions, targets, batch_size)
                recall.update(predictions, targets, batch_size)
                f1.update(predictions, targets, batch_size)
                mse.update(predictions, targets, batch_size)
                mae.update(predictions, targets, batch_size)
                rmse.update(predictions, targets, batch_size)
                bce.update(predictions, targets, batch_size)
                acc65.update(predictions, targets, batch_size, threshold=0.65)
                precision65.update(predictions, targets, batch_size, threshold=0.65)
                recall65.update(predictions, targets, batch_size, threshold=0.65)
                f1_65.update(predictions, targets, batch_size, threshold=0.65)
                bce65.update(predictions, targets, batch_size, threshold=0.65)

            distributed.barrier()

            loss.all_reduce()
            acc.all_reduce()
            precision.all_reduce()
            recall.all_reduce()
            f1.all_reduce()
            mse.all_reduce()
            mae.all_reduce()
            rmse.all_reduce()
            bce.all_reduce()
            acc65.all_reduce()
            precision65.all_reduce()
            recall65.all_reduce()
            f1_65.all_reduce()
            bce65.all_reduce()

            loss.write(epoch)
            acc.write(epoch)
            precision.write(epoch)
            recall.write(epoch)
            f1.write(epoch)
            mse.write(epoch)
            mae.write(epoch)
            rmse.write(epoch)
            bce.write(epoch)
            acc65.write(epoch)
            precision65.write(epoch)
            recall65.write(epoch)
            f1_65.write(epoch)
            bce65.write(epoch)

    def train(self, max_epochs: int):
        """Run the model train loop

        Args:
            max_epochs (int): Maximum number of epochs to train for
        """
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs + 1):
            if epoch == 1:
                self._run_val(0)
            self._run_epoch(epoch)
            self._run_val(epoch)
            if self.gpu_id == 0 and epoch % self.save_interval == 0:
                self._save_snapshot(epoch)


def train(batch_size: int = 16, epochs: int = 320, learning_rate: float = 0.0001):
    """Train the model

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        epochs (int, optional): Number of epochs. Defaults to 150.
        learning_rate (float, optional): Learning rate. Defaults to 0.0005.
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ddp_setup()
    torch.set_grad_enabled(True)
    train_ds = AuthorshipPairDataset("/home/lucasrp/compute/gutenberg/dataset/train/")
    val_ds = AuthorshipPairDataset("/home/lucasrp/compute/gutenberg/dataset/val/")
    model = SiameseAuthorshipModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log_dir = "/home/lucasrp/compute/efaur/logs/try3/"
    checkpoint_dir = "/home/lucasrp/compute/efaur/checkpoints/try3/"
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
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, 1, log_dir, checkpoint_dir
    )
    trainer.train(epochs)
    distributed.destroy_process_group()


if __name__ == "__main__":
    train()
