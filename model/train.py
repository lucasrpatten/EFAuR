import logging
import os
import torch
import torch.distributed as distributed
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from authorship_model import SiameseAuthorshipModel
from dataset import AuthorshipPairDataset


def ddp_setup():
    """Sets up Distributed Data Parallel for training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def accuracy(pred: torch.Tensor, labels: torch.Tensor):
    device=labels.get_device()
    predictions = torch.Tensor([1 if p >= 0.8 else 0 for p in pred]).float().cuda(device)
    correct_predictions = (predictions == labels).sum().item()

    total_predictions = labels.size(0)
    return correct_predictions / total_predictions


class Trainer:
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
        self.epochs_run = 0
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
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
        }
        snapshot_path = os.path.join(self.checkpoint_dir, f"snapshot_{epoch}.pt")
        torch.save(snapshot, snapshot_path)
        logging.info("Snapshot of epoch %s saved to %s", epoch, snapshot_path)

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        logging.info("Resuming training from snapshot at Epoch %s", self.epochs_run)

    def _run_batch(self, source1, source2, targets):
        self.optimizer.zero_grad()
        output = self.model(source1, source2)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy(output, targets)

    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[1])
        logging.info(
            "[GPU:%s] Epoch %s | Batchsize %s | Steps %s",
            self.gpu_id,
            epoch,
            batch_size,
            len(self.train_data),
        )

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        self.train_data.sampler.set_epoch(epoch)  # type: ignore
        for source1, source2, targets in self.train_data:
            source1 = source1.to(self.gpu_id)
            source2 = source2.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss, acc = self._run_batch(source1, source2, targets)
            batch_size = len(source1)
            total_acc += acc * batch_size
            total_samples += batch_size
            total_loss += batch_loss * batch_size

        # Syncrhronize gradients processes
        distributed.barrier()

        # Sum the total loss across all GPUs
        total_loss_tensor = torch.tensor(total_loss).cuda()
        total_samples_tensor = torch.tensor(total_samples).cuda()
        total_acc_tensor = torch.tensor(total_acc).cuda()
        distributed.all_reduce(total_loss_tensor, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(total_samples_tensor, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(total_acc_tensor, op=distributed.ReduceOp.SUM)

        average_acc = total_acc_tensor.item() / total_samples_tensor.item()
        average_loss = total_loss_tensor.item() / total_samples_tensor.item()
        self.writer.add_scalar("AverageLoss/train", average_loss, epoch)
        self.writer.add_scalar("AverageAccuracy/train", average_acc, epoch)

    def _run_val(self, epoch):
        batch_size = len(next(iter(self.val_data))[0])
        with torch.no_grad():
            total_loss = 0.0
            total_acc = 0.0
            total_samples = 0

            for source1, source2, targets in self.val_data:
                source1 = source1.to(self.gpu_id)
                source2 = source2.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source1, source2)
                loss = F.cross_entropy(output, targets)
                acc = accuracy(output, targets)

                batch_size = len(source1)
                total_acc += acc * batch_size
                total_samples += batch_size
                total_loss += loss * batch_size

            distributed.barrier()

            total_loss_tensor = torch.tensor(total_loss).cuda()
            total_samples_tensor = torch.tensor(total_samples).cuda()
            total_acc_tensor = torch.tensor(total_acc).cuda()
            distributed.all_reduce(total_loss_tensor, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_samples_tensor, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_acc_tensor, op=distributed.ReduceOp.SUM)

            average_acc = total_acc_tensor.item() / total_samples_tensor.item()
            average_loss = total_loss_tensor.item() / total_samples_tensor.item()
            self.writer.add_scalar("Loss/val", average_loss, epoch)
            self.writer.add_scalar("Accuracy/val", average_acc, epoch)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self._run_val(epoch)
            if self.gpu_id == 0 and epoch % self.save_interval == 0:
                self._save_snapshot(epoch)


def train(batch_size: int = 2, epochs: int = 15, learning_rate: float = 0.0025):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ddp_setup()
    torch.set_grad_enabled(True)
    train_ds = AuthorshipPairDataset("/home/lucasrp/compute/gutenberg/dataset/train/")
    val_ds = AuthorshipPairDataset("/home/lucasrp/compute/gutenberg/dataset/val/")
    model = SiameseAuthorshipModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log_dir = "/home/lucasrp/compute/efaur/logs/original/"
    checkpoint_dir = "/home/lucasrp/compute/efaur/checkpoints/original/"
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
    destroy_process_group()

if __name__ == "__main__":
    train()