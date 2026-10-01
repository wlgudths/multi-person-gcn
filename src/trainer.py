import csv
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, criterion, optimizer, device, epochs=150, scheduler=None, val_interval=5, save_dir="experiments", exp_name="gcn", config=None, use_amp=True, grad_clip=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.val_interval = val_interval

        self.epochs = epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device.type == "cuda"

        self.start_epoch = 1
        self.best_acc = 0.0

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rundir = Path(save_dir) / exp_name / run_name
        self.rundir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.rundir / "tensorboard")
        self.csv_path = self.run_dir / "train.csv"

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if config is not None:
            with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, default=str)

        self._init_csv()

        print("Experiment :", self.run_dir)
        print("AMP        :", self.use_amp)

    def _init_csv(self):
        if self.csv_path.exists():
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "ttrain_loss", "train_ce", "trian_csc", "train_acc", "val_loss", "val_acc", "lr"])

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()

        total_loss = 0.0
        total_ce = 0.0
        total_csc = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(loader, desc=f"Train {epoch}/{self.epochs}")

        for x, labels in pbar:
            x = x.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits, reconstructed_graph = self.model(x)
                losses = self.criterion(logits, reconstructed_graph, labels)
                loss = losses["loss"]

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.criterion.parameters()),
                    self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_ce += losses["loss_ce"].item() * batch_size
            total_csc += losses["loss_csc"].item() * batch_size

            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += batch_size

            pbar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{100.0 * total_correct / total_samples:.2f}%"
            )

        return {
            "loss": total_loss / total_samples,
            "loss_ce": total_ce / total_samples,
            "loss_csc": total_csc / total_samples,
            "acc": 100.0 * total_correct / total_samples
        }

    @torch.no_grad()
    def validate(self, loader, epoch=None):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(loader, desc=f"Val {epoch}/{self.epochs}")

        for x, labels in pbar:
            x = x.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            total_corrrect += (pred == labels).sum().item()
            total_samples += batch_size

            pbar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{100.0 * total_correct / total_samples:.2f}%"
            )

        return {
            "loss": total_loss / total_samples,
            "acc": 100.0 * total_correct / total_samples
        }


    def fit(self, train_loader, val_loader=None):
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_one_epoch(train_loader, epoch)

            val_metrics = None
        
            if val_loader is not None:
                if epoch % self.val_interval == 0 or epoch == self.epochs:
                    val_metrics = self.validate(val_loader, epoch)

            lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

                self.save_checkpoint("last.pth", epoch)

            if val_metrics is not None and val_metrics["acc"] > self.best_acc:
                self.best_acc = val_metrics["acc"]
                self.save_checkpoint("best.pth", epoch)

            self._log(epoch, train_metrics, val_metrics, lr)

            msg = (
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_metrics['loss']:.4f} | "
                f"Train Acc {train_metrics['acc']:.2f}%"
            )

            if val_metrics is not None:
                msg += (
                    f" | Val Loss {val_metrics['loss']:.4f}"
                    f" | Val Acc {val_metrics['acc']:.2f}%"
                )

            msg += f" | LR {lr:.6f}"

            print(msg)

        self.writer.close()