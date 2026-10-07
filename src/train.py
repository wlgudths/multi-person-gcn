import torch
import argparse

from src.dataset.loader import build_loaders
from src.models.protogcn import ProtoGCNBaseline
from src.loss import ProtoGCNLoss
from src.trainer import Trainer
from src.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dummy_ntu.yaml")

    return parser.parse_args()

def train():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config)

    model_cfg = config["model"]
    loss_cfg = config["loss"]
    opt_cfg = config["optimizer"]

    model = ProtoGCNBaseline(num_classes=model_cfg["num_classes"]).to(device)
    criterion = ProtoGCNLoss(num_classes=model_cfg["num_classes"], csc_weight=loss_cfg["csc_weight"]).to(device)

    # 추후 수정 예정
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=opt_cfg["lr"],
        momentum=opt_cfg["momentum"],
        weight_decay=opt_cfg["weight_decay"]
    ) 

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, config=config)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()