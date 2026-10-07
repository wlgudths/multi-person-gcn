import torch

from src.dataset.loader import get_dummy_ntu_loader
from src.models.protogcn import ProtoGCNBaseline
from src.loss import ProtoGCNLoss
from src.trainer import Trainer
from src.utils import set_seed

def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    if torch.cuda.is_available(): print("GPU: ", torch.cuda.get_device_name(0))

    train_loader, val_loader = get_dummy_ntu_loader(batch_size=8, train_samples=64, val_samples=32, num_classes=4)

    model = ProtoGCNBaseline(num_classes=4).to(device)
    criterion = ProtoGCNLoss(num_classes=4).to(device)

    # 추후 수정 예정
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    ) 

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
        val_interval=2,
        exp_name="dummy_ntu_test",
        use_amp=True
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()