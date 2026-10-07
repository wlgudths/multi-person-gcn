from torch.utils.data import DataLoader
from src.dataset.dummy_ntu import DummyNTUDataset


def get_dummy_ntu_loader(batch_size=8, num_workers=0, train_samples=64, val_samples=32, num_classes=4, pin_memory=True):
    train_dataset = DummyNTUDataset(num_samples=train_samples, num_classes=num_classes, seed=42)
    val_dataset = DummyNTUDataset(num_samples=val_samples, num_classes=num_classes, seed=24)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader