from torch.utils.data import DataLoader
from src.dataset.dummy_ntu import DummyNTUDataset


def build_loaders(config):
    seed = config["experiment"]["seed"]
    data_cfg = config["dataset"]
    
    name = data_cfg["name"]
    
    if name == "dummy_ntu":
        return get_dummy_ntu_loader(data_cfg, seed)

    raise ValueError(f"Unsupported dataset: {name}")



def get_dummy_ntu_loader(config, seed):
    train_dataset = DummyNTUDataset(
        num_samples=config["train_samples"],
        num_classes=config["num_classes"],
        seed=seed)
    
    val_dataset = DummyNTUDataset(
        num_samples=config["val_samples"],
        num_classes=config["num_classes"],
        seed=seed + 10000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    return train_loader, val_loader