import csv
import torch
import random
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.dataset.dummy_ntu import DummyNTUDataset
from src.dataset.ntu120_skeleton import NTU120IndexDataset

XSUB_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
    25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
    50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74,
    78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
    93, 94, 95, 97, 98, 100, 103,
}


def build_loaders(config):
    seed = config["experiment"]["seed"]
    data_cfg = config["dataset"]
    
    name = data_cfg["name"]
    
    if name == "dummy_ntu":
        return get_dummy_ntu_loaders(data_cfg, seed)

    if name == "ntu120_skeleton":
        return get_ntu120_loaders(data_cfg) 

    raise ValueError(f"Unsupported dataset: {name}")

def get_dummy_ntu_loaders(config, seed):
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

def collate_ntu120(batch):
    xs, labels = zip(*batch)

    xs = pad_sequence([x.transpose(0, 1) for x in xs], batch_first=True)
    xs = xs.permute(0, 2, 1, 3, 4).contiguous()

    return xs, torch.stack(labels)

def get_ntu120_loaders(config):
    index_path = Path(config["index_path"]).resolve()
    protocol = config["protocol"]
    train_ratio = config["train_ratio"]
    test = config["test"]

    if protocol not in ("xsub", "xset"):
        raise ValueError(f"Unsupported protocol: {protocol}")

    with index_path.open("r", newline="", encoding="utf-8-sig") as f:
        samples = list(csv.DictReader(f))

    train_groups = defaultdict(list)
    test_samples = []

    for sample in sorted(samples, key=lambda s: s["sample_id"]):
        sample["path"] = str(index_path.parent / sample["path"])

        if protocol == "xsub":
            is_train = int(sample["subject"]) in XSUB_TRAIN_SUBJECTS
        else:
            is_train = int(sample["setup"]) % 2 == 0

        if is_train:
            train_groups[int(sample["action"])].append(sample)
        else:
            test_samples.append(sample)

    train_samples = []
    val_samples = []

    for action in sorted(train_groups):
        group = train_groups[action]
        random.shuffle(group)

        n_train = int(len(group) * train_ratio)

        train_samples.extend(group[:n_train])
        val_samples.extend(group[n_train:])

    train_dataset = NTU120IndexDataset(train_samples)
    val_dataset = NTU120IndexDataset(val_samples)

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
        "collate_fn": collate_ntu120,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    if test:
        test_loader = DataLoader(
            NTU120IndexDataset(test_samples),
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader

    return train_loader, val_loader