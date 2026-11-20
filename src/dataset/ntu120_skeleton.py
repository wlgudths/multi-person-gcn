import csv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from src.utils.ntu_skeleton import read_skeleton, skeleton_to_array


class NTU120IndexDataset(Dataset):
    # NTU120 공식 Cross-Subject
    XSUB_TRAIN_SUBJECT = {
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
        25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
        50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74,
        78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
        93, 94, 95, 97, 98, 100, 103,
    }

    def __init__(self, index_path, data_root, split, protocol, num_frames, val_ids):
        self.data_root = Path(data_root)
        self.num_frames = num_frames

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")

        if protocol == "xsub":
            group_key = "subject"
            train_ids = self.XSUB_TRAIN_SUBJECT

        elif protocol == "xset":
            group_key = "setup"
            train_ids = set(range(2, 33, 2))

        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        val_ids = set(val_ids) if val_ids is not None else set()

        if not val_ids.issubset(train_ids):
            raise ValueError("Validation IDs ust be part of the training IDs.")

        if split == "val" and not val_ids:
            raise ValueError("Validation IDs are required.")

        # CSV 기반 샘플 목록 구성
        self.samples = []

        with Path(index_path).open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                group_id = int(row[group_key])

                if split == "train":
                    include = (group_id in train_ids and group_id not in val_ids)
                
                elif split == "val":
                    include = group_id in val_ids

                else:
                    include = group_id not in train_ids

                if include:
                    self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return {
            "sample_id": sample["sample_id"],
            "path": str(self.data_root / sample["path"]),
            "setup": int(sample["setup"]),
            "camera": int(sample["camera"]),
            "subject": int(sample["subject"]),
            "repetition": int(sample["repetition"]),
            "action": int(sample["action"]),
        }