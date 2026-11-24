import csv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from src.utils.ntu_skeleton import read_skeleton, skeleton_to_array, select_bodies_by_motion


class NTU120IndexDataset(Dataset):
    def __init__(self, samples):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = Path(sample["path"])

        # Skeleton 파싱
        skeleton = read_skeleton(path)
        data, mask, body_ids = skeleton_to_array(skeleton)

        data, _, _ = select_bodies_by_motion(data, mask, body_ids)

        x = np.zeros((2, data.shape[1], 25, 3), dtype=np.float32)
        x[:len(data)] = data

        # 1 ~ 120 -> 0 ~ 119
        label = int(sample["action"]) - 1

        return (torch.from_numpy(x), torch.tensor(label, dtype=torch.long))