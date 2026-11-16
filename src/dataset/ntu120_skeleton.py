import csv
from pathlib import Path
from torch.utils.data import Dataset


class NTU120IndexDataset(Dataset):
    def __init__(self, index_path, data_root):
        self.data_root = Path(data_root)

        with Path(index_path).open("r", newline="", encoding="utf-8-sig") as f:
            self.samples = list(csv.DictReader(f))

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