import math
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.graph import NTUGraph


class DummyNTUDataset(Dataset):
    def __init__(self, num_samples=128, num_classes=4, num_person=2, num_frames=100, in_channels=3, seed=42):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_person = num_person
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.seed = seed

        self.graph = NTUGraph(mode="spatial")
        self.num_joints = self.graph.num_node

        self.action_joints = [
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19]
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(self.seed + idx)
        label = idx % self.num_classes

        x = torch.randn(self.num_person, self.num_frames, self.num_joints, self.in_channels, generator=generator) * 0.02
        t = torch.linspace(0, 2 * math.pi, self.num_frames)

        joints = self.action_joints[label]
        axis = label % self.in_channels
        signal = torch.sin((label + 1) * t)

        x[0, :, joints, axis] += signal[:, None]

        return x, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
   dataset = DummyNTUDataset(num_samples=16, num_classes=4)
   print("===Dataset Test===")
   print("Dataset size:", len(dataset))

   for i in range(4):
       x, label = dataset[i]

       print(f"Sample {i}")
       print("Input :", x.shape)
       print("Label :", label.item())
       print("Mean  :", x.mean().item())
       print("Std   :", x.std().item())
       print("\n")

       assert x.shape == (2, 100, 25, 3)
       assert x.dtype == torch.float32
       assert 0 <= label.item() < 4

   print("Dataset test: PASS")

   print("===Dataloader Test===")

   loader = DataLoader(dataset, batch_size=4, shuffle=True)
   x, labels = next(iter(loader))

   print("Input :", x.shape)
   print("Label :", labels.shape)
   print("Labels:", labels)

   assert x.shape == (4, 2, 100, 25, 3)
   assert labels.shape == (4,)

   print("DataLoader test: PASS")