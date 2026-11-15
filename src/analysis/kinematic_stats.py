import torch
from collections import defaultdict
from torch.utils.data import DataLoader


def compute_kinematics(x):
    if x.ndim != 5:
        raise ValueError(f"Expected [N, M, T, V, C], got {x.shape}")

    velocity = torch.zeros_like(x)
    velocity[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

    acceleration = torch.zeros_like(x)
    acceleration[:, :, 2:] = x[:, :, 2:] - velocity[:, :, 1:-1]

    speed = torch.norm(velocity, dim=-1)
    accel_mag = torch.norm(acceleration, dim=-1)

    pos_min = x.amin(dim=2)
    pos_max = x.amax(dim=2)
    motion_range = torch.norm(pos_max - pos_min, dim=-1)

    path_length = speed.sum(dim=2)

    return {
        "speed": speed,
        "acceleration": accel_mag,
        "motion_range": motion_range,
        "path_length": path_length,
    }

def compute_joint_statistics(x):
    kinematics = compute_kinematics(x)

    speed = kinematics["speed"][:, :, 1:]
    acceleration = kinematics["acceleration"][:, :, 2:]
    motion_range = kinematics["motion_range"]
    path_length = kinematics["path_length"]

    return {
        "speed_mean": speed.mean(dim=(0, 1, 2)),
        "speed_max": speed.amax(dim=(0, 1, 2)),
        "speed_std": speed.std(dim=(0, 1, 2)),

        "acceleration_mean": acceleration.mean(dim=(0, 1, 2)),
        "acceleration_max": acceleration.amax(dim=(0, 1, 2)),
        "acceleration_std": acceleration.std(dim=(0, 1, 2)),

        "range_mean": motion_range.mean(dim=(0, 1)),
        "range_max": motion_range.amax(dim=(0, 1)),

        "path_mean": path_length.mean(dim=(0, 1)),
        "path_max": path_length.amax(dim=(0, 1)),
    }

@torch.no_grad()
def compute_action_statistics(loader, num_classes):
    action_samples = defaultdict(list)

    for x, labels in loader:
        x = x.float()
        labels = labels.long()

        for action_id in labels.unique():
            aciton_id = int(action_id.item())
            action_samples[action_id].append(x[labels == action_id].cpu())

    action_stats = {}

    for action_id in range(num_classes):
        if action_id not in action_samples:
            continue

        x = torch.cat(action_samples[action_id], dim=0)

        stats = compute_joint_statistics(x)
        stats["num_samples"] = x.shape[0]

        action_stats[action_id] = stats

    return action_stats

def print_action_statistics(action_stats):
    for action_id, stats in action_stats.items():
        print(f"\nAction {action_id} | Samples: {stats['num_samples']}")

        print("Speed mean       :", stats["speed_mean"].mean().item())
        print("Acceleration mean:", stats["acceleration_mean"].mean().item())
        print("Range mean       :", stats["range_mean"].mean().item())
        print("Path mean        :", stats["path_mean"].mean().item())


if __name__ == "__main__":
    # config 작성 후 test
    pass