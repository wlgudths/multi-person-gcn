import torch


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

if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(8, 2, 100, 25, 3)

    stats = compute_joint_statistics(x)

    for name, value in stats.items():
        print(f"{name:20s}: {value.shape}")

    assert stats["speed_mean"].shape == (25,)
    assert stats["acceleration_mean"].shape == (25,)
    assert stats["range_mean"].shape == (25,)
    assert stats["path_mean"].shape == (25,)