import torch
import torch.nn as nn

class JointMotionInput(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [N, M, T, V, C], got {x.shape}")

        motion = torch.zeros_like(x)
        motion[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

        x = torch.cat([x, motion], dim=-1)
        x = self.proj(x)

        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    model = JointMotionInput(in_channels=6, hidden_dim=64)

    x = torch.randn(2, 3, 100, 25, 3)
    out = model(x)

    print("Input :", x.shape)
    print("Output:", out.shape)

    assert out.shape == (2, 3, 100, 25, 64)

    motion = torch.zeros_like(x)
    motion[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    joint_motion = torch.cat([x, motion], dim=-1)

    expected = model.proj(joint_motion)

    print("Projection check:", torch.allclose(out, expected))
    print("Forward PASS")