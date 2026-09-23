import torch
import torch.nn as nn


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm=True, dropout=0.0):
        super().__init__()

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(pad, 0), dilation=(dilation, 1))

        if norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        self.dropout = nn.Dropout(dropout, inplace=True)

    def foeward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)

        return x

class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_joint=25, dropout=0.0, ms_cfg=None):
        super().__init__()

        if ms_cfg is None:
            ms_cfg = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"]

        self.num_joint = num_joint
        self.relu = nn.ReLU(inplace=True)

        num_branches = len(ms_cfg)

        mid_channels = out_channels // num_branches

        rem_channels = (out_channels - mid_channels * (num_branches - 1))

        branches = []

        for index, cfg in enumerate(ms_cfg):
            branch_channels = (rem_channels if index == 0 else mid_channels)

            if cfg == "1x1":
                branch = nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=(stride, 1))

            elif cfg[0] == "max":
                kernel_size = cfg[1]

                branch = nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(kernel_size // 2, 0))
                )

            else:
                kernel_size = cfg[0]
                dilation = cfg[1]

                branch = nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),

                    UnitTCN(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, norm=False)
                )

                branches.append(branch)

            self.branches = nn.ModuleList(branches)

