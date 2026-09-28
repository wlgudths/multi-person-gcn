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

        total_channels = (rem_channels + mid_channels * (num_branches - 1))

        self.transform = nn.Sequential(nn.BatchNorm2d(total_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(total_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.add_coeff = nn.Parameter(torch.zeros(num_joint))

    def forward(self, x):
        _, _, _, V = x.shape

        global_joint = x.mean(dim=-1, keepdim=True)

        x = torch.cat([x, global_joint], dim=-1)

        outputs = []

        for branch in self.branches:
            outputs.append(branch(x))

        out = torch.cat(outputs, dim=1)

        local_feat = out[..., :V]

        global_feat = out[..., :V]

        global_feat = torch.einsum("nct, v->nctv", global_feat, self.add_coeff[:V])

        out = local_feat + global_feat

        out = self.transform(out)
        out = self.bn(out)
        out = self.dropout(out)

        return out

class UnitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, ratio=0.125):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_subsets = A.shape[0]

        self.mid_channels = int(ratio * out_channels)

        self.A = nn.Parameter(A.clone())

        hidden_channels = (self.mid_channels * self.num_subsets)

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.post = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))

        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if in_channels == out_channels:
            self.down = nn.Identity()

        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.shape

        residual = self.down(x)

        # Base adjacency [K, V, V]
        A = self.A[None, :, None, None, :, :]

        # Feature projection
        pre_x = self.pre(x)
        pre_x = pre_x.reshape(N, self.num_subsets, self.mid_channels, T, V)

        # Motion topology enhancement
        x1 = self.conv1(x)
        x1 = x1.reshape(N, self.num_subsets, self.mid_channels, T, V)

        x2 = self.conv2(x)
        x2 = x2.reshape(N, self.num_subsets, self.mid_channels, T, V)

        # Temporal average
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)

        # Inter topology
        diff = (x1.unsqueeze(-1) - x2.unsqueeze(-2))
        inter_graph = torch.tanh(diff)
        inter_graph = (inter_graph * self.alpha[0])

        A = A + inter_graph

        # Intra topology
        intra_graph = torch.einsum("nkctv,nkctw->nktvm", x1, x2)
        intra_graph = intra_graph[:, :, None]
        intra_graph = torch.softmax(intra_graph, dim=-2)
        intra_graph = (intra_graph * self.beta[0])

        A = A + intra_graph

        # Graph convolution
        A = A.squeeze(3)

        out = torch.einsum("nkctv,nkcvw->nkctw", pre_x, A)
        out = out.contiguous()
        out = out.reshape(N, -1, T, V)
        out = self.post(out)
        out = self.bn(out)
        out = out + residual
        out = self.relu(out)

        # Graph feature for PRN
        graph_feature = (inter_graph + intra_graph)
        graph_feature = graph_feature.squeeze(3)
        graph_feature = graph_feature.reshape(N, -1, V, V)

        return out, graph_feature