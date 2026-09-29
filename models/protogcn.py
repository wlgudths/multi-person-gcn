import torch
import torch.nn as nn

from .graph import NTUGraph
from .layers import GCNBlock


class PrototypeReconstructionNetwork(nn.Module):
    def __init__(self, dim, num_prototypes=400, dropout=0.1):
        super().__init__()

        self.query = nn.Linear(dim, num_prototypes, bias=False)
        self.memory = nn.Linear(num_prototypes, dim, bias=False)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        query = self.query(x)
        query = torch.softmax(query, dim=-1)

        reconstructed = self.memory(query)
        reconstructed = self.dropout(reconstructed)

        return reconstructed

class ProtoGCNBackbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stage=(5, 8),
                 down_stages=(5, 8),
                 num_prototypes=400,
                 num_filters=8,
                 init_std=0.02,
                 init_off=0.04):
        
        super().__init__()

        graph = NTUGraph(mode="random", num_filter=num_filters, init_std=init_std, init_off=init_off)

        A = graph.A

        self.num_node = 25
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_node)

        blocks = []
        current_channels = base_channels

        # Stage 1
        blocks.append(GCNBlock(in_channels, base_channels, A.clone(), stride=1, residual=False))

        inflate_times = 0

        # Stage 2 - 10
        for stage in range(2, num_stages + 1):
            in_c = current_channels

            if stage in inflate_stage:
                inflate_times += 1

            out_c = int(base_channels * (ch_ratio ** inflate_times))

            stride = (2 if stage in down_stages else 1)

            blocks.append(GCNBlock(in_c, out_c, A.clone(), stride=stride))

            current_channels = out_c

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = current_channels
        self.prn = PrototypeReconstructionNetwork(dim=self.out_channels, num_prototypes=num_prototypes)
        self.graph_post = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.graph_bn = nn.BatchNorm2d(self.out_channels)
        self.graph_relu = nn.ReLU(inplace=True)

    def forward(self, x, return_stages=False):
        """
        Input:
            x: [N, M, T, V, C]

        Output:
            feature:
                [N, M, 384, T', V]

            reconstructed_graph:
                [N, 625]
        """

        N, M, T, V, C = x.shape

        if V != 25:
            raise ValueError(f"NTU expects V=25, got {V}")

        # Data BN
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        # [N*M, V*C, T]
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)

        # GCN Format
        x = x.view(N, M, V, C, T).contiguous()
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        # [N*M, C, T, V]
        x = x.view(N*M, C, T, V)

        stage_features = []

        last_graph = None

        for stage_idx, block in enumerate(self.blocks, start=1):
            x, last_graph = block(x)

            # Feture multi-level relation module
            if stage_idx in (4, 7, 10):
                stage = x.view(N, M, *x.shape[1:])
                stage_features.append(stage)

        # Final skeleton feature
        x = x.view(N, M, *x.shape[1:])

        # [N, M, 384, T', V]
        C_out = x.shape[2]

        # Last Graph
        graph = last_graph.view(N, M, C_out, V, V)

        graph = graph.mean(dim=1)

        # [N, C, V*V]
        graph = graph.view(N, C_out, V * V)

        reconstructed_list = []

        for batch_idx in range(N):
            # [V*V, C]
            graph_sample = graph[batch_idx].transpose(0, 1)

            graph_sample = self.prn(graph_sample)

            # [C, V, V]
            graph_sample = (graph_sample.transpose(0, 1).contiguous().view(C_out, V, V))

            reconstructed_list.append(graph_sample)

        reconstructed_graph = torch.stack(reconstructed_list, dim=0)
        reconstructed_graph = self.graph_post(reconstructed_graph)
        reconstructed_graph = self.graph_bn(reconstructed_graph)
        reconstructed_graph = self.graph_relu(reconstructed_graph)

        # [N, 625]
        reconstructed_graph = (reconstructed_graph.mean(dim=1).view(N, V * V))

        if return_stages:
            return (x, reconstructed_graph, stage_features)

        return x, reconstructed_graph

class ProtoGCNHead(nn.Module):
    def __init__(self, in_channels=384, num_classes=60, dropout=0.0):
        super().__init__()

        self.dropout = (nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        self.fc = nn.Linear(in_channels, num_classes)

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x, return_features=False):
        """
        x : [N, M, C, T, V]
        """

        # Person-wise T, V pooling
        person_features = x.mean(dim=(-2, -1)) # [N, M, C]

        # Original ProtoGCN person pooling
        scene_feature = person_features.mean(dim=1) # [N, C]

        scene_feature = self.dropout(scene_feature)
        logits = self.fc(scene_feature)

        if return_features:
            return (logits, person_features, scene_feature)

        return logits

class ProtoGCNBaseline(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()

        self.backbone = ProtoGCNBackbone()
        self.head = ProtoGCNHead(in_channels=self.backbone.out_channels, num_classes=num_classes)

    def forward(self, x, return_features=False):
        if return_features:
            (feature, reconstructed_graph, stages) = self.backbone(x, return_stages=True)
            (logits, person_features, scene_feature) = self.head(feature, return_features=True)

            return {"logits": logits, "reconstructed_graph": reconstructed_graph, "person_features": person_features, "scene_feature": scene_feature, "stages": stages}

        feature, reconstructed_graph = (self.backbone(x))
        logits = self.head(feature)

        return logits, reconstructed_graph


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = ProtoGCNBaseline(num_classes=60).to(device)
    model.eval()

    # [N, M, T, V, C]
    x = torch.randn(
        2,      # batch
        2,      # persons
        100,    # frames
        25,     # joints
        3,      # xyz
        device=device
    )

    print("Input:", x.shape)

    with torch.no_grad():
        outputs = model(x, return_features=True)

    print("\n")
    print("=== Final Outputs ===")
    print("Logits              :", outputs["logits"].shape)
    print("Reconstructed graph :", outputs["reconstructed_graph"].shape)
    print("Person features     :", outputs["person_features"].shape)
    print("Scene feature       :", outputs["scene_feature"].shape)

    print("\n")
    print("=== Multi-Level Features ===")

    for i, stage in enumerate(outputs["stages"], start=1):
        print(f"Stage {i}:", stage.shape)

    print("\n")
    print("Forward test: PASS")