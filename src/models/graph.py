import torch
import numpy as np


class NTUGraph:
    def __init__(self, mode="random", num_filter=8, init_std=0.02, init_off=0.04):
        self.num_node = 25

        neighbor_base = [
            (1, 2), (2, 21), (3, 21), (4, 3),
            (5, 21), (6, 5), (7, 6), (8, 7),
            (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15),
            (17, 1), (18, 17), (19, 18), (20, 19),
            (22, 8), (23, 8), (24, 12), (25, 12)
        ]

        self.inward = [(i - 1, j - 1) for i, j in neighbor_base]
        self.outward = [(j, i) for i, j in self.inward]
        self.self_link = [(i, i) for i in range(self.num_node)]

        if mode == "random":
            A = np.random.randn(num_filter, self.num_node, self.num_node)
            A = A * init_std + init_off

        elif mode == "spatial":
            A = self._spatial_graph()

        else:
            raise ValueError(f"Unsupported graph mode: {mode}")

        self.A = torch.tensor(A, dtype=torch.float32)

    def _edge2mat(self, edges):
        A = np.zeros((self.num_node, self.num_node), dtype=np.float32)

        for i, j in edges: A[j, i] = 1.0

        return A

    @staticmethod
    def _normalize_digraph(A):
        degree = A.sum(axis=0)

        D = np.zeros_like(A)

        for i in range(A.shape[0]):
            if degree[i] > 0:
                D[i, i] = degree[i] ** -1

        return A @ D

    def _spatial_graph(self):
        I = self._edge2mat(self.self_link)

        inward = self._normalize_digraph(self._edge2mat(self.inward))

        outward = self._normalize_digraph(self._edge2mat(self.outward))

        return np.stack([I, inward, outward], axis=0)


if __name__ == "__main__":
    print("=== Random Graph ===")

    graph = NTUGraph(
        mode="random",
        num_filter=8,
        init_std=0.02,
        init_off=0.04
    )

    A = graph.A

    print("A shape :", A.shape)
    print("A dtype :", A.dtype)
    print("A mean  :", A.mean().item())
    print("A std   :", A.std().item())

    assert A.shape == (8, 25, 25)

    print("\n=== Spatial Graph ===")

    graph_spatial = NTUGraph(mode="spatial")
    A_spatial = graph_spatial.A

    print("A shape :", A_spatial.shape)
    print("Self    :", A_spatial[0].shape)
    print("Inward  :", A_spatial[1].shape)
    print("Outward :", A_spatial[2].shape)

    print("\n=== Edge Info ===")

    print("Num joints    :", graph_spatial.num_node)
    print("Self links    :", len(graph_spatial.self_link))
    print("Inward edges  :", len(graph_spatial.inward))
    print("Outward edges :", len(graph_spatial.outward))

    print("\nFirst 5 inward edges:")
    print(graph_spatial.inward[:5])

    print("First 5 outward edges:")
    print(graph_spatial.outward[:5])

    assert A_spatial.shape == (3, 25, 25)
    assert graph_spatial.outward == [(j, i) for i, j in graph_spatial.inward]