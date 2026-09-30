import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassSpecificContrastiveLoss(nn.Module):
    def __init__(self, num_classes=60, in_channels=625, hidden_channels=256, temperature=0.125, momentum=0.9, pred_threshold=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = momentum
        self.pred_threshold = pred_threshold

        self.avg_f = torch.randn(hidden_channels, num_classes)
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def one_hot(self, label):
        label = label.view(-1)
        
        one_hot = torch.eye(self.num_classes, device=label.device)
        one_hot = one_hot.index_select(0, label.long())

        return one_hot.float()

    def get_mask(self, label_one, pred_one, probability):
        mask = label_one * pred_one
        mask = mask * (probability > self.pred_threshold).float()

        return mask

    def local_average(self, feature, mask):
        feature = feature.permute(1, 0)

        avg_f = self.avg_f.detach().to(feature.device)
        mask_sum = mask.sum(dim=0, keepdim=True)

        feature_mask = torch.matmul(feature, mask)
        feature_mask = feature_mask / (mask_sum + 1e-12)

        has_object = (mask_sum > 1e-8).float()

        has_object[has_object > 0.1] = self.momentum
        has_object[has_object <= 0.1] = 1.0

        feature_memory = avg_f * has_object + (1.0 - has_object) * feature_mask

        with torch.no_grad():
            self.avg_f = feature_memory

        return feature_memory

    def get_score(self, feature, feature_memory):
        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)
        

        feature_memory = feature_memory.permute(1, 0)
        feature_memory = feature_memory / (torch.norm(feature_memory, p=2, dim=1, keepdim=True) + 1e-12)

        score = torch.matmul(feature_memory, feature.permute(1, 0))
        score = score / self.temperature

        return score

    def forward(self, feature, label, logits):
        feature = self.fc(feature)

        pred = logits.max(dim=1)[1]
        pred_one = self.one_hot(pred)
        label_one = self.one_hot(label)

        probability = torch.softmax(logits, dim=1)
        mask = self.get_mask(label_one, pred_one, probability)

        feature_memory = self.local_average(feature, mask)

        score = self.get_score(feature, feature_memory)
        score = score.permute(1, 0).contiguous()

        return self.loss(score, label).mean()

    
class ProtoGCNLoss(nn.Module):
    def __init__(self, num_classes=60, csc_weight=0.2):
        super().__init__()

        self.csc_weight = csc_weight
        self.csc_loss = ClassSpecificContrastiveLoss(num_classes=num_classes)

    def forward(self, logits, reconstructed_graph, labels):
        loss_ce = F.cross_entropy(logits, labels)

        loss_csc = self.csc_loss(reconstructed_graph, labels.detach(), logits.detach())
        loss = loss_ce + self.csc_weight * loss_csc

        return {"loss": loss, "loss_ce": loss_ce, "loss_csc": loss_csc}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device);

    criterion = ProtoGCNLoss(num_classes=60).to(device)
    logits = torch.randn(4, 60, device=device, requires_grad=True)
    reconstructed_graph = torch.randn(4, 625, device=device, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3], device=device)

    losses = criterion(logits, reconstructed_graph, labels)

    print("=== Loss ===")
    print("Total :", losses["loss"].item())
    print("CE    :", losses["loss_ce"].item())
    print("CSC   :", losses["loss_csc"].item())

    losses["loss"].backward()

    print("=== Gradient ===")
    print("Logits grad :", logits.grad is not None)
    print("Graph grad  :", reconstructed_graph.grad is not None)

    print("Backward test: PASS")