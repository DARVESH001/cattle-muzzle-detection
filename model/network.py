import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class EmbeddingNetwork(nn.Module):
    """Extracts a compact embedding vector from a cattle muzzle image."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Custom embedding head
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        # L2 normalize so cosine similarity = dot product
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNetwork(nn.Module):
    """Siamese network with shared weights for comparing two images."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_net = EmbeddingNetwork(embedding_dim)

    def forward(self, img1, img2):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        return emb1, emb2

    def get_embedding(self, img):
        return self.embedding_net(img)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks.

    label=1 means same cattle (positive pair), label=0 means different (negative pair).
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distance = F.pairwise_distance(emb1, emb2)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()
