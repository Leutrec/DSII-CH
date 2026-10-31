"""Arquitectura del clasificador."""

import torch.nn as nn


class MLPClassifier(nn.Module):
    """Perceptrón multicapa para clasificación multiclase.

    Devuelve logits (sin softmax). La pérdida CrossEntropyLoss aplica
    log-softmax de forma numéricamente estable.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.network(features)
