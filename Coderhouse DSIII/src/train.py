"""Loop de entrenamiento y evaluación en un conjunto de validación separado."""

import torch
import torch.nn as nn

from src.dataset import make_loaders
from src.device import get_device
from src.model import MLPClassifier

LEARNING_RATE = 1e-3
EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_DIM = 16
SEED = 42


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Calcula la pérdida promedio y el accuracy sobre el loader recibido."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, correct / total_examples


def run() -> None:
    torch.manual_seed(SEED)
    device = get_device()
    train_loader, val_loader, input_dim, num_classes = make_loaders(
        batch_size=BATCH_SIZE, seed=SEED
    )

    model = MLPClassifier(input_dim, HIDDEN_DIM, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"PyTorch {torch.__version__}")
    print(f"Dispositivo: {device}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"Arquitectura: MLP {input_dim} -> {HIDDEN_DIM} -> {num_classes}")
    print("-" * 72)
    print(f"{'epoca':>6} {'train_loss':>12} {'val_loss':>12} {'val_accuracy':>14}")

    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history.append((epoch, train_loss, val_loss, val_accuracy))
        print(f"{epoch:6d} {train_loss:12.4f} {val_loss:12.4f} {val_accuracy:14.4f}")

    first = history[0]
    last = history[-1]
    print("-" * 72)
    print(
        "Resumen: "
        f"train_loss {first[1]:.4f} -> {last[1]:.4f}; "
        f"val_loss {first[2]:.4f} -> {last[2]:.4f}; "
        f"val_accuracy {first[3]:.4f} -> {last[3]:.4f}"
    )


if __name__ == "__main__":
    run()
