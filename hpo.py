import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from train import train, val_test
from utils.datasets import get_trashnet_train
from backend.models.resnet18 import get_model


def objective(trial):
    # Load base config
    num_epochs = 5

    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model()
    criterion = nn.CrossEntropyLoss()

    best_loss = 0.0
    train_loader, val_loader = get_trashnet_train(batch_size=batch_size)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=5,
        eta_min=1e-5
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_test(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_loss:
            best_loss = val_loss

        scheduler.step()

    return best_loss



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best Trial:")
    print(study.best_trial.params)