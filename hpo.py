import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from train import train, val_test
from utils.datasets import get_trashnet_train
from backend.models.resnet18 import get_model


def build_objective(
    num_classes,
    class_weights,
    num_epochs=6,
    full_num_epochs=30,
    device=None,
    data_root="data",
    num_workers=2,
):
    """
    Build an Optuna objective for fine-tuning ResNet18 on TrashNet.

    Search space (kept to training hyperparameters only, not model
    architecture, so every trial produces a checkpoint with the same shape
    that backend/inference.py already expects to load):
      - lr: log-uniform 1e-4 .. 3e-3
        Typical AdamW fine-tuning range for a pretrained CNN backbone;
        the existing 30-epoch run used 3e-4 and got ~92-95% val acc, so
        this brackets that with room on both sides.
      - weight_decay: log-uniform 1e-5 .. 1e-2
        Wide range since TrashNet is small (~2.5k images) and prone to
        overfitting (train acc hit ~99.5% vs val acc ~92-94% previously).
      - batch_size: {16, 32, 64}
        Dataset is small, so batch size mainly trades off gradient noise
        vs. steps per epoch rather than throughput.

    Uses AdamW + CosineAnnealingLR to match the final training recipe in
    main.py, so a trial's ranking is representative of how it will actually
    perform once trained for the full run. `T_max` is set to
    `full_num_epochs` (the final run's epoch count), not `num_epochs` (the
    trial's shortened epoch count) — a trial only runs the first
    `num_epochs` steps of that schedule, so its LR trajectory matches the
    *start* of the eventual full run instead of being its own fully
    annealed 6-epoch schedule. Without this, a trial's "best" lr reflects
    what works when annealed to near-zero in 6 steps, not what works
    during the first 6 steps of a slower 30-step anneal.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = get_model(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader, val_loader = get_trashnet_train(
            data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=full_num_epochs, eta_min=1e-5
        )

        best_val_acc = 0.0
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_test(model, val_loader, criterion, device)

            print(f"[trial {trial.number}] Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
                  f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_val_acc = max(best_val_acc, val_acc)
            scheduler.step()

        return best_val_acc

    return objective


if __name__ == "__main__":
    from utils.datasets import get_class_weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_trashnet_train()
    num_classes = len(train_loader.dataset.classes)
    class_weights = get_class_weights(train_loader.dataset)

    objective = build_objective(num_classes, class_weights, num_epochs=6, device=device)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    print("Best Trial:")
    print(f"  val_acc: {study.best_trial.value:.4f}")
    print(f"  params: {study.best_trial.params}")
