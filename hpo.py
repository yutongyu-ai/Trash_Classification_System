import mlflow
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from train import train, val_test
from utils.datasets import get_trashnet_train
from utils.seed import set_seed
from backend.models.resnet18 import get_model


def build_objective(
    num_classes,
    class_weights,
    num_epochs=6,
    full_num_epochs=30,
    device=None,
    data_root="data",
    num_workers=2,
    seed=42,
):
    """Build an Optuna objective for fine-tuning ResNet18 on TrashNet.

    T_max=full_num_epochs (not the trial's shortened num_epochs) so a
    trial's LR schedule mirrors the real run instead of its own
    fully-annealed short one.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Same seed every trial to isolate hyperparameter effects from RNG noise.
        set_seed(seed)

        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params({
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "trial_number": trial.number,
            })

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

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )

                trial.report(val_acc, epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", True)
                    raise optuna.exceptions.TrialPruned()

                best_val_acc = max(best_val_acc, val_acc)
                scheduler.step()

            mlflow.log_metric("best_val_acc", best_val_acc)

        return best_val_acc

    return objective
