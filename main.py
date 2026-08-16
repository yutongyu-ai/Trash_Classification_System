import os
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report

from hpo import build_objective
from train import train, val_test
from utils.visualization import plot_train_val, plot_confusion_matrix
from utils.datasets import get_trashnet_train, get_trashnet_test, get_class_weights
from backend.models.resnet18 import get_model

DEVICE = "cuda"       # falls back to CPU automatically if unavailable
DATA_ROOT = "./data"
NUM_WORKERS = 4


def get_predictions(model, loader, device):
    """
    Collect all predictions and labels from a dataloader
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def main():

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Initial loaders to trigger the split and get class weights;
    # batch_size gets overridden below once HPO picks one.
    train_loader, val_loader = get_trashnet_train(
        data_root=DATA_ROOT, batch_size=32, num_workers=NUM_WORKERS
    )
    test_loader = get_trashnet_test(
        data_root=DATA_ROOT, batch_size=32, num_workers=NUM_WORKERS
    )
    num_classes = len(train_loader.dataset.classes)
    class_weights = get_class_weights(train_loader.dataset)

    # ---- Hyperparameter search ----
    # Short trials just rank configs; T_max uses full_num_epochs (not
    # hpo_epochs) so a trial's LR schedule mirrors the start of the eventual
    # full run instead of its own fully-annealed short one (see hpo.py).
    n_hpo_trials = 25
    hpo_epochs = 6
    full_num_epochs = 30
    mlflow.set_experiment("trashnet-hpo")
    objective = build_objective(
        num_classes=num_classes,
        class_weights=class_weights,
        num_epochs=hpo_epochs,
        full_num_epochs=full_num_epochs,
        device=device,
        data_root=DATA_ROOT,
        num_workers=NUM_WORKERS,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_hpo_trials)

    print("Best HPO trial:")
    print(f"  val_acc: {study.best_trial.value:.4f}")
    print(f"  params: {study.best_trial.params}")

    best_params = study.best_trial.params

    # ---- Final training with the best hyperparameters found above ----
    train_loader, val_loader = get_trashnet_train(
        data_root=DATA_ROOT, batch_size=best_params["batch_size"], num_workers=NUM_WORKERS
    )
    test_loader = get_trashnet_test(
        data_root=DATA_ROOT, batch_size=best_params["batch_size"], num_workers=NUM_WORKERS
    )

    model = get_model(num_classes=num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"]
    )
    num_epochs = full_num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5
    )

    mlflow.set_experiment("trashnet-final-training")
    with mlflow.start_run(run_name="final-training"):
        mlflow.log_params(best_params)
        mlflow.log_param("num_epochs", num_epochs)

        # Train and validate the model
        best_val_acc = -1
        log_file = "./outputs/trashnet_training_log.csv"
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

            for epoch in range(num_epochs):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = val_test(model, val_loader, criterion, device)

                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
                f.flush()

                print(f"Epoch [{epoch + 1}/{num_epochs}] "
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

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), "checkpoints/best_resnet18_trashnet.pth")

                scheduler.step()

        # Test the model
        model.load_state_dict(
            torch.load("checkpoints/best_resnet18_trashnet.pth",
                       map_location=device,
                       weights_only=True)
        )
        test_loss, test_acc = val_test(model, test_loader, criterion, device)

        # Generate plots
        # Translate raw folder name to the app's display name (see ui.py/inference.py).
        DISPLAY_NAMES = {"trash": "General Waste"}
        classes = [DISPLAY_NAMES.get(c, c) for c in train_loader.dataset.classes]
        y_true, y_pred = get_predictions(model, test_loader, device)
        cm_save_path = "./outputs/trashnet_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, classes, cm_save_path)
        loss_path = "./outputs/trashnet_loss.png"
        acc_path = "./outputs/trashnet_acc.png"
        plot_train_val(load_path=log_file, loss_path=loss_path, acc_path=acc_path)

        print(f"\n Test Acc: {test_acc:.4f}")

        report = classification_report(
            y_true, y_pred, target_names=classes, output_dict=True
        )

        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_artifact(cm_save_path)
        mlflow.log_artifact(loss_path)
        mlflow.log_artifact(acc_path)
        mlflow.log_artifact("checkpoints/best_resnet18_trashnet.pth")
        # MLflow 3.x's default 'pt2' serialization needs tracing/signature
        # gymnastics; serialization_format="pickle" avoids that.
        input_example = torch.randn(1, 3, 224, 224, device=device)
        mlflow.pytorch.log_model(
            model, name="model", input_example=input_example, serialization_format="pickle"
        )


if __name__ == '__main__':
    main()