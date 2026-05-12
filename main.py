import torch
import optuna
import torch.nn as nn
import torch.optim as optim
import csv
import yaml
import numpy as np

# from hpo import objective
from train import train, val_test
from utils.visualization import plot_train_val, plot_confusion_matrix
from utils.datasets import get_trashnet_train, get_trashnet_test
from backend.models.resnet18 import get_model

def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    cfg = load_config()
    device = torch.device(
        cfg['train']['device'] if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = get_trashnet_train(batch_size=cfg['train']["batch_size"])
    test_loader = get_trashnet_test(batch_size=cfg['train']["batch_size"])
    num_classes = len(train_loader.dataset.classes)

    model = get_model(num_classes=num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['optimizer']["lr"]),
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=1e-5
    )

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=20)
    #
    # print("Best Trial:")
    # print(study.best_trial.params)

    # Train and validate the model
    num_epochs = 30
    best_val_acc = -1
    log_file = "./outputs/trashnet_training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_test(model, val_loader, criterion, device)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
    classes = [str(i) for i in range(10)]
    y_true, y_pred = get_predictions(model, test_loader, device)
    cm_save_path = "./outputs/trashnet_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, classes, cm_save_path)
    loss_path = "./outputs/trashnet_loss.png"
    acc_path = "./outputs/trashnet_acc.png"
    plot_train_val(load_path=log_file, loss_path=loss_path, acc_path=acc_path)

    print(f"\n Test Acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()