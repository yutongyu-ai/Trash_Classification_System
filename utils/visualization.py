import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_predictions(dataset, y_pred, num_images=4, save_path=None):
    """
    Visualize predicted results
    """
    plt.figure(figsize=(6, 6))
    for i in range(min(num_images, len(dataset))):
        img, label = dataset[i]
        pred = y_pred[i]
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(f"GT: {label}, Pred: {pred}", color="green" if label == pred else "red")
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plot and save confusion matrix heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_train_val(load_path,loss_path, acc_path):
    df = pd.read_csv(load_path)

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Plot accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()