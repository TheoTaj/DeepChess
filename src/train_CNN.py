import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse
import numpy as np
import random

from dataset import ChessDataset, fen_to_tensor
from CNN import ChessCNN
from AsymmetricMSE import AsymmetricMSE


# d'après gemeni il faut faire tout ca pour que tout soit reproducible.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_CNN(model, train_loader, test_loader, device, model_path, n_epochs=10, lr=0.01, alpha=1.0):
    
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. No training")
        return

    if alpha == 1.0: 
        # if alpha is 1, we use MSELoss for computation efficiency. Otherwise we would compare signs for nothing.
        criterion = nn.MSELoss()
    else:
        criterion = AsymmetricMSE(alpha)

    model_config = model.get_config()

    train_config = {
        "architecture": "CNN",
        "lr": lr,
        "n_epochs": n_epochs,
        "batch_size": train_loader.batch_size,
        "optimizer": "SGD",
        "alpha": alpha
    }
    
    run = wandb.init(
        entity="DeepChess",
        name=f"CNN_100K_10epochs_2",
        project="DeepChess",
        config={**model_config, **train_config}
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # model.parameters() returns the weights and biases of the model

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct_sign = 0
        train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # forward
            outputs = model(x).squeeze(1)
            loss = criterion(outputs, y)

            # backward
            optimizer.zero_grad() # gradients to zero
            loss.backward()
            optimizer.step()

            same_sign = (outputs * y) > 0 
            train_correct_sign += same_sign.sum().item()
            train_loss += loss.item()
            train_samples += y.size(0)

        model.eval()
        test_loss = 0
        test_correct_sign = 0
        test_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x).squeeze(1)
                loss = criterion(outputs, y)

                same_sign = (outputs * y) > 0 
                test_correct_sign += same_sign.sum().item()
                test_loss += loss.item()
                test_samples += y.size(0)

        metrics = {
            "train_loss": train_loss /len(train_loader),
            "test_loss": test_loss / len(test_loader),
            "train_sign_acc": train_correct_sign / train_samples,
            "test_sign_acc": test_correct_sign / test_samples,
            "epoch": epoch + 1
        }
        run.log(metrics)
        print(f"Epoch [{epoch+1}/{n_epochs}]: train_loss={metrics['train_loss']:.4f} test_loss={metrics['test_loss']:.4f} train_sign_acc={metrics['train_sign_acc']:.2f} test_sign_acc={metrics['test_sign_acc']:.2f}")

    print("Finished Training.")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    run.finish()


if __name__== "__main__":
    
    config = {
        "epochs": 10,
        "lr": 0.01,
        "alpha": 1,
        "batch_size": 128,
        "parquet_path": "data/dataset_100000.parquet",
        "conv_filters": [20, 50],
        "conv_kernels": [5, 3],
        "fc_layers": [500],
        "dropout": 0.3,
        "activation": nn.ELU,
        "model_name": "CNN_100k.pth"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_set = ChessDataset(parquet_path=config["parquet_path"], train=True)
    test_set = ChessDataset(parquet_path=config["parquet_path"], train=False)

    train_loader = DataLoader(
        train_set, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=2
    )

    model = ChessCNN(
        conv_filters=config["conv_filters"],
        conv_kernels=config["conv_kernels"],
        fc_dim=config["fc_layers"]
    )

    model.to(device)
    model_path = f"models/{config['model_name']}"

    train_CNN(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_path=model_path,
        n_epochs=config["epochs"],
        lr=config["lr"],
        alpha=config["alpha"]
    )
