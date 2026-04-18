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


def train_CNN(model, train_loader, test_loader, device, model_path, model_name, n_epochs=100, lr=0.01, alpha=1.0, patience=10):
    
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. No training")
        return

    if alpha == 1.0: 
        # if alpha is 1, we use MSELoss for computation efficiency. Otherwise we would compare signs for nothing.
        criterion_train = nn.MSELoss()
    else:
        criterion_train = AsymmetricMSE(alpha)

    criterion_test = nn.MSELoss() # for evaluation we want to have comparable metrics.

    model_config = model.get_config()

    train_config = {
        "architecture": "CNN",
        "lr": lr,
        "n_epochs": n_epochs,
        "batch_size": train_loader.batch_size,
        "optimizer": "Adam",
        "alpha": alpha,
        "patience": patience
    }
    
    run = wandb.init(
        entity="DeepChess",
        name=model_name,
        project="DeepChess",
        config={**model_config, **train_config}
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # model.parameters() returns the weights and biases of the model

    best_test_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    for epoch in range(n_epochs):
        if early_stop:
            break
            
        model.train()
        train_loss = 0
        train_correct_sign = 0
        train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # forward
            outputs = model(x).squeeze(1)
            loss = criterion_train(outputs, y)

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
                loss = criterion_test(outputs, y)

                same_sign = (outputs * y) > 0 
                test_correct_sign += same_sign.sum().item()
                test_loss += loss.item()
                test_samples += y.size(0)

        avg_test_loss = test_loss / len(test_loader)

        metrics = {
            "train_loss": train_loss /len(train_loader),
            "test_loss": avg_test_loss,
            "train_sign_acc": train_correct_sign / train_samples,
            "test_sign_acc": test_correct_sign / test_samples,
            "epoch": epoch + 1
        }
        run.log(metrics)
        print(f"Epoch [{epoch+1}/{n_epochs}]: train_loss={metrics['train_loss']:.4f} test_loss={metrics['test_loss']:.4f} train_sign_acc={metrics['train_sign_acc']:.2f} test_sign_acc={metrics['test_sign_acc']:.2f}")

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), model_path) # save the best model
            print(f"New best model saved with test_loss={best_test_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                early_stop = True

    print("Finished Training.")
    run.finish()


""" grid search for hyperparameters. Déjà on fixe alpha à 1 pour le CNN classique. Ensuite on refera en tunant alpha.

    1. CNN MSE sym:

        - lr: [0.005, 0.001, 0.0005]
        - dropout: [0.2, 0.3, 0.4]
        - conv_filters: [ [20, 50] , [32, 64, 128]]
        - fc_layers: [[500], [512, 256]]

    2. CNN Asym:

        - alpha: [1.2, 1.5, 2]
        - lr: [0.01, 0.005, 0.001]
        - dropout: [0.2, 0.3, 0.4]
        - conv_filters: [ [20, 50] , [32, 64, 128]]
        - fc_layers: [[500], [512, 256]]


"""
if __name__== "__main__":

    parser = argparse.ArgumentParser(description="Train ChessCNN with custom hyperparameters")

    # Définition des arguments à tuner
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha for Asymmetric MSE")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for WandB and saving")
    
    # Pour les listes (filters et layers), on utilise nargs='+' pour accepter plusieurs nombres
    parser.add_argument("--conv_filters", type=int, nargs='+', required=True, help="List of conv filters")
    parser.add_argument("--fc_layers", type=int, nargs='+', required=True, help="List of hidden layers size")

    args = parser.parse_args()
    
    config = {
        "epochs": 100,
        "lr": args.lr,
        "alpha": args.alpha,
        "batch_size": 128,
        "parquet_path": "data/dataset_100000.parquet",
        "conv_filters": args.conv_filters,
        "conv_kernels": [5, 3], # on ne tune pas
        "fc_layers": args.fc_layers,
        "dropout": args.dropout,
        "activation": nn.ELU,
        "model_name": args.model_name, #used for saving the model and for wandb run name
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
        in_channels=18,
        conv_filters=config["conv_filters"],
        conv_kernels=config["conv_kernels"],
        fc_dim=config["fc_layers"],
        dropout=config["dropout"]
    )

    model.to(device)
    model_path = f"models/{config['model_name']}.pth"

    train_CNN(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_path=model_path,
        model_name=config["model_name"],
        n_epochs=config["epochs"],
        lr=config["lr"],
        alpha=config["alpha"]
    )
