import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse
import numpy as np
import random
import os

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


def train_CNN(model, train_loader, test_loader, device, model_path, model_name, n_epochs=100, lr=0.01, alpha=1.0, patience=10, scheduler_factor=1, scheduler_patience=5):
    
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
        "patience": patience,
        "scheduler": "ReduceLROnPlateau" if scheduler_factor < 1 else "None",
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
    }
    
    run = wandb.init(
        entity="DeepChess",
        name=model_name,
        project="DeepChess",
        config={**model_config, **train_config}
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # model.parameters() returns the weights and biases of the model

    if scheduler_factor < 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
    else:
        scheduler = None

    if os.path.exists(model_path):
        print(f"Loading model from {model_path} !")
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_test_loss = checkpoint['best_test_loss']
        print(f"Continuing training at epoch {start_epoch}")

    else:
        start_epoch = 0
        best_test_loss = float('inf')

    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(start_epoch, n_epochs):
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

        if scheduler is not None:
            scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        metrics = {
            "train_loss": train_loss /len(train_loader),
            "test_loss": avg_test_loss,
            "train_sign_acc": train_correct_sign / train_samples,
            "test_sign_acc": test_correct_sign / test_samples,
            "learning_rate": current_lr,
            "epoch": epoch + 1
        }
        run.log(metrics)
        print(f"Epoch [{epoch+1}/{n_epochs}]: train_loss={metrics['train_loss']:.4f} test_loss={metrics['test_loss']:.4f} train_sign_acc={metrics['train_sign_acc']:.2f} test_sign_acc={metrics['test_sign_acc']:.2f}")

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_test_loss': best_test_loss,
            }

            torch.save(checkpoint, model_path) # save the best model
            print(f"New best model saved with test_loss={best_test_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                early_stop = True

    print("Finished Training.")
    run.finish()

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

    parser.add_argument("--epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the parquet dataset")

    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor by which LR is reduced (default: 0.5)")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Epochs without improvement before LR reduction (default: 5)")

    args = parser.parse_args()
    
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "parquet_path": args.dataset_path,
        "conv_filters": args.conv_filters,
        "conv_kernels": [5, 3], # on ne tune pas
        "fc_layers": args.fc_layers,
        "dropout": args.dropout,
        "activation": nn.ELU,
        "model_name": args.model_name, #used for saving the model and for wandb run name
        "patience": args.patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
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
    os.makedirs("models", exist_ok=True)
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
        alpha=config["alpha"],
        patience=config["patience"],
        scheduler_factor=config["scheduler_factor"],
        scheduler_patience=config["scheduler_patience"],
    )
