import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse

import dataset
from ViT import ChessViT

def train(model, train_loader, test_loader, device, model_path="model/model.pth", n_epochs=10, lr=1e-3):
    run = wandb.init(
        entity="adelandsheere-university-of-li-ge",
        project="DeepChess",
        config={
            "architecture":"ViT", 
            "n_epochs": n_epochs, 
            "lr": lr, 
            "batch_size": train_loader.batch_size
        }
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).squeeze(1), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss += criterion(model(x).squeeze(1), y).item()

        run.log({
            "train_loss": train_loss / len(train_loader),
            "test_loss": test_loss / len(test_loader),
            "epoch": epoch + 1
        })
        print(f"Epoch [{epoch+1}/{n_epochs}]: train_loss={train_loss/len(train_loader)} test_loss={test_loss/len(test_loader)}")

    print("Finished Training.")
    torch.save(model.state_dict(), model_path)
    run.finish()

if __name__=="__main__":
    model_folder = "models/"
    data_folder = "data/"

    # By doing this we can pass arguments directly in the command line on Alan like this for example:
    # python run.py --n_epochs 10 --lr 0.001 --data small --save mymodel
    parser = argparse.ArgumentParser(description="Train ChessViT")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="small") # 'data' should be 'small' (100000) or 'large' (5000000)
    parser.add_argument("--save", type=str, default=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth") # 'save' is the name of the saved model (should be .pth)
    args = parser.parse_args()

    # Validate --data
    valid_data_options = {"small", "large"}
    if args.data not in valid_data_options:
        parser.error(f"--data must be one of {valid_data_options}, got '{args.data}'")
    if args.data == "small":
        data_path = data_folder + "dataset_100000.parquet"
    elif args.data == "large":
        data_path = data_folder + "dataset_5000000.parquet"

    # Validate --save
    if not args.save.endswith(".pth"):
        parser.error("--save must end with '.pth'")
    if "/" in args.save or "\\" in args.save:
        parser.error("--save must be a filename only, not a path")

    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_data = dataset.ChessDataset(data_path, train=True)
    test_data = dataset.ChessDataset(data_path, train=False)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2)

    model = ChessViT().to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    train(model, train_loader, test_loader, device, 
          model_path=model_folder+args.save,
          n_epochs=args.n_epochs, 
          lr=args.lr)