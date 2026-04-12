import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb

import dataset
from ViT import ChessViT

def train(model, train_loader, test_loader, device, model_path=None, n_epochs=10, lr=1e-3):
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
    
    if model_path is None:
        model_path = f"models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

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
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_data = dataset.ChessDataset("data/dataset_100000.parquet", train=True)
    test_data = dataset.ChessDataset("data/dataset_100000.parquet", train=False)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2)

    model = ChessViT().to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    train(model, train_loader, test_loader, device)