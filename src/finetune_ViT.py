import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import argparse
import numpy as np
import random
import os

from dataset import ChessDataset
from ViT import ChessViT

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def finetune_ViT(model, train_loader, test_loader, device, pretrained_path, output_path, model_name,
                 n_epochs=50, lr=1e-4, patience=10, scheduler_factor=0.5, scheduler_patience=5,
                 weight_decay=1e-4, warmup_epochs=3, unfreeze_epoch=5, wandb_run_id=None):

    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. No training")
        return

    # Load pretrained checkpoint
    print(f"Loading pretrained model from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Pretrained weights loaded.")

    # # Reinitialize head
    # model.reset_head()
    # print("Head reinitialized.")

    # # Freeze backbone, only train head
    # for name, param in model.named_parameters():
    #     if 'head' not in name:
    #         param.requires_grad = False
    # print("Backbone frozen. Training head only.")

    criterion = nn.MSELoss()

    model_config = model.get_config()
    train_config = {
        "architecture": "ViT",
        "mode": "finetune",
        "lr": lr,
        "n_epochs": n_epochs,
        "warmup_epochs": warmup_epochs,
        "unfreeze_epoch": unfreeze_epoch,
        "batch_size": train_loader.batch_size,
        "optimizer": "AdamW",
        "patience": patience,
        "scheduler": "ReduceLROnPlateau" if scheduler_factor < 1 else "None",
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "weight_decay": weight_decay,
        "pretrained_path": pretrained_path,
    }

    run = wandb.init(
        entity="DeepChess",
        name=model_name,
        project="DeepChess",
        config={**model_config, **train_config},
        id=wandb_run_id,
        resume="allow"
    )

    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=lr,
    #     weight_decay=weight_decay
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-7,
    ) if scheduler_factor < 1 else None

    best_test_loss = float('inf')
    epochs_no_improve = 0
    backbone_unfrozen = False

    for epoch in range(n_epochs):

        # Gradual unfreezing: unfreeze backbone at unfreeze_epoch with lower lr
        # if not backbone_unfrozen and epoch == unfreeze_epoch:
        #     print(f"Epoch {epoch+1}: Unfreezing backbone with lr={lr * 0.1:.2e}")
        #     for param in model.parameters():
        #         param.requires_grad = True
        #     optimizer.add_param_group({
        #         'params': [p for name, p in model.named_parameters() if 'head' not in name],
        #         'lr': lr * 0.1,
        #         'weight_decay': weight_decay
        #     })
        #     backbone_unfrozen = True

        model.train()
        train_loss = 0
        train_correct_sign = 0
        train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x).squeeze(1)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
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

        avg_test_loss = test_loss / len(test_loader)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "test_loss": avg_test_loss,
            "train_sign_acc": train_correct_sign / train_samples,
            "test_sign_acc": test_correct_sign / test_samples,
            "learning_rate": current_lr,
            "epoch": epoch + 1,
            "backbone_unfrozen": backbone_unfrozen,
        }
        run.log(metrics)
        print(f"Epoch [{epoch+1}/{n_epochs}]: train_loss={metrics['train_loss']:.4f} test_loss={metrics['test_loss']:.4f} "
              f"train_sign_acc={metrics['train_sign_acc']:.2f} test_sign_acc={metrics['test_sign_acc']:.2f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'best_test_loss': best_test_loss,
            }
            torch.save(checkpoint, output_path)
            print(f"New best model saved with test_loss={best_test_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Finished fine-tuning.")
    run.finish()
    return best_test_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ChessViT on a balanced dataset")

    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to the pretrained .pth checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Name for WandB run and output file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the balanced parquet dataset")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the head (backbone uses lr*0.1)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--unfreeze_epoch", type=int, default=5, help="Epoch at which to unfreeze the backbone")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="LR reduction factor")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Scheduler patience")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume")

    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--mlp_dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on: {device}")

    train_set = ChessDataset(parquet_path=args.dataset_path, K=300.0, train=True)
    test_set = ChessDataset(parquet_path=args.dataset_path, K=300.0, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ChessViT(
        embed_dim=args.embed_dim,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout
    ).to(device)

    os.makedirs("models", exist_ok=True)
    output_path = f"models/{args.model_name}.pth"

    finetune_ViT(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        pretrained_path=args.pretrained_path,
        output_path=output_path,
        model_name=args.model_name,
        n_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        unfreeze_epoch=args.unfreeze_epoch,
        wandb_run_id=args.wandb_run_id,
    )