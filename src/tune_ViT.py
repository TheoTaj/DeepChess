import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import os
import random
import numpy as np
import time

from dataset import ChessDataset
from ViT import ChessViT
from train_ViT import train_ViT

import optuna.storages

storage = "sqlite:///vit_tuning2.db"
study_name = "chess_vit_tuning2"

FIXED = {
    "epochs": 150,
    "warmup_epochs": 5,
    "alpha": 1.0,
    "batch_size": 512,
    "patience": 15,
    "scheduler_factor": 0.5,
    "scheduler_patience": 6,
    "dataset_path": "data/kaggle_100k_300.parquet",
}

def objective(trial):

    seed = 42 + trial.number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ranges for phase 1
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # dropout = trial.suggest_float("dropout", 0.0, 0.2)
    # n_blocks = trial.suggest_categorical("n_blocks", [2, 4, 6])
    # pair = trial.suggest_categorical("embed_n_heads_pair", [
    #     "64,2", "64,4", "128,4", "128,8", "256,4", "256,8"
    # ])
    # embed_dim, n_heads = [int(x) for x in pair.split(",")]
    # mlp_dim = trial.suggest_categorical("mlp_dim", [128, 256, 512])

    # Ranges for phase 2
    # lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    # dropout = trial.suggest_float("dropout", 0.0, 0.15)
    # n_blocks = trial.suggest_categorical("n_blocks", [2, 4])
    # pair = trial.suggest_categorical("embed_n_heads_pair", ["128,8", "256,8"])
    # embed_dim, n_heads = [int(x) for x in pair.split(",")]
    # mlp_dim = trial.suggest_categorical("mlp_dim", [128, 256, 512])

    # Ranges for phase 3
    # lr = trial.suggest_float("lr", 8e-4, 4e-3, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    # dropout = trial.suggest_float("dropout", 0.0, 0.12)
    # n_blocks = trial.suggest_categorical("n_blocks", [2, 4])
    # mlp_dim = trial.suggest_categorical("mlp_dim", [128, 256])
    # embed_dim = 128
    # n_heads = 8

    # Small tuning
    lr = 1e-4
    weight_decay = 0.000948
    dropout = 0.065
    n_blocks = trial.suggest_categorical("n_blocks", [2, 4, 6, 8])
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 384])
    mlp_dim = 2 * embed_dim
    n_heads = 8


    model_name = f"ViT_TUNE2_{trial.number}"

    # --- Build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessViT(
        embed_dim=embed_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
    ).to(device)

    # --- Build dataloaders ---
    train_set = ChessDataset(parquet_path=FIXED["dataset_path"], train=True)
    test_set = ChessDataset(parquet_path=FIXED["dataset_path"], train=False)
    train_loader = DataLoader(train_set, batch_size=FIXED["batch_size"], shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=FIXED["batch_size"], shuffle=False, num_workers=8)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pth"

    best_loss = train_ViT(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_path=model_path,
        model_name=model_name,
        n_epochs=FIXED["epochs"],
        lr=lr,
        alpha=FIXED["alpha"],
        patience=FIXED["patience"],
        scheduler_factor=FIXED["scheduler_factor"],
        scheduler_patience=FIXED["scheduler_patience"],
        weight_decay=weight_decay,
        warmup_epochs=FIXED["warmup_epochs"],
    )

    return best_loss


if __name__ == "__main__":
    time.sleep(random.uniform(0, 5))  
    
    try:
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

    study.optimize(
        objective,
        n_trials=8,        # total number of trials you want
        n_jobs=1,           # keep at 1 — parallelism handled at cluster level
    )

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")