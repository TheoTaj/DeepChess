import torch
import torch.nn as nn

class AsymmetricMSE(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        different_sign = (pred * target) < 0
        loss = torch.where(different_sign, self.alpha * mse, mse)
        return loss.mean()