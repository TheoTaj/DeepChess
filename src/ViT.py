import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset

class PatchEmbedding(nn.Module):
    def __init__(self, 
                board_size=8,   # size of the chess board
                patch_size=1,   # since a chess board is rather small, each patch will represent one square on the board
                in_channels=18, # number of channels we use to represent a position
                embed_dim=128   # size of the embedding space we will project our representation of the chess board in
                ):
        super().__init__() 
        self.n_patches = (board_size // patch_size)**2
        self.proj = nn.Linear(in_channels, embed_dim) # Project the channels into the embedding space with a linear layer

    def forward(self, x):
        # The size of the input x is [B, C, H, W] (B=batch_size, C=in_channels, H=W=board_size)
        x = x.reshape((x.shape[0], self.n_patches, -1)) # Reshape x to [B, HxW, C] before passing to linear layer
        x = self.proj(x) # [B, HxW, embed_size]
        return x # Now, for each batch B, we have [64, embed_size] tensors
    
class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim=128, 
                 n_heads=4,     # number of heads for the MultiheadAttention layer 
                 mlp_dim=256,   # output dimension of the MLP
                 dropout=0.1
                ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        normed = self.norm1(x)
        att_output, _ = self.attention(normed, normed, normed)
        x = att_output + x
        x = self.mlp(self.norm2(x)) + x
        return x
    
class ChessViT(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 n_blocks=4,
                 board_size=8,
                 patch_size=1,
                 in_channels=18,
                 n_heads=4,
                 mlp_dim=256,
                 dropout=0.1
                 ):
        super().__init__()

        self.patch_emb = PatchEmbedding(
            board_size=board_size, 
            patch_size=patch_size, 
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.patch_emb.n_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim=embed_dim, 
                n_heads=n_heads, 
                mlp_dim=mlp_dim, 
                dropout=dropout
            ) for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_emb(x)

        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        
        x = self.blocks(x)
        x = self.norm(x[:, 0]) 
  
        return torch.tanh(self.head(x))

if __name__=="__main__":
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = ChessViT().to(device)
    print(f'ChessViT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')