import torch
import numpy as np
import pandas as pd
import chess
from torch.utils.data import Dataset

def fen_to_tensor(fen):
    """
    Converts a FEN string to a 18x8x8 tensor representation of the chess board.
    Plans 0-5: P, N, B, R, Q, K (Blancs)
    Plans 6-11: p, n, b, r, q, k (Noirs)

    eg: this fen "rnbqkbnr/pppppppp/8/8/8/8/8/8 w KQkq - 0 1" 
    results in a tensor where the first 6 planes are all zeros (no white pieces) and the last 6 planes have a 1 in the positions corresponding to the black pieces.
    """
    board = chess.Board(fen)
    tensor = np.zeros((18, 8, 8), dtype=np.uint8)
    
    # First 12 planes for pieces, where 0-5 are for white pieces and 6-11 for black pieces.
    piece_map = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
    for square, piece in board.piece_map().items():
        tensor[piece_map[piece.symbol()], chess.square_rank(square), chess.square_file(square)] = 1
        
    # this plane is full of ones if it's white to move.
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
        
    # Same logic for castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1
        
    # Same for en passant square, here we only set one case to 1 and it's the one where the capture is possible.
    if board.ep_square:
        tensor[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
        
    return tensor

class ChessDataset(Dataset):
    def __init__(self, parquet_path, K=300.0, 
                 train=True, # specify train or test split 
                 train_ratio=0.9, # 90% training samples, 10% testing samples
                 seed=42): # ensures the shuffle is always the same
        
        print(f"Loading dataset from {parquet_path}...")
        print(f"Normalizing centipawns with K={K} (y = tanh(cp/K))")
        df = pd.read_parquet(parquet_path)

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffle the dataset
        split_idx = int(len(df) * train_ratio) # define where to split the data 
        
        if train:
            self.df = df.iloc[:split_idx]
            print(f"Training Dataset loaded with {len(self.df)} positions.")
        else:
            self.df = df.iloc[split_idx:]
            print(f"Testing Dataset loaded with {len(self.df)} positions.")

        self.K = K

    def __len__(self):
        return len(self.df)

    def __getfen__(self, idx):
        row = self.df.iloc[idx]
        return row['fen']

    def __getcp__(self, idx):
        row = self.df.iloc[idx]
        return row['cp']

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        x = fen_to_tensor(row['fen'])
        
        cp = row['cp']
        mate = row['mate']
        
        if pd.notna(mate):
            y = 1.0 if mate > 0 else -1.0
        else:
            y = np.tanh(cp / self.K)
            
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = ChessDataset(parquet_path="data/kaggle_100k_300.parquet", train=True)
    for i in range(100):
        x, y = dataset[i]
        # print the board and the target value
        print(f"Sample {i}: Target value (y) = {y.item()}")
        # visualize the board
        board = chess.Board(dataset.__getfen__(i))
        print(board)
        print("\n")
    