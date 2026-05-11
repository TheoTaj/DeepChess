import torch
import numpy as np
import os
import chess
import chess.svg
from cairosvg import svg2png
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ViT import ChessViT
from dataset import fen_to_tensor, ChessDataset

# Config
MODEL_PATH   = "models/ViT3.pth" # Best model with 8 blocks and 8 heads per block
LAYER        = 6          # 0–7
HEAD         = 3          # 0–7
QUERY_SQ     = 28         # 0–63
AVERAGE_HEADS = False     # True to average all heads in a layer
AVERAGE_LAYERS = False    # True to average all layers (ignores LAYER)
FLIP_BOARD = False
FILES = 'abcdefgh'
COLORMAP = plt.cm.viridis

def save_plot(fig, description, save_dir="fig/attention_plots"):
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{description}_sq{sq_name(QUERY_SQ)}.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def head_consistency(model, fens, query_sq, layer, head, device):
    """
    Returns the std of attention maps across positions for a given head.
    Low std = consistent = potentially interpretable.
    """
    maps = []
    for fen in fens:
        attn_weights, handles = register_hooks(model)
        x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            model(x)
        remove_hooks(handles)
        m = get_attention_map(attn_weights, query_sq, layer, head,
                              average_heads=False, average_layers=False)
        maps.append(m)
    
    maps = np.stack(maps)       # [n_positions, 8, 8]
    return maps.std(axis=0).mean()  # scalar: mean std across squares

def sq_name(idx):
    """0–63 → e.g. 'e4'. Assumes a1=0, h1=7, a2=8, ..."""
    return f"{FILES[idx % 8]}{idx // 8 + 1}"

def board_index(file, rank):
    """'e', 4 → square index. file: 'a'–'h', rank: 1–8."""
    return (rank - 1) * 8 + FILES.index(file)

def load_model(path, device):
    model = ChessViT(n_blocks=8, n_heads=8, embed_dim=384, mlp_dim=768, dropout=0.065)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

def register_hooks(model):
    """Register hooks on all blocks, return the storage dict and hook handles."""
    attn_weights = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output[1]: [B, n_heads, N, N] (thanks to average_attn_weights=False)
            attn_weights[layer_idx] = output[1].detach().cpu()
        return hook

    for i, block in enumerate(model.blocks):
        handle = block.attention.register_forward_hook(make_hook(i))
        handles.append(handle)

    return attn_weights, handles

def remove_hooks(handles):
    for h in handles:
        h.remove()

def get_attention_map(attn_weights, query_sq, layer, head, 
                      average_heads, average_layers):
    """
    Extract and return an (8, 8) attention map for the query square.
    Square tokens start at index 1 (index 0 is CLS).
    """
    query_idx = query_sq + 1  # offset for CLS token

    if average_layers:
        # Stack all layers: [n_layers, B, n_heads, N, N]
        stacked = torch.stack([attn_weights[l] for l in range(len(attn_weights))])
        attn = stacked.mean(0)  # [B, n_heads, N, N]
    else:
        attn = attn_weights[layer]  # [B, n_heads, N, N]

    if average_heads:
        attn = attn.mean(dim=1)     # [B, N, N]
        row = attn[0, query_idx, 1:]  # [64], skip CLS column
    else:
        row = attn[0, head, query_idx, 1:]  # [64]

    board = row.numpy().reshape(8, 8)
    if FLIP_BOARD:
        board = np.flipud(board)
    return board

def attention_to_colors(board_8x8, query_sq):
    """
    Map attention weights to colors like the paper:
    low attention = purple, high attention = yellow.
    Query square highlighted in red.
    """
    cmap = COLORMAP
    flat = board_8x8.flatten()
    norm = mcolors.Normalize(vmin=flat.min(), vmax=flat.max())

    fill = {}
    for sq in chess.SQUARES:  # chess.SQUARES: a1=0 ... h8=63
        rank = chess.square_rank(sq)  # 0–7
        file = chess.square_file(sq)  # 0–7
        weight = board_8x8[rank, file]
        rgba = cmap(norm(weight))
        fill[sq] = mcolors.to_hex(rgba)

    # Highlight query square in red
    fill[query_sq] = '#FF0000'
    return fill

def draw_chessboard_heatmap(board_8x8, query_sq, fen, layer, head,
                             average_heads, average_layers, ax):
    board = chess.Board(fen)
    fill  = attention_to_colors(board_8x8, query_sq)

    svg_data = chess.svg.board(board=board, fill=fill, size=400)
    png_data = svg2png(bytestring=svg_data.encode('utf-8'), scale=2.0)
    img = mpimg.imread(BytesIO(png_data), format='png')

    ax.imshow(img)
    ax.axis('off')

    if average_layers and average_heads:
        title = f"All layers & heads avg | Query: {sq_name(query_sq)}"
    elif average_layers:
        title = f"All layers avg, Head {head} | Query: {sq_name(query_sq)}"
    elif average_heads:
        title = f"Layer {layer}, All heads avg | Query: {sq_name(query_sq)}"
    else:
        title = f"Layer {layer}, Head {head} | Query: {sq_name(query_sq)}"

    flat = board_8x8.flatten()
    norm = mcolors.Normalize(vmin=flat.min(), vmax=flat.max())
    sm = cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)

def plot_all_heads(attn_weights, query_sq, layer, fen):
    """Plot all 8 heads for a given layer side by side."""
    n_heads = attn_weights[layer].shape[1]
    fig, axes = plt.subplots(2, n_heads // 2, figsize=(4 * n_heads // 2, 9))
    axes = axes.flatten()
    for h in range(n_heads):
        board = get_attention_map(attn_weights, query_sq, layer, h,
                                  average_heads=False, average_layers=False)
        draw_chessboard_heatmap(board, query_sq, fen, layer, h,
                                 average_heads=False, average_layers=False, ax=axes[h])
    plt.suptitle(f"All heads — Layer {layer} | Query: {sq_name(query_sq)}", 
                 fontsize=14)
    plt.tight_layout()
    save_plot(fig, f"all_heads_layer{layer}")

def plot_all_layers(attn_weights, query_sq, head, fen):
    """Plot a single head across all 8 layers."""
    n_layers = len(attn_weights)
    fig, axes = plt.subplots(2, n_layers // 2, figsize=(4 * n_layers // 2, 9))
    axes = axes.flatten()
    for l in range(n_layers):
        board = get_attention_map(attn_weights, query_sq, l, head,
                                  average_heads=False, average_layers=False)
        draw_chessboard_heatmap(board, query_sq, fen, l, head,
                                 average_heads=False, average_layers=False, ax=axes[l])
    plt.suptitle(f"Head {head} across all layers | Query: {sq_name(query_sq)}",
                 fontsize=14)
    plt.tight_layout()
    save_plot(fig, f"all_layers_head{head}")

def plot_all_heads_all_layers(attn_weights, query_sq, fen, save_dir="fig/attention_plots"):
    os.makedirs(save_dir, exist_ok=True)
    for layer in range(len(attn_weights)):
        n_heads = attn_weights[layer].shape[1]
        fig, axes = plt.subplots(2, n_heads // 2, figsize=(4 * n_heads // 2, 9))
        axes = axes.flatten()
        for h in range(n_heads):
            board = get_attention_map(attn_weights, query_sq, layer, h,
                                      average_heads=False, average_layers=False)
            draw_chessboard_heatmap(board, query_sq, fen, layer, h,
                                     average_heads=False, average_layers=False, ax=axes[h])
        plt.suptitle(f"All heads — Layer {layer} | Query: {sq_name(query_sq)}", fontsize=14)
        plt.tight_layout()
        
        path = os.path.join(save_dir, f"all_heads_layer{layer}_sq{sq_name(query_sq)}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

def make_rook_mask(sq):
    rank, file = sq // 8, sq % 8
    mask = np.zeros((8, 8))
    mask[rank, :] = 1
    mask[:, file] = 1
    mask[rank, file] = 0  # exclude the square itself
    return mask

def make_bishop_mask(sq):
    rank, file = sq // 8, sq % 8
    mask = np.zeros((8, 8))
    for d in range(1, 8):
        for dr, df in [(d,d),(d,-d),(-d,d),(-d,-d)]:
            r, f = rank+dr, file+df
            if 0 <= r < 8 and 0 <= f < 8:
                mask[r, f] = 1
    return mask

def make_knight_mask(sq):
    rank, file = sq // 8, sq % 8
    mask = np.zeros((8, 8))
    for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        r, f = rank+dr, file+df
        if 0 <= r < 8 and 0 <= f < 8:
            mask[r, f] = 1
    return mask

def make_king_mask(sq):
    rank, file = sq // 8, sq % 8
    mask = np.zeros((8, 8))
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            if dr == 0 and df == 0:
                continue
            r, f = rank + dr, file + df
            if 0 <= r < 8 and 0 <= f < 8:
                mask[r, f] = 1
    return mask

def score_against_mask(attn_map, mask):
    """Pearson correlation between attention map and a binary movement mask."""
    a = attn_map.flatten()
    m = mask.flatten()
    return np.corrcoef(a, m)[0, 1]

def scan_all_heads(model, fens, query_sq, device):
    masks = {
        'rook':   make_rook_mask(query_sq),
        'bishop': make_bishop_mask(query_sq),
        'knight': make_knight_mask(query_sq),
        'king':   make_king_mask(query_sq),
    }

    # Accumulate maps per (layer, head) across positions
    all_maps = {(l, h): [] for l in range(8) for h in range(8)}

    for fen in fens:
        attn_weights, handles = register_hooks(model)
        x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            model(x)
        remove_hooks(handles)

        for layer in range(8):
            for head in range(8):
                m = get_attention_map(attn_weights, query_sq, layer, head,
                                      average_heads=False, average_layers=False)
                all_maps[(layer, head)].append(m)

    # Now compute consistency + pattern scores
    results = []
    for layer in range(8):
        for head in range(8):
            maps = np.stack(all_maps[(layer, head)])  # [50, 8, 8]
            consistency = maps.std(axis=0).mean()
            mean_map = maps.mean(axis=0)  # average map is more robust for scoring
            scores = {name: score_against_mask(mean_map, mask)
                      for name, mask in masks.items()}
            best_match = max(scores, key=scores.get)
            results.append({
                'layer': layer, 'head': head,
                'consistency': consistency,
                'best_match': best_match,
                **{f'corr_{k}': v for k, v in scores.items()}
            })

    import pandas as pd
    return pd.DataFrame(results).sort_values('consistency')

def plot_avg_heads_all_layers(attn_weights, query_sq, fen, save_dir="fig/attention_plots"):
    """For each layer, show the average over all heads."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, len(attn_weights) // 2, figsize=(4 * len(attn_weights) // 2, 9))
    axes = axes.flatten()
    for layer in range(len(attn_weights)):
        board = get_attention_map(attn_weights, query_sq, layer, head=0,
                                  average_heads=True, average_layers=False)
        draw_chessboard_heatmap(board, query_sq, fen, layer, head=0,
                                 average_heads=True, average_layers=False, ax=axes[layer])
    plt.suptitle(f"Heads averaged — all layers | Query: {sq_name(query_sq)}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"avg_heads_all_layers_sq{sq_name(query_sq)}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_avg_layers_all_heads(attn_weights, query_sq, fen, save_dir="fig/attention_plots"):
    """For each head, show the average over all layers."""
    os.makedirs(save_dir, exist_ok=True)
    n_heads = attn_weights[0].shape[1]
    fig, axes = plt.subplots(2, n_heads // 2, figsize=(4 * n_heads // 2, 9))
    axes = axes.flatten()
    for head in range(n_heads):
        board = get_attention_map(attn_weights, query_sq, layer=0, head=head,
                                  average_heads=False, average_layers=True)
        draw_chessboard_heatmap(board, query_sq, fen, layer=0, head=head,
                                 average_heads=False, average_layers=True, ax=axes[head])
    plt.suptitle(f"Layers averaged — all heads | Query: {sq_name(query_sq)}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"avg_layers_all_heads_sq{sq_name(query_sq)}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(MODEL_PATH, device)
    attn_weights, handles = register_hooks(model)

    fen = "rn2kb1r/pp3ppp/4p1qn/1p4B1/2B5/3P2QP/PPP2PP1/R3K2R w KQkq - 0 1"
    # fen = "8/8/8/8/4R3/8/8/8 w - - 0 1"
    # fen = "r3k2r/ppp2ppp/8/8/4R3/8/PPP2PPP/R3K2R w KQkq - 0 1"
    tensor_input = fen_to_tensor(fen)
    x = torch.from_numpy(tensor_input).float().unsqueeze(0).to(device)  # [1, 18, 8, 8]
    with torch.no_grad():
        model(x)

    remove_hooks(handles)

    # --- Single map ---
    # fig, ax = plt.subplots(figsize=(5, 5))
    # board = get_attention_map(attn_weights, QUERY_SQ, LAYER, HEAD,
    #                           AVERAGE_HEADS, AVERAGE_LAYERS)
    # draw_chessboard_heatmap(board, QUERY_SQ, fen, LAYER, HEAD,
    #                          AVERAGE_HEADS, AVERAGE_LAYERS, ax)
    # plt.tight_layout()
    # save_plot(fig, f"single_layer{LAYER}_head{HEAD}")

    # --- Uncomment to explore further ---
    # plot_all_heads(attn_weights, QUERY_SQ, layer=LAYER, fen=fen)
    # plot_all_layers(attn_weights, QUERY_SQ, head=HEAD, fen=fen)
    plot_all_heads_all_layers(attn_weights, QUERY_SQ, fen)
    plot_avg_heads_all_layers(attn_weights, QUERY_SQ, fen)
    plot_avg_layers_all_heads(attn_weights, QUERY_SQ, fen)

# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = load_model(MODEL_PATH, device)

#     dataset_df = ChessDataset(parquet_path="data/kaggle_100k_300.parquet", K=300, train=True, train_ratio=1).df
#     random_rows = dataset_df.sample(n=50, random_state=42)
#     fens = random_rows["fen"].tolist()
#     query_sq = board_index('e', 4)  # central square, best for pattern visibility

#     df = scan_all_heads(model, fens, query_sq, device)
#     df.to_csv("fig/attention_plots/head_scan.csv", index=False)
#     print(df.head(20))