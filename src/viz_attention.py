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
import math

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
    if query_sq is not None:
        fill[query_sq] = '#FF0000'
    return fill

def draw_chessboard_heatmap(board_8x8, query_sq, fen, layer, head,
                             average_heads, average_layers, ax, norm=None, show_colorbar=True):
    board = chess.Board(fen)
    
    # Use provided norm if given, otherwise compute a local one (original behaviour)
    if norm is None:
        flat = board_8x8.flatten()
        norm = mcolors.Normalize(vmin=flat.min(), vmax=flat.max())
    
    fill = attention_to_colors_with_norm(board_8x8, query_sq, norm)

    svg_data = chess.svg.board(board=board, fill=fill, size=400)
    if query_sq is not None:
        sq_file = chess.square_file(query_sq)
        sq_rank = chess.square_rank(query_sq)
        square_size = 400 / 8
        x = sq_file * square_size
        y = (7 - sq_rank) * square_size
        border_svg = (
            f'<rect x="{x}" y="{y}" width="{square_size}" height="{square_size}" '
            f'fill="none" stroke="red" stroke-width="3"/>'
        )
        svg_data = svg_data.replace('</svg>', f'{border_svg}</svg>')
    png_data = svg2png(bytestring=svg_data.encode('utf-8'), scale=2.0)
    img = mpimg.imread(BytesIO(png_data), format='png')

    ax.imshow(img)
    ax.axis('off')

    query_str = sq_name(query_sq) if query_sq is not None else "all"
    if average_layers and average_heads:
        title = f"All layers & heads avg | Query: {query_str}"
    elif average_layers:
        title = f"All layers avg, Head {head} | Query: {query_str}"
    elif average_heads:
        title = f"Layer {layer}, All heads avg | Query: {query_str}"
    else:
        title = f"Layer {layer}, Head {head} | Query: {query_str}"

    sm = cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    if show_colorbar:
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

def attention_to_colors_with_norm(board_8x8, query_sq, norm):
    """Like attention_to_colors but accepts an external norm for shared scaling."""
    cmap = COLORMAP
    fill = {}
    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        weight = board_8x8[rank, file]
        rgba = cmap(norm(weight))
        fill[sq] = mcolors.to_hex(rgba)
    if query_sq is not None:
        fill[query_sq] = '#FF0000'
    return fill


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

def plot_global_attention(attn_weights, fen, save_dir="fig/attention_plots"):
    """Average attention over all query squares — global view comparable to GradCAM.
    Uses a shared colorbar across all layers for honest cross-layer comparison.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_layers = len(attn_weights)

    # --- Pass 1: compute all maps to find the global vmin/vmax ---
    global_maps = []
    for layer in range(n_layers):
        maps = []
        for sq in range(64):
            m = get_attention_map(attn_weights, sq, layer, head=0,
                                  average_heads=True, average_layers=False)
            maps.append(m)
        global_maps.append(np.stack(maps).mean(axis=0))  # [8, 8]

    all_values = np.concatenate([m.flatten() for m in global_maps])
    shared_norm = mcolors.Normalize(vmin=all_values.min(), vmax=all_values.max())

    # --- Pass 2: plot with the shared norm ---
    fig, axes = plt.subplots(2, n_layers // 2, figsize=(4 * n_layers // 2, 9))
    axes = axes.flatten()

    for layer, global_map in enumerate(global_maps):
        draw_chessboard_heatmap(global_map, query_sq=None, fen=fen, layer=layer,
                                 head=0, average_heads=True, average_layers=False,
                                 ax=axes[layer], norm=shared_norm, show_colorbar=False)
        axes[layer].set_title(f"Layer {layer}")

    # plt.suptitle("Global attention (avg over all query squares)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "global_attention_all_layers_black.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def compute_and_save_attention(fen, model, device, save_dir="fig/attention_cache"):
    """
    Run a forward pass for the given FEN and save all attention weights to disk.

    Saves:
      - attn_weights: raw tensors [1, n_heads, N, N] for each layer
      - fen: the position string
      - model config: n_layers, n_heads, N

    Args:
        fen      : FEN string of the position to analyse
        model    : loaded ChessViT (already eval()'d and on device)
        device   : torch device
        save_dir : directory where the .pt file will be written

    Returns:
        save_path : path to the saved file
    """

    os.makedirs(save_dir, exist_ok=True)

    # --- Forward pass with hooks ---
    attn_weights, handles = register_hooks(model)

    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)

    remove_hooks(handles)

    # --- Build payload ---
    # Convert each tensor to CPU numpy to make the file self-contained
    # (no torch version dependency issues when reloading later)
    attn_np = {
        layer_idx: tensor.cpu().numpy()          # [1, n_heads, N, N]
        for layer_idx, tensor in attn_weights.items()
    }

    n_layers = len(attn_np)
    n_heads  = attn_np[0].shape[1]
    N        = attn_np[0].shape[2]   # sequence length (64 squares + 1 CLS = 65)

    payload = {
        "fen"      : fen,
        "n_layers" : n_layers,
        "n_heads"  : n_heads,
        "N"        : N,
        "attn"     : attn_np,         # dict {int -> np.ndarray [1, n_heads, N, N]}
    }

    # Use a filename derived from the FEN (sanitised) so different positions
    # don't overwrite each other
    safe_fen = fen.split(" ")[0].replace("/", "-")   # keep only the board part
    save_path = os.path.join(save_dir, f"attn_{safe_fen}.pt")
    torch.save(payload, save_path)
    print(f"Saved attention weights → {save_path}")
    print(f"  layers={n_layers}, heads={n_heads}, seq_len={N}")

    return save_path


def load_attention(save_path):
    """
    Load attention weights previously saved by compute_and_save_attention().

    Returns:
        attn_weights : dict {layer_idx (int) -> torch.Tensor [1, n_heads, N, N]}
                       Same format as the live attn_weights dict produced by
                       register_hooks(), so every existing plot function works
                       unchanged.
        fen          : FEN string of the saved position
        meta         : dict with n_layers, n_heads, N
    """
    payload = torch.load(save_path, map_location="cpu", weights_only=False)

    # Restore as torch tensors so get_attention_map() keeps working as-is
    attn_weights = {
        layer_idx: torch.from_numpy(arr)
        for layer_idx, arr in payload["attn"].items()
    }

    fen  = payload["fen"]
    meta = {k: payload[k] for k in ("n_layers", "n_heads", "N")}

    print(f"Loaded attention weights ← {save_path}")
    print(f"  FEN    : {fen}")
    print(f"  layers={meta['n_layers']}, heads={meta['n_heads']}, seq_len={meta['N']}")

    return attn_weights, fen, meta

def plot_global_attention_per_head(attn_weights, fen, save_dir="fig/attention_plots"):
    """
    For each (layer, head) pair, compute the attention map averaged over all
    64 query squares, then plot the full 8-layer × 8-head grid with a single
    shared colormap scale.

    Layout: 8 rows (layers, top=0) × 8 columns (heads, left=0).
    """
    os.makedirs(save_dir, exist_ok=True)

    n_layers = len(attn_weights)
    n_heads  = attn_weights[0].shape[1]

    # --- Pass 1: compute all 64 maps and find the global vmin/vmax ---
    maps = {}                          # (layer, head) -> np.ndarray [8, 8]
    for layer in range(n_layers):
        for head in range(n_heads):
            avg = np.stack([
                get_attention_map(attn_weights, sq, layer, head,
                                  average_heads=False, average_layers=False)
                for sq in range(64)
            ]).mean(axis=0)            # [8, 8]
            maps[(layer, head)] = avg

    all_values  = np.concatenate([m.flatten() for m in maps.values()])
    shared_norm = mcolors.Normalize(vmin=all_values.min(), vmax=all_values.max())

    # --- Pass 2: plot ---
    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(2.5 * n_heads, 2.5 * n_layers)
    )

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            draw_chessboard_heatmap(
                maps[(layer, head)],
                query_sq=None,          # no red square — we averaged over all
                fen=fen,
                layer=layer, head=head,
                average_heads=False, average_layers=False,
                ax=ax,
                norm=shared_norm,
                show_colorbar=False,
            )
            ax.set_title(f"L{layer} H{head}", fontsize=7)

    # Row and column labels
    for layer in range(n_layers):
        axes[layer][0].set_ylabel(f"Layer {layer}", fontsize=8)
    for head in range(n_heads):
        axes[0][head].set_xlabel(f"Head {head}", fontsize=8)
        axes[0][head].xaxis.set_label_position("top")

    # Single shared colorbar on the right
    sm = cm.ScalarMappable(cmap=COLORMAP, norm=shared_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.01, pad=0.02)

    plt.suptitle("Global attention per head (avg over all 64 query squares)", fontsize=13)
    plt.tight_layout()

    path = os.path.join(save_dir, "global_attention_per_head.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def plot_global_attention_selected_heads(attn_weights, fen, selected_heads,
                                         save_dir="fig/attention_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # --- Pass 1: compute query-averaged map for each requested (layer, head) ---
    maps = {}
    for (layer, head) in selected_heads:
        maps[(layer, head)] = np.stack([
            get_attention_map(attn_weights, sq, layer, head,
                              average_heads=False, average_layers=False)
            for sq in range(64)
        ]).mean(axis=0)

    all_values  = np.concatenate([m.flatten() for m in maps.values()])
    shared_norm = mcolors.Normalize(vmin=all_values.min(), vmax=all_values.max())

    # --- Pass 2: plot ---
    n      = len(selected_heads)
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_rows),
                             squeeze=False,
                             layout="constrained")

    fig.patch.set_alpha(0)  # transparent background — let Typst/paper show through

    for idx, (layer, head) in enumerate(selected_heads):
        ax = axes[idx // n_cols][idx % n_cols]
        draw_chessboard_heatmap(
            maps[(layer, head)],
            query_sq=None,
            fen=fen,
            layer=layer, head=head,
            average_heads=False, average_layers=False,
            ax=ax,
            norm=shared_norm,
            show_colorbar=False,
        )
        ax.set_title(f"Layer {layer} — Head {head}", fontsize=7)

    for idx in range(len(selected_heads), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    sm = cm.ScalarMappable(cmap=COLORMAP, norm=shared_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)

    path = os.path.join(save_dir, "global_attention_selected_heads.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {path}")

if __name__ == "__main__":
    save_path = "fig/attention_cache/attn_rn2kb1r-pp3ppp-4p1qn-1p4B1-2B5-3P2QP-PPP2PP1-R3K2R.pt"
    attn_weights, fen, meta = load_attention(save_path)

    # heads = [(1,0), (0,3), (1,3),(1,4),(0,4),(1,7),(5,3),(5,2),(4,3),(5,4),(5,5),(4,5),(6,1),(6,7)]
    heads = [(1,3),(1,4),(5,3)]

    plot_global_attention_selected_heads(attn_weights, fen, heads)


# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     model = load_model(MODEL_PATH, device)
    
#     fen = "rn2kb1r/pp3ppp/4p1qn/1p4B1/2B5/3P2QP/PPP2PP1/R3K2R w KQkq - 0 1"
#     compute_and_save_attention(fen, model, device)

    # attn_weights, handles = register_hooks(model)

    # # fen = "8/8/8/8/4R3/8/8/8 w - - 0 1"
    # # fen = "r3k2r/ppp2ppp/8/8/4R3/8/PPP2PPP/R3K2R w KQkq - 0 1"
    # tensor_input = fen_to_tensor(fen)
    # x = torch.from_numpy(tensor_input).float().unsqueeze(0).to(device)  # [1, 18, 8, 8]
    # with torch.no_grad():
    #     model(x)

    # remove_hooks(handles)

    # # --- Single map ---
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
    
    # plot_all_heads_all_layers(attn_weights, QUERY_SQ, fen)
    # plot_avg_heads_all_layers(attn_weights, QUERY_SQ, fen)
    # plot_avg_layers_all_heads(attn_weights, QUERY_SQ, fen)
    # plot_global_attention(attn_weights, fen)

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