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

from CNN import ChessCNN
from dataset import fen_to_tensor
from viz_attention import sq_name, board_index, COLORMAP, FILES


MODEL_PATH = "models/CNN_5M_5.pth"
LAYER = 2     # which conv layer to target (0,1,2)


def save_plot(fig, description, save_dir="fig/gradcam"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{description}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def load_model(path, device):
    model = ChessCNN(
        conv_filters=[64, 128, 256],
        conv_kernels=[5, 3, 3],
        fc_dim=[512, 256, 128],
        dropout=0.05,
    )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

def get_conv_layers(model):
    """Extract the Conv2d layers from the conv_block in order."""
    return [m for m in model.conv_block if isinstance(m, torch.nn.Conv2d)]

def compute_gradcam(model, x, target_layer_idx):
    """
    Compute GradCAM heatmap for a given conv layer.
    
    target_layer_idx: which Conv2d to target (0=first, 1=second, 2=third/last)
    x: input tensor [1, 18, 8, 8], requires_grad not needed on input
    
    Returns an (8, 8) numpy array.
    """
    conv_layers = get_conv_layers(model)
    target_layer = conv_layers[target_layer_idx]

    # Storage for forward activations and backward gradients
    activations = {}
    gradients   = {}

    def forward_hook(module, input, output):
        activations['value'] = output  # [1, C, 8, 8]

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]  # [1, C, 8, 8]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass — need gradients so no torch.no_grad()
    output = model(x)  # scalar output [1, 1]
    
    # Backward pass w.r.t. the model output
    model.zero_grad()
    output.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # GradCAM computation
    acts = activations['value'].detach().cpu()  # [1, C, 8, 8]
    grads = gradients['value'].detach().cpu()   # [1, C, 8, 8]

    # Global average pool the gradients over spatial dims → channel weights
    weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

    # Weighted sum of activation maps
    cam = (weights * acts).sum(dim=1).squeeze(0)    # [8, 8]
    cam = torch.clamp(cam, min=0).numpy()            # ReLU, [8, 8]

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam

def draw_gradcam(cam_8x8, fen, layer_idx, ax):
    """Reuses the same chess SVG rendering as viz_attention.py."""
    board = chess.Board(fen)

    cmap = COLORMAP
    norm = mcolors.Normalize(vmin=cam_8x8.min(), vmax=cam_8x8.max())
    fill = {}
    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        rgba = cmap(norm(cam_8x8[rank, file]))
        fill[sq] = mcolors.to_hex(rgba)

    svg_data = chess.svg.board(board=board, fill=fill, size=400)
    png_data = svg2png(bytestring=svg_data.encode('utf-8'), scale=2.0)
    img = mpimg.imread(BytesIO(png_data), format='png')

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"GradCAM — Conv layer {layer_idx}")

    sm = cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

def plot_all_layers_gradcam(model, x, fen, save_dir="fig/gradcam"):
    """Plot GradCAM for all 3 conv layers side by side."""
    n_layers = len(get_conv_layers(model))
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 6))
    for i, ax in enumerate(axes):
        cam = compute_gradcam(model, x, target_layer_idx=i)
        draw_gradcam(cam, fen, i, ax)
    plt.suptitle("GradCAM — all conv layers", fontsize=14)
    plt.tight_layout()
    save_plot(fig, "gradcam_all_layers")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(MODEL_PATH, device)

    fen = "rn2kb1r/pp3ppp/4p1qn/1p4B1/2B5/3P2QP/PPP2PP1/R3K2R w KQkq - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).to(device)

    # Single layer
    fig, ax = plt.subplots(figsize=(5, 5))
    cam = compute_gradcam(model, x, target_layer_idx=LAYER)
    draw_gradcam(cam, fen, LAYER, ax)
    plt.tight_layout()
    save_plot(fig, f"gradcam_layer{LAYER}")

    # All layers side by side
    plot_all_layers_gradcam(model, x, fen)