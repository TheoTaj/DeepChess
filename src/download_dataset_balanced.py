import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

def count_material(fen):
    """Return the total material count from a FEN string"""
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9,
                    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    board_part = fen.split(' ')[0]
    return sum(piece_values.get(char, 0) for char in board_part)

def download_balanced_dataset(
    df_name="mateuszgrzyb/lichess-stockfish-normalized",
    output_dir="data",
    filename="dataset_stratified",
    total_rows=100_000,
    n_bins=20,
    K=750.0,
    seed=42,
    mate_frac=0.01,
    max_mate_depth=5,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{filename}.parquet")

    n_mate_target = int(total_rows * mate_frac)
    n_cp_target = total_rows - n_mate_target
    
    target_per_bin = n_cp_target // n_bins
    max_per_phase = n_cp_target // 3
    
    phase_edges = [0, 30, 60, 150]
    phases = ["Endgame", "Middlegame", "Opening"]

    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    counts_per_bin = [0 for _ in range(n_bins)]
    counts_per_phase = [0, 0, 0]
    
    collected_rows = []
    mate_rows = []
    cp_filled = 0
    mate_filled = 0

    print(f"Target per bin   : {target_per_bin}")
    print(f"Target per phase : {max_per_phase}")

    dataset = load_dataset(df_name, split="train", streaming=True, token=hf_token)
    dataset = dataset.shuffle(seed=seed, buffer_size=100_000)

    for row in dataset:
        if cp_filled == n_cp_target and mate_filled == n_mate_target:
            break

        fen = row['fen']
        is_mate = pd.notna(row.get('mate'))

        if is_mate:
            if mate_filled < n_mate_target:
                mate_depth = abs(int(row['mate']))
                if mate_depth <= max_mate_depth:
                    mate_rows.append({'fen': fen, 'cp': None, 'mate': row['mate'], 'material': count_material(fen)})
                    mate_filled += 1
            continue

        if cp_filled >= n_cp_target:
            continue

        cp = row.get('cp')
        if cp is None or pd.isna(cp):
            continue

        score = np.tanh(cp / K)
        bin_idx = np.clip(np.searchsorted(bin_edges, score, side='right') - 1, 0, n_bins - 1)

        material = count_material(fen)
        phase_idx = np.searchsorted(phase_edges, material, side='right') - 1
        phase_idx = np.clip(phase_idx, 0, 2)

        if counts_per_bin[bin_idx] < target_per_bin and counts_per_phase[phase_idx] < max_per_phase:
            collected_rows.append({
                'fen': fen,
                'cp': cp,
                'mate': None,
                'material': material
            })
            counts_per_bin[bin_idx] += 1
            counts_per_phase[phase_idx] += 1
            cp_filled += 1

        if (cp_filled + mate_filled) % 10_000 == 0:
            print(f"Progress: {cp_filled + mate_filled}/{total_rows} "
                  f"| Phases: E:{counts_per_phase[0]} M:{counts_per_phase[1]} O:{counts_per_phase[2]}")

    df = pd.DataFrame(collected_rows + mate_rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nFinal distributions :\n{df['material'].apply(lambda x: phases[np.searchsorted(phase_edges, x, side='right')-1] if pd.notna(x) else 'Mate').value_counts()}")
    
    df.to_parquet(full_path, index=False)
    print(f"Saved dataset : {full_path}")
if __name__ == "__main__":
    download_stratified_dataset(
        total_rows=100000, 
        n_bins=50, 
        K=750.0, 
        filename="df_100k_balanced_phases"
    )