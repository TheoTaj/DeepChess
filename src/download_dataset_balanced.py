import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

def download_balanced_dataset(
    df_name="mateuszgrzyb/lichess-stockfish-normalized",
    output_dir="data",
    filename="dataset_balanced_small",
    total_rows=100_000,
    n_bins=20,
    K=300.0,
    seed=42,
    mate_frac=0.01,   # 1% de positions de mat
    max_mate_depth=5, # profondeur max des mats acceptés
):
    """
    Creates a balanced parquet file where each score bin is equally represented.
    A small fraction of mate positions (shallow only) is included.
    
    Args:
        df_name (str): HuggingFace dataset name.
        output_dir (str): Output directory.
        filename (str): Output filename (without extension).
        total_rows (int): Total number of positions to collect.
        n_bins (int): Number of equally spaced bins over (-1, 1).
        K (float): Normalization constant for tanh(cp / K).
        seed (int): Random seed.
        mate_frac (float): Fraction of total_rows reserved for mate positions.
        max_mate_depth (int): Maximum mate-in-N accepted (e.g. 5 = mat en 5 coups max).
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{filename}.parquet")

    # ── Quotas ───────────────────────────────────────────────────────────────
    n_mate_target  = int(total_rows * mate_frac)
    n_cp_target    = total_rows - n_mate_target
    per_bin        = n_cp_target // n_bins
    bin_edges      = np.linspace(-1.0, 1.0, n_bins + 1)

    bins      = [[] for _ in range(n_bins)]
    mate_rows = []   # positions de mat acceptées
    cp_filled   = 0
    mate_filled = 0

    print(f"Target cp   : {n_cp_target:,} ({per_bin} per bin, {n_bins} bins)")
    print(f"Target mate : {n_mate_target:,} (max depth = {max_mate_depth})")
    print(f"Total       : {total_rows:,}")

    dataset = load_dataset(df_name, split="train", streaming=True, token=hf_token)
    dataset = dataset.shuffle(seed=seed, buffer_size=100_000)

    for row in dataset:
        if cp_filled == n_cp_target and mate_filled == n_mate_target:
            break

        is_mate = pd.notna(row.get('mate'))

        # ── Positions de mat ─────────────────────────────────────────────────
        if is_mate:
            if mate_filled >= n_mate_target:
                continue
            mate_depth = abs(int(row['mate']))
            if mate_depth > max_mate_depth:
                continue  # trop profond → ignoré
            mate_rows.append({
                'fen':  row['fen'],
                'cp':   None,
                'mate': row['mate'],
            })
            mate_filled += 1
            continue

        # ── Positions cp ─────────────────────────────────────────────────────
        if cp_filled >= n_cp_target:
            continue

        cp = row.get('cp')
        if cp is None or pd.isna(cp):
            continue

        score   = np.tanh(cp / K)
        bin_idx = np.clip(np.searchsorted(bin_edges, score, side='right') - 1, 0, n_bins - 1)

        if len(bins[bin_idx]) < per_bin:
            bins[bin_idx].append({
                'fen':  row['fen'],
                'cp':   cp,
                'mate': None,
            })
            cp_filled += 1

        if (cp_filled + mate_filled) % 10_000 == 0:
            counts = [len(b) for b in bins]
            print(f"  cp={cp_filled}/{n_cp_target}  mate={mate_filled}/{n_mate_target}"
                  f"  | bins: min={min(counts)}, max={max(counts)}")

    print(f"\nDone streaming.")
    print(f"  cp collected   : {cp_filled}")
    print(f"  mate collected : {mate_filled}  (depth ≤ {max_mate_depth})")
    print(f"  bin counts     : {[len(b) for b in bins]}")

    all_rows = [r for b in bins for r in b] + mate_rows
    df = pd.DataFrame(all_rows)
    if 'depth' in df.columns:
        df = df.drop(columns=['depth'])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Saving {len(df):,} rows to {full_path}...")
    df.to_parquet(full_path, compression='snappy', index=False)
    print("Done.")


if __name__ == "__main__":
    download_balanced_dataset(
        total_rows=5_000_000,
        n_bins=50,
        K=750.0,
        mate_frac=0.01,
        max_mate_depth=5,
        filename="df_5M_50_750"
    )