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
    n_bins=50,
    K=300.0,
    seed=42,
):
    """
    Creates a balanced parquet file where each score bin is equally represented.
    Mate positions are excluded. Only cp positions are used.
    
    Args:
        df_name (str): HuggingFace dataset name.
        output_dir (str): Output directory.
        filename (str): Output filename (without extension).
        total_rows (int): Total number of positions to collect.
        n_bins (int): Number of equally spaced bins over (-1, 1).
        K (float): Normalization constant for tanh(cp / K).
        seed (int): Random seed.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{filename}.parquet")

    per_bin = total_rows // n_bins
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bins = [[] for _ in range(n_bins)]
    filled = 0

    print(f"Target: {per_bin} positions per bin, {n_bins} bins, {total_rows} total.")

    dataset = load_dataset(df_name, split="train", streaming=True, token=hf_token)
    dataset = dataset.shuffle(seed=seed, buffer_size=100_000)

    for row in dataset:
        if filled == total_rows:
            break

        # Skip mate positions
        if pd.notna(row.get('mate')):
            continue
        
        cp = row.get('cp')
        if cp is None or pd.isna(cp):
            continue

        score = np.tanh(cp / K)
        
        # Find which bin this score belongs to
        bin_idx = np.searchsorted(bin_edges, score, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        if len(bins[bin_idx]) < per_bin:
            bins[bin_idx].append({
                'fen': row['fen'],
                'cp': cp,
                'mate': None,
            })
            filled += 1

            if filled % 10_000 == 0:
                counts = [len(b) for b in bins]
                print(f"  Collected {filled}/{total_rows} | bin counts: min={min(counts)}, max={max(counts)}")

    print(f"\nDone streaming. Collected {filled} positions.")
    print("Bin counts:", [len(b) for b in bins])

    all_rows = [row for b in bins for row in b]
    df = pd.DataFrame(all_rows)
    if 'depth' in df.columns:
        df = df.drop(columns=['depth'])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle before saving

    print(f"Saving to {full_path}...")
    df.to_parquet(full_path, compression='snappy', index=False)
    print("Done.")

if __name__ == "__main__":
    # download_balanced_dataset(
    #     total_rows=5_000_000,
    #     n_bins=50,
    #     K=300.0,
    #     filename="dataset_balanced_large"
    # )

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # --- Config ---
    PARQUET_PATH = "data/dataset_balanced_large.parquet"
    OUTPUT_IMG   = "balanced_large.png"
    K = 300
    N = 5_000_000
    # --------------

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Dataset length: {len(df)}")
    print(df.head())

    scores = np.tanh(df["cp"].values[:N] / K)

    plt.hist(scores, bins=100)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"Normalized scores tanh(cp / {K})")
    plt.axvline(0, color='red', linestyle='--', label='0 (equal)')
    plt.legend()
    plt.savefig("balanced_large.png")
    plt.show()