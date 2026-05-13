import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

def count_material(fen):
    """Return the total material count from a FEN string"""
    board_part = fen.split(' ')[0]
    values = {
        'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9,
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9
    }
    return sum(values.get(char, 0) for char in board_part)

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

def balance_dataset_from_parquet(
    input_path,
    output_dir="data",
    filename="dataset_balanced_2D",
    total_rows=100_000,
    n_bins=20,
    K=750.0,
    seed=42,
    mate_frac=0.01,
    max_mate_depth=5,
):
    """
    Loads a parquet file with columns [FEN, Evaluation] and creates a balanced dataset.
    Balanced on two independent dimensions:
        - Score bins : equal number of positions per bin
        - Game phase : equal number of Opening / Middlegame / Endgame globally
    Evaluation format: "+56", "-10" for cp, "#3" or "#-6" for mate.

    Args:
        input_path (str)  : Path to the input parquet file.
        output_dir (str)  : Output directory.
        filename (str)    : Output filename (without extension).
        total_rows (int)  : Total number of positions to collect.
        n_bins (int)      : Number of score bins over (-1, 1).
        K (float)         : Normalization constant for tanh(cp / K).
        seed (int)        : Random seed.
        mate_frac (float) : Fraction of total_rows reserved for mate positions.
        max_mate_depth (int): Maximum mate-in-N accepted.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{filename}.parquet")

    def parse_evaluation(eval_str):
        """
        Parses the evaluation string.
        Returns (cp, mate) where one of them is None.
        Examples:
            "+56"  → (56,   None)
            "-10"  → (-10,  None)
            "#3"   → (None, 3)     white mates in 3
            "#-6"  → (None, -6)    black mates in 6
        """
        eval_str = str(eval_str).strip()
        if eval_str.startswith('#'):
            return None, int(eval_str[1:])
        else:
            return int(eval_str), None

    # ── Quotas ───────────────────────────────────────────────────────────────
    n_mate_target    = int(total_rows * mate_frac)
    n_cp_target      = total_rows - n_mate_target
    target_per_bin   = n_cp_target // n_bins   # contrainte par bin
    target_per_phase = n_cp_target // 3        # contrainte globale par phase

    bin_edges    = np.linspace(-1.0, 1.0, n_bins + 1)
    phase_edges  = [0, 30, 60, 150]
    phases       = ["Endgame", "Middlegame", "Opening"]

    counts_per_bin   = [0] * n_bins
    counts_per_phase = [0, 0, 0]

    collected_rows = []
    mate_rows      = []
    cp_filled      = 0
    mate_filled    = 0

    print(f"Loading {input_path}...")
    df_input = pd.read_parquet(input_path)
    df_input = df_input.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Loaded {len(df_input):,} rows.")
    print(f"Target per bin   : {target_per_bin}")
    print(f"Target per phase : {target_per_phase}")
    print(f"Target mate      : {n_mate_target} (depth ≤ {max_mate_depth})")
    print(f"Total target     : {total_rows:,}\n")

    for _, row in df_input.iterrows():
        if cp_filled == n_cp_target and mate_filled == n_mate_target:
            break

        fen      = row['FEN']
        eval_str = row['Evaluation']

        try:
            cp, mate = parse_evaluation(eval_str)
        except (ValueError, TypeError):
            continue

        # ── Positions de mat ─────────────────────────────────────────────────
        if mate is not None:
            if mate_filled < n_mate_target and abs(mate) <= max_mate_depth:
                mate_rows.append({
                    'fen':  fen,
                    'cp':   None,
                    'mate': mate,
                })
                mate_filled += 1
            continue

        # ── Positions cp ─────────────────────────────────────────────────────
        if cp_filled >= n_cp_target:
            continue

        score     = np.tanh(cp / K)
        bin_idx   = np.clip(np.searchsorted(bin_edges, score, side='right') - 1, 0, n_bins - 1)
        material  = count_material(fen)
        phase_idx = np.clip(np.searchsorted(phase_edges, material, side='right') - 1, 0, 2)

        if counts_per_bin[bin_idx] < target_per_bin and counts_per_phase[phase_idx] < target_per_phase:
            collected_rows.append({
                'fen':  fen,
                'cp':   cp,
                'mate': None,
            })
            counts_per_bin[bin_idx]     += 1
            counts_per_phase[phase_idx] += 1
            cp_filled += 1

        if (cp_filled + mate_filled) % 10_000 == 0:
            print(f"  cp={cp_filled}/{n_cp_target}  mate={mate_filled}/{n_mate_target}"
                  f"  | E:{counts_per_phase[0]:,}  M:{counts_per_phase[1]:,}  O:{counts_per_phase[2]:,}")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    df_out = pd.DataFrame(collected_rows + mate_rows)
    df_out = df_out.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nFinal phase distribution :")
    for phase, count in zip(phases, counts_per_phase):
        print(f"  {phase:<12} : {count:>8,}  ({100*count/max(cp_filled,1):.1f}%)")
    print(f"  Mate         : {mate_filled:>8,}")
    print(f"  Total        : {len(df_out):>8,}")

    df_out.to_parquet(full_path, index=False, compression='snappy')
    print(f"\nSaved → {full_path}")

def generate_poster_dataset_plots(parquet_path, K=750.0, path="fig/distribution.png", sample_size=None):
    """
    Generates a high-quality figure for the poster containing:
    - Normalized target distribution (tanh(cp/K))
    - Game phase distribution (bar chart)
    """
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Analyzing {len(df):,} positions.")

    mate_mask = df['mate'].notna()
    cp_mask = ~mate_mask
    
    y_cp = np.tanh(df[cp_mask]['cp'] / K)
    # Map mates to exact -1 or 1
    y_mate = df[mate_mask]['mate'].apply(lambda m: 1.0 if m > 0 else -1.0)
    y_all = pd.concat([y_cp, y_mate])

    if 'material' not in df.columns:
        print("Computing material counts...")
        df['material'] = df['fen'].apply(count_material)

    phase_edges = [0, 30, 60, 150]
    phases = ["Endgame", "Middlegame", "Opening"]
    df['phase'] = pd.cut(df['material'], bins=phase_edges, labels=phases)

    plt.style.use('ggplot') 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(y_all, bins=80, color='#69b3a2', edgecolor='white', alpha=0.9)
    axes[0].set_title("Normalized Evaluation Distribution", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel(f"Target Score: $y = \\tanh(cp / {K})$", fontsize=12)
    axes[0].set_ylabel("Number of Positions", fontsize=12)
    axes[0].axvline(0, color='#e74c3c', linewidth=1.5, linestyle='--')
    axes[0].grid(axis='y', alpha=0.3)

    phase_counts = [(df['phase'] == p).sum() for p in phases]
    colors = ['#432371', '#7b4397', '#dc2430'] # Nice gradient palette
    bars = axes[1].bar(phases, phase_counts, color=colors, edgecolor='none', alpha=0.85)
    
    axes[1].set_title("Game Phase Distribution", fontsize=14, fontweight='bold', pad=15)
    # axes[1].set_ylabel("Number of Positions", fontsize=12)
    axes[1].set_xlabel("Game Stage", fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, phase_counts):
        percentage = 100 * count / len(df)
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (len(df) * 0.01),
            f'{percentage:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{path}' (300 DPI)")
    plt.show()

def load_opening_book_fens(file_path, n=50):
    """
    Lit le fichier Book.txt et extrait n FENs uniques au hasard.
    """
    import random
    fens = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # On ne garde que les lignes qui commencent par 'pos'
                if line.startswith('pos '):
                    # On retire le préfixe 'pos ' pour ne garder que la FEN
                    fen = line.replace('pos ', '').strip()
                    fens.append(fen)
        
        unique_fens = list(set(fens))
        
        print(f"Total de positions trouvées : {len(fens)}")
        print(f"Positions uniques : {len(unique_fens)}")
        
        if len(unique_fens) < n:
            print(f"Attention : Seulement {len(unique_fens)} positions uniques disponibles.")
            return unique_fens
        
        selected_fens = random.sample(unique_fens, n)
        
        with open(f"data/opening_book_fens_{n}.txt", "w") as f:
            for fen in selected_fens:
                f.write(f"{fen}\n")
        

    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return []


if __name__ == "__main__":
    # get_fighting_fens(
    #     parquet_path="data/kaggle_100k_300.parquet",
    #     n=50,
    #     output_path="data/fighting_fens_2.txt"
    # )
    generate_poster_dataset_plots(
        parquet_path="data/kaggle_5M_300.parquet",
        K=300.0,
        path="fig/poster_dataset_distribution.png",
        sample_size=None,
    )