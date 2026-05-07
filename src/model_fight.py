import chess
import torch
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from game import ChessGame, load_model
from CNN import ChessCNN


# ── Worker function (must be top-level for multiprocessing) ──────────────────

def play_game(args):
    """
    Plays a single headless game and returns the result.
    Must be a top-level function for multiprocessing.
    """
    torch.set_num_threads(1)  # Important pour éviter les conflits CPU dans multiprocessing
    fen, model1_cfg, model2_cfg, white_idx, depth, game_id = args

    device = torch.device("cpu")  # multiprocessing → CPU par worker

    model1 = load_model(model1_cfg['path'], model1_cfg['conv_filters'],
                        model1_cfg['fc_layers'], device)
    model2 = load_model(model2_cfg['path'], model2_cfg['conv_filters'],
                        model2_cfg['fc_layers'], device)

    # white_idx=1 → model1 joue blancs, white_idx=2 → model2 joue blancs
    white_model = model1 if white_idx == 1 else model2
    black_model = model2 if white_idx == 1 else model1

    game = ChessGame(
        headless=True,
        white_model=white_model,
        black_model=black_model,
        white_depth=depth,
        black_depth=depth,
        device=device,
        starting_fen=fen,
    )
    print(f"  Starting game {game_id} | white=model{white_idx} | depth={depth} | fen={fen}")
    result = game.play_headless(max_moves=200)
    print(f"  Finished game {game_id} | result={result['result']} | moves={result['n_moves']}")

    # Traduit le résultat en termes de model1/model2
    if result['result'] == "draw":
        winner = "draw"
    elif (result['result'] == "white" and white_idx == 1) or \
         (result['result'] == "black" and white_idx == 2):
        winner = "model1"
    else:
        winner = "model2"

    return {
        'game_id':   game_id,
        'fen':       fen,
        'white':     f"model{white_idx}",
        'result':    result['result'],   # "white", "black", "draw"
        'winner':    winner,             # "model1", "model2", "draw"
        'n_moves':   result['n_moves'],
    }


# ── Stats ────────────────────────────────────────────────────────────────────

def print_stats(results, model1_name, model2_name):
    total  = len(results)
    m1_wins = sum(1 for r in results if r['winner'] == 'model1')
    m2_wins = sum(1 for r in results if r['winner'] == 'model2')
    draws   = sum(1 for r in results if r['winner'] == 'draw')

    # Winrate en tant que blancs / noirs
    m1_white = [r for r in results if r['white'] == 'model1']
    m2_white = [r for r in results if r['white'] == 'model2']

    m1_wins_as_white = sum(1 for r in m1_white if r['winner'] == 'model1')
    m1_wins_as_black = sum(1 for r in m2_white if r['winner'] == 'model1')
    m2_wins_as_white = sum(1 for r in m2_white if r['winner'] == 'model2')
    m2_wins_as_black = sum(1 for r in m1_white if r['winner'] == 'model2')

    avg_moves = sum(r['n_moves'] for r in results) / max(total, 1)

    print("\n" + "=" * 60)
    print(f"  RÉSULTATS — {model1_name} vs {model2_name}")
    print("=" * 60)
    print(f"  Total parties         : {total}")
    print(f"  Moyenne coups/partie  : {avg_moves:.1f}")
    print()
    print(f"  {model1_name:<20} : {m1_wins:>3} wins  ({100*m1_wins/total:.1f}%)")
    print(f"  {model2_name:<20} : {m2_wins:>3} wins  ({100*m2_wins/total:.1f}%)")
    print(f"  {'Draw':<20} : {draws:>3}       ({100*draws/total:.1f}%)")
    print()
    print(f"  {model1_name} as White : {m1_wins_as_white}/{len(m1_white)} wins  ({100*m1_wins_as_white/max(len(m1_white),1):.1f}%)")
    print(f"  {model1_name} as Black : {m1_wins_as_black}/{len(m2_white)} wins  ({100*m1_wins_as_black/max(len(m2_white),1):.1f}%)")
    print(f"  {model2_name} as White : {m2_wins_as_white}/{len(m2_white)} wins  ({100*m2_wins_as_white/max(len(m2_white),1):.1f}%)")
    print(f"  {model2_name} as Black : {m2_wins_as_black}/{len(m1_white)} wins  ({100*m2_wins_as_black/max(len(m1_white),1):.1f}%)")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model vs Model tournament")
    parser.add_argument("--model1_path",         type=str,   required=True)
    parser.add_argument("--model2_path",         type=str,   required=True)
    parser.add_argument("--model1_name",         type=str,   default="Model1")
    parser.add_argument("--model2_name",         type=str,   default="Model2")
    parser.add_argument("--conv_filters",        type=int,   nargs='+', required=True)
    parser.add_argument("--fc_layers",           type=int,   nargs='+', required=True)
    parser.add_argument("--depth",               type=int,   default=3)
    parser.add_argument("--n_fens",              type=int,   default=50)
    parser.add_argument("--n_workers",           type=int,   default=4)
    parser.add_argument("--opening_book",        type=str,   default="data/opening_book_fens.txt")
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Charger les FENs ─────────────────────────────────────────────────────
    with open(args.opening_book, 'r') as f:
        all_fens = [line.strip() for line in f if line.strip()]

    fens = random.sample(all_fens, min(args.n_fens, len(all_fens)))
    print(f"Loaded {len(fens)} FENs from {args.opening_book}")

    model1_cfg = {'path': args.model1_path, 'conv_filters': args.conv_filters, 'fc_layers': args.fc_layers}
    model2_cfg = {'path': args.model2_path, 'conv_filters': args.conv_filters, 'fc_layers': args.fc_layers}

    # ── Créer les jobs : 2 parties par FEN ───────────────────────────────────
    jobs = []
    for i, fen in enumerate(fens):
        jobs.append((fen, model1_cfg, model2_cfg, 1, args.depth, f"{i}_m1white"))  # model1 = blancs
        jobs.append((fen, model1_cfg, model2_cfg, 2, args.depth, f"{i}_m2white"))  # model2 = blancs

    print(f"Total games : {len(jobs)}  |  Workers : {args.n_workers}")

    # ── Lancer en parallèle ───────────────────────────────────────────────────
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(play_game, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"  [{i+1:>3}/{len(jobs)}] game_id={result['game_id']:<12}"
                  f" white={result['white']:<8}"
                  f" result={result['result']:<6}"
                  f" winner={result['winner']:<8}"
                  f" moves={result['n_moves']}")

    print_stats(results, args.model1_name, args.model2_name)