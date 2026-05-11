import chess
import chess.engine
import torch
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from game import ChessGame, StockfishPlayer
from ONNX import ONNXModelWrapper

# ── Player factory ───────────────────────────────────────────────────────────

def make_player(path, stockfish_path=None, elo=None, time_limit=0.1, depth=None):
    """
    Returns an ONNXModelWrapper or a StockfishPlayer depending on `path`.
    Pass path="stockfish" to get a StockfishPlayer.
    """
    if path == "stockfish":
        assert stockfish_path is not None, "--stockfish_path required when using stockfish"
        return StockfishPlayer(stockfish_path, elo=elo, time_limit=time_limit, depth=depth)
    return ONNXModelWrapper(path)


# ── Worker function ──────────────────────────────────────────────────────────

def play_game(args):
    """
    Plays a single headless game. Handles both ONNX vs ONNX and ONNX vs Stockfish.
    Must be a top-level function for multiprocessing.
    """
    print(f"--- Worker started for game {args[7]} ---", flush=True)

    player1 = player2 = None
    try:
        torch.set_num_threads(1)
        fen, p1_path, p2_path, stockfish_path, elo, sf_depth, white_idx, game_id, depth, time_limit = args
        device = torch.device("cpu")

        player1 = make_player(p1_path, stockfish_path, elo, time_limit, sf_depth)
        player2 = make_player(p2_path, stockfish_path, elo, time_limit, sf_depth)

        white_model = player1 if white_idx == 1 else player2
        black_model = player2 if white_idx == 1 else player1

        game = ChessGame(
            root=None,
            white_model=white_model,
            black_model=black_model,
            white_depth=depth,
            black_depth=depth,
            device=device,
            starting_fen=fen,
            headless=True,
        )

        print(f"  [Game {game_id}] Initialized. Playing...", flush=True)
        result = game.play_headless(max_moves=200)
        print(f"  [Game {game_id}] Done. Result: {result['result']}", flush=True)

        winner = "draw"
        if result['result'] == 'white':
            winner = 'player1' if white_idx == 1 else 'player2'
        elif result['result'] == 'black':
            winner = 'player2' if white_idx == 1 else 'player1'

        return {
            'game_id': game_id,
            'fen':     fen,
            'white':   'player1' if white_idx == 1 else 'player2',
            'result':  result['result'],
            'winner':  winner,
            'n_moves': result['n_moves'],
        }

    except Exception as e:
        print(f"CRITICAL ERROR in Game {args[7]}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

    finally:
        for p in [player1, player2]:
            if isinstance(p, StockfishPlayer):
                try:
                    p.close()
                except Exception:
                    pass


# ── Stats ────────────────────────────────────────────────────────────────────

def print_stats(results, name1, name2):
    results = [r for r in results if r is not None]
    total   = len(results)
    if total == 0:
        print("No valid results.")
        return

    p1_wins = sum(1 for r in results if r['winner'] == 'player1')
    p2_wins = sum(1 for r in results if r['winner'] == 'player2')
    draws   = sum(1 for r in results if r['winner'] == 'draw')

    p1_white = [r for r in results if r['white'] == 'player1']
    p2_white = [r for r in results if r['white'] == 'player2']

    p1_wins_as_white = sum(1 for r in p1_white if r['winner'] == 'player1')
    p1_wins_as_black = sum(1 for r in p2_white if r['winner'] == 'player1')
    p2_wins_as_white = sum(1 for r in p2_white if r['winner'] == 'player2')
    p2_wins_as_black = sum(1 for r in p1_white if r['winner'] == 'player2')

    avg_moves = sum(r['n_moves'] for r in results) / total

    print("\n" + "=" * 60)
    print(f"  RESULTS — {name1} vs {name2}")
    print("=" * 60)
    print(f"  Total games           : {total}")
    print(f"  Avg moves/game        : {avg_moves:.1f}")
    print()
    print(f"  {name1:<22} : {p1_wins:>3} wins  ({100*p1_wins/total:.1f}%)")
    print(f"  {name2:<22} : {p2_wins:>3} wins  ({100*p2_wins/total:.1f}%)")
    print(f"  {'Draw':<22} : {draws:>3}       ({100*draws/total:.1f}%)")
    print()
    print(f"  {name1} as White : {p1_wins_as_white}/{len(p1_white)} wins  ({100*p1_wins_as_white/max(len(p1_white),1):.1f}%)")
    print(f"  {name1} as Black : {p1_wins_as_black}/{len(p2_white)} wins  ({100*p1_wins_as_black/max(len(p2_white),1):.1f}%)")
    print(f"  {name2} as White : {p2_wins_as_white}/{len(p2_white)} wins  ({100*p2_wins_as_white/max(len(p2_white),1):.1f}%)")
    print(f"  {name2} as Black : {p2_wins_as_black}/{len(p1_white)} wins  ({100*p2_wins_as_black/max(len(p1_white),1):.1f}%)")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Model fight — supports Model vs Model and Model vs Stockfish.\n\n'
                    'Model vs Model:\n'
                    '  python model_fight.py --p1 models/A.onnx --p2 models/B.onnx\n\n'
                    'Model vs Stockfish:\n'
                    '  python model_fight.py --p1 models/A.onnx --p2 stockfish --stockfish_path ./stockfish --elo 1500\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--p1",             type=str, required=True,
                        help="Path to player 1 ONNX model, or 'stockfish'")
    parser.add_argument("--p2",             type=str, required=True,
                        help="Path to player 2 ONNX model, or 'stockfish'")
    parser.add_argument("--p1_name",        type=str, default=None,
                        help="Display name for player 1 (default: filename or 'Stockfish')")
    parser.add_argument("--p2_name",        type=str, default=None,
                        help="Display name for player 2 (default: filename or 'Stockfish')")
    parser.add_argument("--stockfish_path", type=str, default="stockfish",
                        help="Path to the Stockfish binary (default: 'stockfish' if in PATH)")
    parser.add_argument("--elo",            type=int, default=None,
                        help="Stockfish ELO cap, 1320-3190 (only used if a player is 'stockfish')")
    parser.add_argument("--sf_depth", type=int, default=None,
                        help="Depth limit for Stockfish (None = no limit, relies on --elo and --time_limit)")
    parser.add_argument("--depth",          type=int, default=3,
                        help="Minimax depth for ONNX models")
    parser.add_argument("--time_limit",     type=float, default=0.1,
                        help="Time limit (s) per Stockfish move")
    parser.add_argument("--n_fens",         type=int, default=50)
    parser.add_argument("--n_workers",      type=int, default=4)
    parser.add_argument("--opening_book",   type=str, default="data/opening_book_fens.txt")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    # Default display names
    def default_name(path, elo):
        if path == "stockfish":
            return f"Stockfish (ELO {elo})" if elo is not None else "Stockfish"
        return path.split("/")[-1].replace(".onnx", "")

    name1 = args.p1_name or default_name(args.p1, args.elo)
    name2 = args.p2_name or default_name(args.p2, args.elo)

    random.seed(args.seed)

    # ── Load FENs ────────────────────────────────────────────────────────────
    with open(args.opening_book, 'r') as f:
        all_fens = [line.strip() for line in f if line.strip()]

    fens = random.sample(all_fens, min(args.n_fens, len(all_fens)))
    print(f"Loaded {len(fens)} FENs from {args.opening_book}")
    print(f"Matchup : {name1}  vs  {name2}")

    # ── Build jobs: 2 games per FEN (each player plays both colours) ──────────
    jobs = []
    for i, fen in enumerate(fens):
        # (fen, p1_path, p2_path, stockfish_path, elo, white_idx, game_id, depth, time_limit)
        jobs.append((fen, args.p1, args.p2, args.stockfish_path, args.elo, args.sf_depth, 1, f"{i}_p1white", args.depth, args.time_limit))
        jobs.append((fen, args.p1, args.p2, args.stockfish_path, args.elo, args.sf_depth, 2, f"{i}_p2white", args.depth, args.time_limit))

    print(f"Total games : {len(jobs)}  |  Workers : {args.n_workers}")

    # ── Run in parallel ───────────────────────────────────────────────────────
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(play_game, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is None:
                print(f"  [{i+1:>3}/{len(jobs)}] FAILED")
                continue
            results.append(result)
            print(f"  [{i+1:>3}/{len(jobs)}] game_id={result['game_id']:<16}"
                  f" white={result['white']:<10}"
                  f" result={result['result']:<6}"
                  f" winner={result['winner']:<10}"
                  f" moves={result['n_moves']}")

    print_stats(results, name1, name2)