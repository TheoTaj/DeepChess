import chess
import torch
import torch.nn as nn
import numpy as np
from dataset import fen_to_tensor

EXACT = 0 # score is exact
LOWERBOUND = 1 # score is a lower bound (alpha cut-off)
UPPERBOUND = 2 # score is an upper bound (beta cut-off)


PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

def move_score(board, move):
    """
    Heuristic score for move ordering. Higher = more promising.
    Order: captures (MVV-LVA) > checks > quiet moves
    """
    score = 0

    # ── Captures : MVV-LVA ───────────────────────────────────────────────────
    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_val   = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        score += 10_000 + victim_val - attacker_val  # capturer une dame avec un pion = très bon

    # ── Promotion ────────────────────────────────────────────────────────────
    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 0)

    # ── Échec ────────────────────────────────────────────────────────────────
    board.push(move)
    if board.is_check():
        score += 5_000
    board.pop()

    return score


def ordered_moves(board):
    """Returns legal moves sorted by move_score descending."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: move_score(board, m), reverse=True)
    return moves


def evaluate(board, model, device):
    """
    Evaluates a board position using the CNN model.
    Returns a score in [-1, 1] from white's perspective.
    """
    tensor = fen_to_tensor(board.fen())
    x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item()
    return score

def quiescence(board, alpha, beta, maximizing_white, model, device, tt=None):
    """
    Quiescence search : continue searching captures and checks to avoid
    the horizon effect. Only stops when the position is 'quiet'.

    Args:
        board (chess.Board)    : Current board state.
        alpha (float)          : Alpha bound.
        beta (float)           : Beta bound.
        maximizing_white (bool): True if white is maximizing.
        model                  : CNN model.
        device                 : torch device.
        tt (dict)              : Transposition table.

    Returns:
        float: Score from white's perspective.
    """
    # ── Stand-pat : évaluation statique de la position ───────────────────────
    # Si la position est déjà bonne sans chercher plus loin, on peut couper
    stand_pat = evaluate(board, model, device)

    if maximizing_white:
        if stand_pat >= beta:
            return beta             # beta cut-off
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha            # alpha cut-off
        beta = min(beta, stand_pat)

    # ── Terminaison ──────────────────────────────────────────────────────────
    if board.is_checkmate():
        return -1.0 if maximizing_white else 1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0

    # ── Générer uniquement les coups "bruyants" (captures + échecs) ──────────
    noisy_moves = []
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            noisy_moves.append(move)

    # Position calme → on s'arrête
    if not noisy_moves:
        return stand_pat

    # Trier les coups bruyants avec MVV-LVA
    noisy_moves.sort(key=lambda m: move_score(board, m), reverse=True)

    # ── Récursion ─────────────────────────────────────────────────────────────
    if maximizing_white:
        best_score = stand_pat
        for move in noisy_moves:
            board.push(move)
            score = quiescence(board, alpha, beta, False, model, device, tt)
            board.pop()
            best_score = max(best_score, score)
            alpha      = max(alpha, score)
            if beta <= alpha:
                break
        return best_score

    else:
        best_score = stand_pat
        for move in noisy_moves:
            board.push(move)
            score = quiescence(board, alpha, beta, True, model, device, tt)
            board.pop()
            best_score = min(best_score, score)
            beta       = min(beta, score)
            if beta <= alpha:
                break
        return best_score


def minimax(board, depth, alpha, beta, maximizing_white, model, device, tt=None):
    """
    Minimax with alpha-beta pruning.
    Score is always from white's perspective:
        +1 = white wins, -1 = black wins.

    Args:
        board (chess.Board) : Current board state.
        depth (int)         : Remaining search depth.
        alpha (float)       : Alpha bound (best for white).
        beta (float)        : Beta bound (best for black).
        maximizing_white (bool): True if it's white's turn to maximize.
        model               : CNN model.
        device              : torch device.
        tt (dict)          : Transposition table for caching evaluations. {zobrist_hash: (depth, flag, score)}.

    Returns:
        float: Best score found.
    """

    alpha_orig = alpha
    key = board._transposition_key()

    if key in tt:
        tt_depth, tt_flag, tt_score = tt[key]
        if tt_depth >= depth:  # entrée valide seulement si profondeur >= actuelle
            if tt_flag == EXACT:
                return tt_score
            elif tt_flag == LOWERBOUND:
                alpha = max(alpha, tt_score)
            elif tt_flag == UPPERBOUND:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score  # cut-off via TT

    # ── Terminaison ──────────────────────────────────────────────────────────
    if board.is_checkmate():
        return -1.0 if maximizing_white else 1.0  # le joueur actuel est mat
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    if depth == 0:
        return quiescence(board, alpha, beta, maximizing_white, model, device, tt)

    # ── Récursion ────────────────────────────────────────────────────────────
    if maximizing_white:
        best_score = -float('inf')
        for move in ordered_moves(board):
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, False, model, device, tt)
            board.pop()
            best_score = max(best_score, score)
            alpha      = max(alpha, score)
            if beta <= alpha:
                break
    else:
        best_score = float('inf')
        for move in ordered_moves(board):
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, True, model, device, tt)
            board.pop()
            best_score = min(best_score, score)
            beta       = min(beta, score)
            if beta <= alpha:
                break

    if best_score <= alpha_orig:
        flag = UPPERBOUND  # on n'a pas amélioré alpha → upper bound
    elif best_score >= beta:
        flag = LOWERBOUND  # beta cut-off → lower bound
    else:
        flag = EXACT       # score exact
        
    tt[key] = (depth, flag, best_score)
    return best_score

def get_best_move(fen, model, device, depth=3, tt=None):
    if tt is None:
        tt = {}

    board            = chess.Board(fen)
    maximizing_white = board.turn == chess.WHITE
    best_move        = None
    best_score       = -float('inf') if maximizing_white else float('inf')
    alpha            = -float('inf')
    beta             = float('inf')

    for move in ordered_moves(board):
        board.push(move)
        score = minimax(
            board, depth - 1,
            alpha=alpha,
            beta=beta,
            maximizing_white=not maximizing_white,
            model=model,
            device=device,
            tt=tt,
        )
        board.pop()

        if maximizing_white and score > best_score:
            best_score = score
            best_move  = move
            alpha = max(alpha, score)
        elif not maximizing_white and score < best_score:
            best_score = score
            best_move  = move
            beta = min(beta, score)

    return best_move, best_score

if __name__ == "__main__":
    from CNN import ChessCNN

    MODEL_PATH = "models/CNN_5M_5.pth"
    FEN        = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    DEPTH      = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = ChessCNN(
        conv_filters=[64, 128, 256],
        conv_kernels=[5, 3, 3],
        fc_dim=[512, 256, 128],
        dropout=0.05
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Searching at depth {DEPTH}...")
    best_move, best_score = get_best_move(FEN, model, device, depth=DEPTH)
    print(f"Best move  : {best_move}")
    print(f"Best score : {best_score:+.4f}")