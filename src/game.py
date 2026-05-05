import chess
import torch
import tkinter as tk
import threading
import time
from PIL import Image, ImageTk

from CNN import ChessCNN
from minimax import get_best_move

# ── Couleurs ─────────────────────────────────────────────────────────────────
LIGHT   = "#F0D9B5"
DARK    = "#B58863"
SELECT  = "#7FC97F"
LEGAL   = "#7EC8E3"

SQUARE_SIZE = 80


def load_model(model_path, conv_filters, fc_layers, device):
    from CNN import ChessCNN
    model = ChessCNN(
        conv_filters=conv_filters,
        conv_kernels=[5, 3, 3],
        fc_dim=fc_layers,
        dropout=0.0
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model




class ChessGame:
    def __init__(self, root=None,
             white_model=None, black_model=None,
             white_depth=3, black_depth=3,
             device=None,
             delay_ms=500,
             headless=False,
             starting_fen=None):

        self.root        = root
        self.board       = chess.Board(starting_fen) if starting_fen else chess.Board()
        self.white_model = white_model
        self.black_model = black_model
        self.white_depth = white_depth
        self.black_depth = black_depth
        self.device      = device or torch.device("cpu")
        self.delay_ms    = delay_ms
        self.headless    = headless
        self.selected_square = None
        self.legal_targets   = []
        self.game_over       = False
        self.tt              = {}
        self.result          = None
        self.move_history    = []

        if self.headless:
            pass
        else:
            assert root is not None, "root (tk.Tk()) est requis en mode non-headless"
            self.root.title("DeepChess")
            self.root.resizable(False, False)

            size = SQUARE_SIZE * 8
            self.canvas = tk.Canvas(root, width=size, height=size)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.on_click)

            self.piece_images = self.load_piece_images("assets/pieces", SQUARE_SIZE)

            self.status_var = tk.StringVar(value="Game started !")
            tk.Label(root, textvariable=self.status_var,
                    font=("Helvetica", 13)).pack(pady=4)

            self.info_var = tk.StringVar(value="")
            tk.Label(root, textvariable=self.info_var,
                    font=("Helvetica", 11), fg="gray").pack()

            self.time_var = tk.StringVar(value="")
            tk.Label(root, textvariable=self.time_var,
                    font=("Helvetica", 11), fg="gray").pack()

            btn = tk.Frame(root)
            btn.pack(pady=5)
            tk.Button(btn, text="Restart", command=self.restart).pack(side=tk.LEFT, padx=5)
            tk.Button(btn, text="Quit",    command=root.quit).pack(side=tk.LEFT, padx=5)

            self.render_board()
            self.root.after(self.delay_ms, self.game_loop)

    # ── Rendering ────────────────────────────────────────────────────────────

    def load_piece_images(self, folder, size):
        """
        Loads piece images from a folder.
        Expected filenames: wP.bmp, wN.bmp, ..., bK.bmp
        """
        mapping = {
            (chess.WHITE, chess.PAWN):   "wP",
            (chess.WHITE, chess.KNIGHT): "wN",
            (chess.WHITE, chess.BISHOP): "wB",
            (chess.WHITE, chess.ROOK):   "wR",
            (chess.WHITE, chess.QUEEN):  "wQ",
            (chess.WHITE, chess.KING):   "wK",
            (chess.BLACK, chess.PAWN):   "bP",
            (chess.BLACK, chess.KNIGHT): "bN",
            (chess.BLACK, chess.BISHOP): "bB",
            (chess.BLACK, chess.ROOK):   "bR",
            (chess.BLACK, chess.QUEEN):  "bQ",
            (chess.BLACK, chess.KING):   "bK",
        }
        images = {}
        for key, name in mapping.items():
            path = f"{folder}/{name}.bmp"
            img  = Image.open(path).resize((size, size), Image.LANCZOS)
            images[key] = ImageTk.PhotoImage(img)
        return images

    def square_to_xy(self, square):
        """Returns top-left (x, y) pixel of a square."""
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        return col * SQUARE_SIZE, row * SQUARE_SIZE

    def render_board(self):
        self.canvas.delete("all")
        for square in chess.SQUARES:
            col  = chess.square_file(square)
            row  = chess.square_rank(square)
            x, y = self.square_to_xy(square)

            # Couleur de la case
            if square == self.selected_square:
                color = SELECT
            elif square in self.legal_targets:
                color = LEGAL
            elif (col + row) % 2 == 0:
                color = DARK
            else:
                color = LIGHT

            self.canvas.create_rectangle(
                x, y, x + SQUARE_SIZE, y + SQUARE_SIZE,
                fill=color, outline=""
            )

            # Pièce
            piece = self.board.piece_at(square)
            if piece:
                img = self.piece_images.get((piece.color, piece.piece_type))
                if img:
                    self.canvas.create_image(
                        x + SQUARE_SIZE // 2,
                        y + SQUARE_SIZE // 2,
                        image=img
                    )

        # Coordonnées
        for i in range(8):
            self.canvas.create_text(
                4, i * SQUARE_SIZE + 10,
                text=str(8 - i), font=("Arial", 10), fill="gray"
            )
            self.canvas.create_text(
                i * SQUARE_SIZE + SQUARE_SIZE - 6,
                SQUARE_SIZE * 8 - 6,
                text=chess.FILE_NAMES[i], font=("Arial", 10), fill="gray"
            )

    # ── Game logic ────────────────────────────────────────────────────────────

    def current_model(self):
        return self.white_model if self.board.turn == chess.WHITE else self.black_model

    def current_depth(self):
        return self.white_depth if self.board.turn == chess.WHITE else self.black_depth

    def get_dynamic_depth(self):
        base_depth = self.current_depth()
        if self.get_material_count() < 20:
            return base_depth + 1
        return base_depth

    def get_material_count(self):
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        return sum(len(self.board.pieces(pt, chess.WHITE)) + len(self.board.pieces(pt, chess.BLACK)) for pt in values) 

    def is_human_turn(self):
        return self.current_model() is None

    def game_loop(self):
        if self.game_over:
            return
        if self.headless:
            # En headless : boucle synchrone jusqu'à la fin
            while not self.game_over:
                if self.is_human_turn():
                    break  # impossible en headless sans input
                self.ai_move_headless()
        else:
            if self.is_human_turn():
                player = "White" if self.board.turn == chess.WHITE else "Black"
                self.status_var.set(f"Your turn ({player}) — click a piece")
            else:
                self.status_var.set("AI is thinking...")
                self.root.update()
                threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move_headless(self):
        """Synchronous AI move for headless mode."""
        start = time.time()
        move, score = get_best_move(
            self.board.fen(), self.current_model(),
            self.device, depth=self.current_depth(),
            tt=self.tt
        )
        elapsed = time.time() - start
        self.move_history.append({
            'move':    move.uci() if move else None,
            'score':   score,
            'elapsed': elapsed,
            'player':  'white' if self.board.turn == chess.WHITE else 'black'
        })
        self.apply_move(move, score)

    def ai_move(self):
        start = time.time()
        current_depth = self.get_dynamic_depth()
        move, score = get_best_move(
            self.board.fen(), self.current_model(),
            self.device, depth=current_depth,
            tt=self.tt
        )
        elapsed = time.time() - start
        self.root.after(0, lambda: self.apply_move(move, score, elapsed))

    def apply_move(self, move, score=None, elapsed=None):
        if move is None or move not in self.board.legal_moves:
            if not self.headless:
                self.status_var.set("Illegal move.")
            return
        self.board.push(move)

        if not self.headless:
            score_str   = f"{score:+.4f}" if score is not None else ""
            elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else ""
            self.info_var.set(f"Last move: {move.uci()}  |  score: {score_str}")
            self.time_var.set(f"AI thinking time: {elapsed_str}")
            self.selected_square = None
            self.legal_targets   = []
            self.render_board()

        self.check_game_over()

        if not self.game_over and not self.headless:
            self.root.after(self.delay_ms, self.game_loop)

    def check_game_over(self):
        if self.board.is_checkmate():
            self.result    = "black" if self.board.turn == chess.WHITE else "white"
            self.game_over = True
            reason         = "Checkmate"
        elif self.board.is_stalemate():
            self.result    = "draw"
            self.game_over = True
            reason         = "Stalemate"
        elif self.board.is_insufficient_material():
            self.result    = "draw"
            self.game_over = True
            reason         = "Insufficient material"
        elif self.board.can_claim_draw():
            # Détermination de la raison précise
            if self.board.can_claim_threefold_repetition():
                reason = "Draw claimed: Threefold repetition"
            elif self.board.can_claim_fifty_moves():
                reason = "Draw claimed: Fifty-move rule"
            else:
                reason = "Draw claimed (repetition or 50 moves)"

            # Mise à jour de l'UI
            self.status_var.set(reason)
            
            if self.headless:
                print(f"  [RESULT] {reason}", flush=True)
                
            self.game_over = True
            reason         = "Draw claimed"
        else:
            return  # partie pas terminée

        if not self.headless:
            msg = f"{reason} ! {self.result.capitalize()} wins !" if self.result != "draw" \
                else f"{reason} ! Draw."
            self.status_var.set(msg)

    # ── Human click ──────────────────────────────────────────────────────────

    def on_click(self, event):
        if self.game_over or not self.is_human_turn():
            return

        col    = event.x // SQUARE_SIZE
        row    = 7 - (event.y // SQUARE_SIZE)
        square = chess.square(col, row)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_targets   = [m.to_square for m in self.board.legal_moves
                                        if m.from_square == square]
                self.render_board()
        else:
            move = chess.Move(self.selected_square, square)

            # Promotion automatique en dame
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if chess.square_rank(square) in (0, 7):
                    move = chess.Move(self.selected_square, square,
                                      promotion=chess.QUEEN)

            self.selected_square = None
            self.legal_targets   = []
            if move in self.board.legal_moves:
                self.apply_move(move)
            else:
                self.render_board()

    def restart(self):
        self.board           = chess.Board()
        self.game_over       = False
        self.selected_square = None
        self.legal_targets   = []
        self.status_var.set("Game restarted !")
        self.info_var.set("")
        self.render_board()
        self.root.after(self.delay_ms, self.game_loop)
        self.tt = {}

    def play_headless(self, max_moves=200):
        """
        Runs the game synchronously to completion.
        Returns result dict.
        """
        assert self.headless, "Use game_loop() for non-headless mode."
        for _ in range(max_moves):
            if self.game_over:
                break
            self.ai_move_headless()
        if not self.game_over:
            self.result = "draw"  # partie trop longue → nulle par défaut
        return {
            'result':       self.result,
            'n_moves':      len(self.move_history),
            'move_history': self.move_history,
        }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH   = "models/CNN_5M_5.pth"
    CONV_FILTERS = [64, 128, 256]
    FC_LAYERS    = [512, 256, 128]

    model_sym = load_model(MODEL_PATH, CONV_FILTERS, FC_LAYERS, device)

    MODEL_PATH  = "models/CNN_ASYM_3.pth"
    model_asym = load_model(MODEL_PATH, CONV_FILTERS, FC_LAYERS, device)

    root = tk.Tk()
    ChessGame(
        root,
        white_model=None,   # None = humain
        black_model=model_sym,
        white_depth=3,
        black_depth=3,
        device=device,
        delay_ms=500,
        starting_fen="4r3/4k3/8/8/8/3K4/8/8 w - - 0 1"
    )
    root.mainloop()