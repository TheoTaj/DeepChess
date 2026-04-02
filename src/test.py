import os
import chess
from dotenv import load_dotenv
from datasets import load_dataset
import chess.svg
import webbrowser

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

dataset = load_dataset("mateuszgrzyb/lichess-stockfish-normalized", split="train", streaming=True, token=hf_token) # streaming=True to avoid loading the whole dataset in memory

testRow = dataset.take(1) # take the first n elements of the dataset
for e in testRow:
    print(e)
    
fen = e["fen"]
board = chess.Board(fen)
print(board) # print the board in text format in the console

# visualize the board in a web browser using chess.svg
svg_code = chess.svg.board(board, size=400)
with open("temp/temp.svg", "w") as f:
    f.write(svg_code)

webbrowser.open("file://" + os.path.realpath("temp/temp.svg"))