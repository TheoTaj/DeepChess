>**_NOTE:_** The formula used for the ELO ratings or for the confidence interval are grom gemini and are approximations.

In the file, we detail the process of comparing our 4 final models by making each model play against each other. 

### Procedure:
- 50 differents FENs are sample from `data/Book.txt` and these samples are saved in `data/opening_book_fens.txt`. 
- We use the same 50 FENs for each confrontation.
- Each model plays 100 games against each other because from 1 FEN we can play 2 games (one with the model playing white and one with the model playing black).
- Search `DEPTH = 3` and is incremented by 1 when the material count is less than 20.

## 1. CNN_SYM vs CNN_ASYM

| Model | As White | As Black | Total |
|---|---|---|---|
| CNN_SYM | 17 / 50  | 15 / 50  | 32 / 100|
| CNN_ASYM | 20 / 50  | 17 / 50  | 37 / 100|
| Draws | | | 31 / 100 |

## 2. ViT_SYM vs CNN_SYM

| Model | As White | As Black | Total |
|---|---|---|---|
| ViT_SYM | 8 / 50 | 9 / 50 | 17 / 100 |
| CNN_SYM | 34 / 50 | 26 / 50 | 60 / 100 |
| Draws |  |  | 23 / 100 |

## 3. ViT_SYM vs CNN_ASYM

| Model | As White | As Black | Total |
|---|---|---|---|
| ViT_SYM | 10 / 50 | 8 / 50 | 18 / 100 |
| CNN_ASYM | 32 / 50 | 35 / 50 | 67 / 100 |
| Draws |  |  | 15 / 100 |


# Matches against Stockfish

## 1. Stockfish vs CNN_SYM

| Model | As White | As Black | Total |
|---|---|---|---|
| Stockfish | 15 / 50  | 3 / 50  | 18 / 100|
| CNN_SYM | 18 / 50  | 10 / 50  | 28 / 100|
| Draws | | | 54 / 100 |

## 2. Stockfish vs CNN_ASYM

| Model | As White | As Black | Total |
|---|---|---|---|
| Stockfish | -- / 50  | -- / 50  | -- / 100|
| CNN_ASYM | -- / 50  | -- / 50  | -- / 100|
| Draws | | | -- / 100 |

## 3. Stockfish vs ViT_SYM

| Model | As White | As Black | Total |
|---|---|---|---|
| Stockfish | -- / 50  | -- / 50  | -- / 100|
| ViT_SYM | -- / 50  | -- / 50  | -- / 100|
| Draws | | | -- / 100 |

## 4. Stockfish vs ViT_ASYM

| Model | As White | As Black | Total |
|---|---|---|---|
| Stockfish | -- / 50  | -- / 50  | -- / 100|
| ViT_ASYM | -- / 50  | -- / 50  | -- / 100|
| Draws | | | -- / 100 |