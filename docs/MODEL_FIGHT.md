>**_NOTE:_** The formula used for the ELO ratings or for the confidence interval are grom gemini and are approximations.

In the file, we detail the process of comparing our 4 final models by making each model play against each other. 

### Procedure:
- 50 differents FENs are sample from `data/Book.txt` and these samples are saved in `data/opening_book_fens.txt`. 
- We use the same 50 FENs for each confrontation.
- Each model plays 100 games against each other because from 1 FEN we can play 2 games (one with the model playing white and one with the model playing black).
- Search `DEPTH = 3` and is incremented by 1 when the material count is less than 20.

### `CNN_SYMMETRIC` vs `CNN_ASYMMETRIC`:

| Model  | Win Rate (%) |
| :--- | :---: |
| **CNN_SYM**   | 38.0% |
| **CNN_ASYM**  | 37.0% |
| **Draws**  | 25.0% |

### CNN_SYM (Symmetric)
*   **As White:** 19 / 50 wins (38.0%)
*   **As Black:** 19 / 50 wins (38.0%)

### CNN_ASYM (Asymmetric)
*   **As White:** 20 / 50 wins (40.0%)
*   **As Black:** 17 / 50 wins (34.0%)

### Discussion:

I don't see a significant difference between the two models. Should we conclude that both loss function seems to be equally good ? Or should we increase the number of games to have more significant results ?

### ELO ratings:
1. The ELO Formula

$$E = \frac{Wins + (0.5 \times Draws)}{Total\ Games}$$

2. The Elo Inversion Formula

$$\Delta R = -400 \log_{10} \left( \frac{1}{E} - 1 \right)$$

- **CNN_SYMMETRIC ELO**: 0.505 %
- **CNN_ASYMMETRIC ELO**: 0.495 %
- $\Delta R$ = 3.47 ELO points in favor of CNN_SYMMETRIC.

### Significant results:

In order to determine if the difference in performance is statistically significant, we use a 95% confidence interval. This is approximated by this formula:
$$Margin_{Elo} \approx \frac{700}{\sqrt{n}}$$

- n = 100 => 70 ELO points margin error

Thus our results are not significant because the difference in ELO points (3.47) is much smaller than the margin of error (70).

- n = 500 => 31 ELO points margin error.


### Revised procedure:

- Let's increase the number of FENs to 250 and play 500 games.
- Theses FENs are saved in `data/opening_book_fens_250.txt`.