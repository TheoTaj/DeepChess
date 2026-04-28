# ViT Hyperparameter Tuning Summary

## 0. Overview

Globally, same principle as the CNN tuning:
- `train_ViT.py` is almost the same as `train_CNN.py` but just a few changes to adapt to the ViT.
- Training with different hyperparameter configurations on 90% of the small dataset, and testing on the 10%.

A few differences:
- Use of **AdamW** instead of Adam to implement weight decay, which is a well-known issue of transformers. Thus, `weight_decay` is now a hyperparameter too.
- Use of the scheduler everywhere from the start and introduction of a warmup scheduler for the first few epochs (5 by default). The warmup scheduler helps in the beginning because attention weights are highly sensitive to large gradient updates early in training, so starting with a reduced LR and ramping up prevents the attention mechanism from being pushed into a bad region before it has seen enough data.
- Instead of a manual grid search, I use `optuna`, which handles the hyperparameter search entirely. By doing that, the search is more efficient than a simple grid search.

**Concerning Optuna**:

`[Claude]`:
Optuna is a hyperparameter optimization framework that uses a Tree-structured Parzen Estimator (TPE) algorithm instead of exhaustive grid search. Rather than trying all combinations, it builds a probabilistic model of which regions of the hyperparameter space tend to yield good results, based on trials completed so far. Early trials are essentially random exploration; after roughly 20 trials, the sampler starts exploiting what it has learned to propose more promising configurations. In practice this means fewer wasted trials compared to grid search. The study is persisted in a SQLite database (vit_tuning.db), which allows 4 parallel jobs on the cluster to share the same study — each job picks up the next available trial from the database, trains it, reports the result, and picks up the next one, without overlap.

## 1. Optuna search - Phase 1

In this Phase 1, I do a first search in order to already detect good hyperparameters. The search will be refined in Phase 2.

In `tune_ViT.py`, I first specify fixed parameters that won't be explored:

| Hyperparameter | Value | Justification |
| :--- | :--- | :--- |
| **Epochs** | `100` | Same as CNN tuning. |
| **Warmup Epochs** | `5` | 5% of total epochs, which is a standard warmup ratio. Short enough to not waste training budget, long enough to stabilize attention weights before the main LR takes over. |
| **Alpha** | `1.0` | Standard MSE loss for now|
| **Batch Size** | `128` | Same as CNN tuning. |
| **Patience** | `10` | With a warmup + plateau scheduler, convergence is slower and noisier early on, so a higher patience avoids premature stopping. |
| **Scheduler Factor** | `0.5` | Standard halving of the LR on plateau. Aggressive enough to escape plateaus, conservative enough to not overshoot. |
| **Scheduler Patience** | `4` | Slightly shorter than the early stopping patience, so the LR has a chance to reduce and recover before early stopping triggers. |


and then I specify how to search for the hyperparameters that need tuning:

| Hyperparameter | Search Space | Justification |
| :--- | :--- | :--- |
| **Learning Rate** | `[1e-5, 1e-3]` (log scale) | Standard range for Transformers. Searched on a log scale since the optimal LR typically spans orders of magnitude. |
| **Weight Decay** | `[1e-5, 1e-2]` (log scale) | Specific to AdamW, which applies weight decay correctly decoupled from the gradient update. Regularization strength is unknown a priori so a wide log-scale range is explored. |
| **Dropout** | `[0.0, 0.2]` (continuous) | CNN tuning showed that smaller dropout was consistently better, so we start with a range biased towards low values. Upper bound of 0.2 matches the CNN's best dropout. |
| **N Blocks** | `[2, 4, 6]` | Controls the depth of the Transformer. 2 is a minimal baseline, 6 is likely an upper bound for an 8x8 board where spatial complexity is limited. |
| **MLP Dim** | `[128, 256, 512]` | Controls the width of the feed-forward layer inside each Transformer block. Typically set as a multiple of `embed_dim`, so the range covers 1x to 4x of the largest embed_dim. |
| **Embed Dim** | `[64, 128, 256]` | Controls the size of the token embedding space. Larger values give more expressive representations but increase compute. Coupled with `n_heads` via divisibility constraint. |
| **N Heads** | `[2, 4, 8]` | Number of attention heads. Must divide `embed_dim`, so valid pairs are enforced in the code via `embed_n_heads_pair`. More heads allow the model to attend to different aspects of the position simultaneously. |

I ran 43 trials (on wandb, `ViT_TUNE_0` -> `ViT_TUNE_42`) in which Optuna explored different configurations. 

## 2. Results of Phase 1

Top configurations across the 43 trials:

| Rank | Trial | Loss | lr | weight_decay | dropout | n_blocks | embed_dim, n_heads | mlp_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 36 | 0.16149 | 9.97e-4 | 4.34e-3 | 0.045 | 4 | 128, 8 | 512 |
| 2 | 39 | 0.16172 | 9.97e-4 | 4.92e-3 | 0.124 | 2 | 256, 8 | 128 |
| 3 | 34 | 0.16356 | 9.97e-4 | 4.52e-4 | 0.124 | 2 | 128, 8 | 128 |
| 4 | 38 | 0.16555 | 9.49e-4 | 1.73e-4 | 0.127 | 2 | 256, 8 | 128 |
| 5 | 23 | 0.16593 | 4.93e-4 | 2.11e-3 | 0.072 | 4 | 64, 2 | 512 |
| 6 | 42 | 0.16660 | 9.77e-4 | 4.42e-3 | 0.127 | 2 | 256, 8 | 128 |

### Analysis of the results:
- `lr`: initially constrained between 1e-5 and 1e-3, the best models are close to 1e-3 so we should consider larger learning rates: [5e-4, 1e-2]
- `weight_decay`: we can narrow down the search to [1e-4, 1e-2]
- `dropout`: narrow down to [0.0, 0.15]
- `n_blocks`: no need to keep 6: [2, 4]
- `embed_dim, n_heads`: larger attention heads seem to be important so only keep the two pairs: [128, 8] and [256, 8]
- `mlp_dim`: keep all three initial values since it doesn't seem to have a big impact.

## 3. Optuna search - Phase 2

Here I apply the same principle as for Phase 1 but with restrained ranges for the hyperparameters and I again do 40 runs.

For the fixed parameters, we change in the following way (the ones not mentioned in this table were not changed):
| Hyperparameter | Phase 1 | Phase 2 | Justification |
| :--- | :--- | :--- | :--- |
| **Epochs** | `100` | `150` | Top Phase 1 models used high LRs near the boundary. With an even wider LR range in Phase 2, the scheduler will need more reduction steps to converge, requiring more epochs. |
| **Patience** | `10` | `15` | Increased to avoid premature early stopping. With higher LRs and more epochs, the model needs more time to recover after LR reductions. |
| **Scheduler Patience** | `4` | `6` | Increased because with higher LRs in Phase 2 the loss curve will be noisier, and a patience of 4 might trigger LR reductions too early on temporary fluctuations. |

For the hyperparameters to tune, we now use the following ranges:
| Hyperparameter | Value | Justification |
| :--- | :--- | :--- |
| **Learning Rate** | `[5e-4, 1e-2]` (log scale) | Phase 1 top models clustered near the upper boundary `1e-3`, suggesting the optimum lies higher. The range is shifted upward. |
| **Weight Decay** | `[1e-4, 1e-2]` (log scale) | Phase 1 best models had weight decay between `1e-4` and `5e-3`. The lower bound is raised from `1e-5` to `1e-4` to focus the search on the promising region. |
| **Dropout** | `[0.0, 0.15]` (continuous) | Phase 1 best model used dropout of `0.045` and most top models were below `0.13`. The upper bound is reduced from `0.2` to `0.15` to focus on the low-dropout region. |
| **N Blocks** | `[2, 4]` | `n_blocks=6` never appeared in the top results. Both `2` and `4` appear among the best models so both are kept. |
| **Embed Dim, N Heads** | `["128,8", "256,8"]` | `n_heads=8` appears in 5 of the top 6 Phase 1 models. `embed_dim=64` is dropped as it only appeared once in the top 6 and with a weaker result. The constraint that `embed_dim` must be divisible by `n_heads` is enforced via the pair encoding. |
| **MLP Dim** | `[128, 256, 512]` | No clear pattern emerged from Phase 1 — all three values appear among the top models. The full range is kept for Phase 2. |


## 4. Results of Phase 2

| Rank | Trial | Loss | lr | weight_decay | dropout | n_blocks | embed_dim, n_heads | mlp_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 82 | 0.15224 | 1.42e-3 | 5.38e-3 | 0.103 | 2 | 128, 8 | 128 |
| 2 | 69 | 0.15414 | 2.17e-3 | 1.22e-4 | 0.054 | 4 | 128, 8 | 256 |
| 3 | 59 | 0.15444 | 5.34e-3 | 6.09e-4 | 0.077 | 4 | 128, 8 | 512 |
| 4 | 75 | 0.15717 | 1.77e-3 | 3.06e-4 | 0.096 | 4 | 128, 8 | 128 |
| 5 | 46 | 0.16058 | 1.54e-3 | 7.97e-3 | 0.110 | 2 | 128, 8 | 128 |
| 6 | 53 | 0.16150 | 3.13e-3 | 1.90e-4 | 0.044 | 4 | 128, 8 | 256 |

### Analysis of the results:
- `lr`: The best model uses `1.42e-3`, toward the lower end of the Phase 2 range `[5e-4, 1e-2]`. The optimum seems to sit around `1e-3` to `3e-3` rather than higher as initially suspected. We narrow the range to `[8e-4, 4e-3]` for Phase 3.
- `weight_decay`: the weight decay values are quite spread out, we keep the same range.
- `dropout`: The best model uses `0.103` and all top models are below `0.13`. We slightly narrow the range to `[0.0, 0.12]`.
- `n_blocks`: Both `2` and `4` appear in the top 4 with no clear winner. We keep both for Phase 3.
- `embed_dim, n_heads`: `128, 8` appears in all 10 top models — `256, 8` completely disappeared from the top results. We fix this to `128, 8` for Phase 3.
- `mlp_dim`: `128` and `256` dominate the top results. We drop `512` for Phase 3.

## 5. Optuna search - Phase 3

For this phase I keep the same fixed parameters as for phase 2. Here are the changes I make for the hyperparameters to tune:

| Hyperparameter | Value | Justification |
| :--- | :--- | :--- |
| **Learning Rate** | `[8e-4, 4e-3]` (log scale) | Narrow down a bit the interval compared to phase 2. |
| **Weight Decay** | `[1e-4, 1e-2]` (log scale) | Same range as phase 2 since no pattern was identified. |
| **Dropout** | `[0.0, 0.12]` (continuous) | Narrow down a bit the interval compared to phase 2. |
| **N Blocks** | `[2, 4]` | Both values appear in top models so no reason to leave one out. |
| **Embed Dim, N Heads** | `"128,8"` | Completely dominates so we only keep these values. |
| **MLP Dim** | `[128, 256]` | Drop 512 compared to phase 2 |

## 6. Results of Phase 3

| Rank | Trial | Loss | lr | weight_decay | dropout | n_blocks | embed_dim, n_heads | mlp_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 108 | 0.15221 | 1.95e-3 | 9.48e-4 | 0.065 | 2 | 128, 8 | 256 |
| 2 | 104 | 0.15516 | 1.90e-3 | 6.52e-4 | 0.064 | 2 | 128, 8 | 256 |
| 3 | 121 | 0.15653 | 1.42e-3 | 3.90e-4 | 0.051 | 2 | 128, 8 | 128 |
| 4 | 106 | 0.15908 | 2.24e-3 | 3.56e-3 | 0.064 | 2 | 128, 8 | 256 |
| 5 | 93 | 0.15946 | 1.07e-3 | 2.33e-4 | 0.037 | 2 | 128, 8 | 256 |
| 6 | 102 | 0.15960 | 2.43e-3 | 5.67e-4 | 0.055 | 2 | 128, 8 | 256 |

### Analysis of the results: 
The improvement in the loss between phase 2 and phase 3 is negligible (0.15221 and 0.15224). This suggests that the search has converged and that further tuning on the small dataset will not yield meaningful gains.

- `lr`: the top models all have a `lr` close to 2e-3. For the final training, we'll keep the value of the best model: 1.95e-3. Anyway, the scheduler will change the `lr` during training so we just need an initial guess.
- `weight_decay`: no clear pattern emerged across phases, so we simply keep the value of the best model: 9.48e-4
- `dropout`: keep the value of the best model: 0.065
- `n_blocks`: value of 2 dominates entirely. We'll keep that value for the final training.
- `embed_dim, n_heads`: these parameters were already fixed at the end of phase 2.
- `mlp_dim`: value of 256 dominates. We'll keep that value for the final training.

## 7. Final training

**Final configuration:**
```bash
"epochs": 100
"warmup_epochs": 5
"alpha": 1.0
"batch_size": 128
"patience": 10
"scheduler_factor": 0.5
"scheduler_patience": 4
"dataset_path": "data/dataset_5000000.parquet"

"lr": 1.95e-3
"weight_decay": 9.48e-4
"dropout": 0.065
"embed_dim": 128
"n_blocks": 2
"n_heads": 8
"mlp_dim": 256
```

I run this configuration once with `train_ViT.py` (Optuna not needed here).

## 8. Final results

| Model name | Loss | lr | weight_decay | dropout | n_blocks | embed_dim, n_heads | mlp_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ViT_5M | 0.10521 | 1.95e-3 | 9.48e-4 | 0.065 | 2 | 128, 8 | 256 |

The final loss of the ViT (0.10521) is larger than for the CNN (0.075707). Several factors likely explain this gap:

1. **Inductive bias mismatch:** CNNs are inherently biased toward local spatial structure — their convolutional filters naturally capture piece interactions within a neighborhood, which maps well onto the 8×8 chessboard. ViTs, by contrast, have no such prior: they must *learn* spatial relationships from scratch through attention, which requires significantly more data and depth to compensate.

2. **Model depth:** The final ViT configuration uses only 2 transformer blocks, which was selected during tuning on 100K samples. This is a very shallow attention stack — with only 2 rounds of self-attention, the model has limited ability to compose complex positional relationships across the board. Deeper ViTs (4-6 blocks) are typically needed before the attention mechanism becomes competitive, but those were implicitly penalized during tuning because they need more data to shine.

3. **Tuning dataset size:** The hyperparameter search was conducted on 100K samples for both models, but this disproportionately disadvantages the ViT. CNNs generalize well from small data due to their inductive biases, while ViTs are known to require substantially larger datasets before their attention mechanism becomes effective. As a result, the tuning process systematically selected a ViT configuration optimized for small-data behavior, which may not be the optimal configuration at the 5M sample scale.

4. **Learning rate schedule:** The ViT's learning rate decayed less aggressively than the CNN's over training (finishing around 6e-5 vs ~1.5e-6 for the CNN), suggesting the ViT may not have fully converged into a sharp minimum by the end of training.

While these results are accepted as the definitive outcome of the tuning protocol, two additional configurations with slightly different were tested on the full 5M dataset out of curiosity, to assess whether the shallow depth imposed by the small-data tuning was the main limiting factor for the ViT (see section 9).

## 9. Additional tests

As mentioned previously, I think the results in section 8 can be accepted as the final results of the ViT training because they come from a scientific and well-developped fine-tuning protocol. However, I still wanted to test out two configurations to see if the ViT could actually perform better than the CNN. 

So I did two more runs with the same config as in section 7, except for the following parameters that I changed:

| Hyperparameter | Old (Section 7: Final Training) | New | Justification |
| :--- | :--- | :--- | :--- |
| `epochs` | 100 | 150 | Deeper models need more epochs to converge; the original run exhausted all 100 epochs without early stopping, suggesting the model had not fully converged. |
| `warmup_epochs` | 5 | 10 | With `n_blocks=4` and `6`, the attention mechanism is more complex and more sensitive to large gradient updates early in training, requiring a longer stabilization phase. |
| `patience` | 10 | 15 | With more epochs and a higher LR schedule, the loss curve is noisier for longer; a higher patience avoids premature early stopping before the model has had time to recover from LR reductions. |
| `scheduler_patience` | 4 | 6 | Consistent with the increased early stopping patience — the LR scheduler should wait longer before reducing on temporary fluctuations, especially with deeper and noisier models. |
| `n_blocks` | 2 | 4 and 6 | The depth of 2 was selected during tuning on 100K samples, which may have penalized deeper models that require more data to outperform shallower ones. |

## 10. Results of the two additional tests

| Model name | Loss | lr | weight_decay | dropout | n_blocks | embed_dim, n_heads | mlp_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ViT_5M_Test1 | `0.08301` | 1.95e-3 | 9.48e-4 | 0.065 | 4 | 128, 8 | 256 |
| ViT_5M_Test2 | `0.07645` | 1.95e-3 | 9.48e-4 | 0.065 | 6 | 128, 8 | 256 |

`Note:` ViT_5M_Test1 stopped after 142 epochs because no improvement was made after 15 consecutive epochs.

We see that now the final losses are much closer to the results of the CNN, which goes toward the fact that the ViT could actually perform better than the CNN with its own training/tuning protocol. We can also see on wandb that continuing the training for `ViT_5M_Test2` could further increase performance.