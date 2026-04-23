# ViT Hyperparameter Tuning Summary

## 0. Overview

Globally, same principle as the CNN tuning:
- `train_ViT.py` is almost the same as `train_CNN.py` but just a few changes to adapt to the ViT.
- Training with different hyperparameter configurations on 90% of the small dataset, and testing on the 10%.

A few differences:
- Use of **AdamW** instead of Adam to implement weight decay, which is a well-known issue of transformers. Thus, `weight_decay`is now a hyperparameter too.
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
| **Learning Rate** | `[1e-5, 1e-3]` (log scale) | Standard range for Transformers. Searched on a log scale since the optimal LR typically spans orders of magnitude. ViTs generally prefer lower LRs than CNNs. |
| **Weight Decay** | `[1e-5, 1e-2]` (log scale) | Specific to AdamW, which applies weight decay correctly decoupled from the gradient update. Regularization strength is unknown a priori so a wide log-scale range is explored. |
| **Dropout** | `[0.0, 0.2]` (continuous) | CNN tuning showed that smaller dropout was consistently better, so we start with a range biased towards low values. Upper bound of 0.2 matches the CNN's best dropout. |
| **N Blocks** | `[2, 4, 6]` | Controls the depth of the Transformer. 2 is a minimal baseline, 6 is likely an upper bound for an 8x8 board where spatial complexity is limited. |
| **MLP Dim** | `[128, 256, 512]` | Controls the width of the feed-forward layer inside each Transformer block. Typically set as a multiple of `embed_dim`, so the range covers 1x to 4x of the largest embed_dim. |
| **Embed Dim** | `[64, 128, 256]` | Controls the size of the token embedding space. Larger values give more expressive representations but increase compute. Coupled with `n_heads` via divisibility constraint. |
| **N Heads** | `[2, 4, 8]` | Number of attention heads. Must divide `embed_dim`, so valid pairs are enforced in the code via `embed_n_heads_pair`. More heads allow the model to attend to different aspects of the position simultaneously. |

I ran 43 trials (on wandb, `ViT_TUNE_0` -> `ViT_TUNE_42`) in which Optuna will explore different configurations. 

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

> COMING SOON