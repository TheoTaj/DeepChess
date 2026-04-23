# ViT Hyperparameter Tuning Summary

## 0. Overview

Globally, same principle as the CNN tuning:
- `train_ViT.py` is almost the same as `train_CNN.py` but just a few changes to adapt to the ViT.
- Training with different hyperparameter configurations on 90% of the small dataset, and testing on the 10%.

A few differences:
- Use of **AdamW** instead of Adam to implement weight decay, which is a well-known issue of transofmers. Thus, `weight_decay`is now a hyperparameter too.
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

I run 40 trial runs (on wandb, `ViT_TUNE_0` -> `ViT_TUNE_39`) in which Optuna will explore different configurations. 

## 2. Results of Phase 1

> COMING SOON

## 3. Optuna search - Phase 2

> COMING SOON