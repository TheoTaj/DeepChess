# Small tuning phase

The three runs I had launched on the large balanced dataset did not seem to have good results, so I redo a small tuning on the small dataset and I try to increase the number of parameters to roughly match the size of the CNN.

**Fixed parameters**:
```bash
"epochs": 150,
"warmup_epochs": 5,
"alpha": 1.0,
"batch_size": 512,
"patience": 15,
"scheduler_factor": 0.5,
"scheduler_patience": 6,
"dataset_path": "data/kaggle_100k_300.parquet",

"lr" = 1e-4
"weight_decay" = 0.000948
"dropout" = 0.065
"n_heads" = 8
```

**Tuned parameters**:
```bash
"n_blocks" = [2, 4, 6, 8]
"embed_dim" = [128, 256, 384]
"mlp_dim" = 2 * embed_dim
```

**Results** (10 top models):

| Trial | Loss    | n_blocks | embed_dim |
|------:|---------|----------|-----------|
| 17    | 0.15891 | 6        | 256       |
| 14    | 0.15911 | 6        | 256       |
| 12    | 0.15914 | 6        | 256       |
| 19    | 0.15971 | 6        | 256       |
| 20    | 0.15981 | 6        | 256       |
| 13    | 0.16028 | 6        | 256       |
| 7     | 0.16061 | 8        | 384       |
| 22    | 0.16080 | 6        | 256       |
| 10    | 0.16086 | 8        | 128       |
| 9     | 0.16090 | 8        | 128       |

The loss across all the best trials is quite similar so there is no configuration that is really better in this case. 

# Training 

I'll train three models with these parameters:
```bash
"epochs": 200,
"warmup_epochs": 10,
"alpha": 1.0,
"batch_size": 512,
"patience": 15,
"scheduler_factor": 0.5,
"scheduler_patience": 6,
"dataset_path": "data/kaggle_5M_300.parquet",

"lr" = 1e-4
"weight_decay" = 0.000948
"dropout" = 0.065
"n_heads" = 8
```
| Model | n_blocks | embed_dim | mlp_dim | n_params |
|------:|----------|-----------|---------|----------|
| ViT1  | 6        | 256       | 512     | 3185153  |
| ViT2  | 8        | 128       | 256     | 1071105  |
| ViT3  | 8        | 384       | 768     | 9504769  |

**Results**:

| Model | epochs | test_loss | test_sign_acc | n_params |
|------:|----------|-----------|---------|----------|
| ViT1  | 124        | 0.0690       | 0.89     | 3185153  |
| ViT2  | 123        | 0.0846       | 0.88     | 1071105  |
| ViT3  | 99 | 0.0613       | 0.90     | 9504769  |

Even if ViT3 is not done running, it is already the best. Thus we'll keep this model as the best model for the ViT architecture.

# Asymmetric loss

For the asymmetric loss, I'll run three models with alpha=[1.25,1.5,1.75] (j'ai que 3 jobs de libre vu que ViT3 tourne toujours) with the same parameters as ViT3.

**Results**:

> COMING SOON