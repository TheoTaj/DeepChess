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

**Results**:

> COMING SOON