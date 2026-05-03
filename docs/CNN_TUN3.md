In this file, I detailled the hyperparameter tuning process for the convolutional neural network. 
We re-re-start from zero because we realized that our previous dataset had a bad distribution thus the learned models were bad.

=> We start a grid search but we adapted it with what we learned from the previous grid search.

### Not tuned parameters:

* Optimzer: Adam
* Loss: MSE
* Scheduler: No scheduler
* Early stopping: Patience of 15 epochs
* Max epochs: 150
* Conv_kernels: [5, 3, 3, 3]
* Batch size: 128
* K = 300
* Dataset: kaggle_100k_300

Here is the grid are the diffenret tested architectures:

- Conv Filters: [32, 64, 128] + FC Layers: [1024, 512, 256]
- Conv Filters: [64, 128, 256] + FC Layers: [512, 256, 128]
- Conv Filters: [32, 64, 128, 256] + FC Layers: [512, 256, 128]

Each architecture is tested with two learning rates: 0.0001 and 0.00005. The dropout is set to 0.05 for all the models. The convolutional kernels are set to [5, 3, 3, 3] for all the models.

 Which results in 6 models to train and to test.

### Results:

| Rank | Model Name | LR | Conv Filters | FC Layers | epochs | n_params | Test Loss | Test sign acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | CNN_TUN3_1 | 0.0001 | [32,64,128] | [1024,512,256] | 22 | 9153249 | 0.19077 | 0.7764 |
| 2 | CNN_TUN3_2 | 0.00005 | [32,64,128] | [1024,512,256] | 26 | 9153249 | 0.19463 | 0.7757 |
| 3 | CNN_TUN3_5 | 0.0001 | [32,64,128,256] | [512,256,128] | 21 | 8956385 | 0.19991 | 0.7700 |
| 4 | CNN_TUN3_3 | 0.0001 | [64,128,256] | [512,256,128] | 21 | 8952257 | 0.20088 | 0.7686 |
| 5 | CNN_TUN3_4 | 0.00005 | [64,128,256] | [512,256,128] | 20 | 8952257 | 0.20107 | 0.7680 |
| 6 | CNN_TUN3_6 | 0.00005 | [32,64,128,256] | [512,256,128] | 22 | 8956385 | 0.20316 | 0.7667 |

Results are not good, the learning stops too early. We need to tune on more data.
J'en ai marre de ce projet, on va faire comme julien a dit. On lance plusieurs tranings sur le dataset final tant pis pour le data leakage. Toute façon la comparaison des modèles en les faisant jouer l'un contre l'autre est ce qui nous intéresse.

### Full training:

### Parameters:

* Scheduler patience: 5
* Scheduler factor: 0.5
* Overall patience: 15
* Max epochs: 200
* Batch size: 512

We also decided to start with a fixed learning rate of 0.00005 for a slower but better learning.

=> Only 3 models on 5M rows

### Results:

Voici les résultats transcrits :

| Rank | Model Name | LR | Conv Filters | FC Layers | epochs | n_params | Test Loss | Test sign acc |
|------|------------|-----|--------------|-----------|--------|----------|-----------|---------------|
| 1 | CNN_5M_5 | 0.00005 | [64,128,256] | [512,256,128] | 200 | 8952257 | 0.052554 | 0.90797 |
| 2 | CNN_5M_4 | 0.00005 | [32,64,128,256] | [512,256,128] | 200 | 8956385 | 0.05347 | 0.90714 |
| 3 | CNN_5M_3 | 0.00005 | [32,64,128] | [1024,512,256] | 197 | 9153249 | 0.053742 | 0.90648 |


