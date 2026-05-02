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

