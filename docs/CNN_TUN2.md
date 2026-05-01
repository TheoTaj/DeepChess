In this file, I detailled the hyperparameter tuning process for the convolutional neural network. 
We restart from zero because we realized that our previous dataset had a bad distribution thus the learned models were bad.

=> We start a grid search but we adapted it with what we learned from the previous grid search.

### Not tuned parameters:

* Optimzer: Adam
* Loss: MSE
* Scheduler: No scheduler
* Early stopping: Patience of 15 epochs
* Max epochs: 150
* Conv_kernels: [5, 3, 3]
* Batch size: 128
* K = 750
* Dataset: df_100k_50_750


Here is the grid :

| Hyperparameter | Values Tested |
| :--- | :--- |
| **Learning Rate (LR)** | `[0.0001, 0.00005]` |
| **Dropout** | `[0.05]` |
| **Conv Filters** | `[[32, 64], [32, 64, 128]]` |
| **FC Layers (MLP)** | `[[512, 256, 128], [1024, 512, 256]]` |

=> We have 8 models to train and to test. Here are the results:

| Rank | Model Name | LR | Conv Filters | FC Layers | epochs | n_params | Test Loss | Test sign acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | CNN_TUN2_2 | 0.00005 | [32,64,128] | [1024,512,256] | 150 | 9153249 | 0.081898 | 0.9205 |
| 2 | CNN_TUN2_6 | 0.00005 | [32,64] | [1024,512,256] | 143 | 4884833 | 0.082128 | 0.9174 |
| 3 | CNN_TUN2_1 | 0.0001 | [32,64,128] | [1024,512,256] | 98 | 9153249 | 0.08346 | 0.9205 |
| 4 | CNN_TUN2_3 | 0.0001 | [32,64,128] | [512,256,128] | 134 | 4466401 | 0.083507 | 0.9204 |
| 5 | CNN_TUN2_7 | 0.0001 | [32,64] | [512,256,128] | 136 | 2295137 | 0.08392 | 0.9181 |
| 6 | CNN_TUN2_4 | 0.00005 | [32,64,128] | [512,256,128] | 150 | 4466401 | 0.0851 | 0.9171 |
| 7 | CNN_TUN2_5 | 0.0001 | [32,64] | [1024,512,256] | 62 | 4884833 | 0.08523 | 0.9154 |
| 8 | CNN_TUN2_8 | 0.00005 | [32,64] | [512,256,128] | 78 | 2295137 | 0.090361 | 0.909 |

=> Based on these results, we can start the full training on the big dataset with the parameters of model CNN_TUN2_2. But with one exception; due to the fact that we'll have a ReduceLROnPlateau scheduler, we can start with a higher learning rate of 0.0001 because the scheduler will reduce it if needed.
Which means that we'll start the full training with the parameters of model CNN_TUN2_1. Yes this model is ranked 3rd but it has the same architecture as the best model just with a higher learning rate. And this model have 9M paramters to train which is higher that than the second best model.

### Final training:

For the scheduler, we don't want it to reduce the LR too early:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=1e-4,
            threshold_mode="rel",
        )
```

### Parameters:

* Scheduler patience: 5
* Scheduler factor: 0.5
* Overall patience: 15
* Max epochs: 200
* Batch size: 256

For the scheduler patience, we can set it to 5 epochs. The scheduler factor to 0.5. And the overall patience to 15. And the max epochs to 200.
For the batch size we double it to 256 because we have more data and we want to speed up the training. This helps also to stabilize the training.

### Results:

Come back for the results. 



