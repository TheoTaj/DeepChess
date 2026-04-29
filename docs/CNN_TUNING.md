# CNN Hyperparameter Tuning Summary - Phase 1

## 1. Overview
* **Dataset Size:** 100,000 rows (90,000 Train / 10,000 Test).
* **Goal:** Identify the most promising architecture and optimization settings before scaling to the full 5-million-row dataset.
* **Environment:** Trained on the Alan Cluster using PyTorch and tracked via Weights & Biases (WandB).

## 2. Methodology
We utilized a **Grid Search** approach to explore the interaction between learning rates, regularization, and model depth.
* **Optimizer:** Adam (Adaptive Moment Estimation).
* **Scheduler:** No scheduler was used in this phase but Adam's adaptive learning rates helps.
* **Epochs:** Maximum of 100 epochs, controlled by **Early Stopping** (patience = 10) to prevent overfitting and save computational resources.

## 3. First Grid Search Configuration
A total of **36 combinations** were tested based on the following axes:

| Hyperparameter | Values Tested |
| :--- | :--- |
| **Learning Rate (LR)** | `[0.005, 0.001, 0.0005]` |
| **Dropout** | `[0.2, 0.3, 0.4]` |
| **Conv Filters** | `[[20, 50], [32, 64, 128]]` |
| **FC Layers (MLP)** | `[[500], [512, 256]]` |

The following parameters were held constant to serve as a baseline:
* **Batch Size (128):** A standard balance between stochastic noise and GPU memory utilization.
* **Conv Kernels ([5, 3]):** Given the 8x8 dimensions of a chessboard, a 5x5 kernel followed by a 3x3 kernel provides an optimal receptive field to capture both global pawn structures and local tactical motifs without over-parameterizing the spatial part of the network.

**_NOTE:_**  In the CNN architecture, we didn't used pooling because the size of the input is already small. Thus the effective receptive field is linear.


## 4. Results: Top 4 Models

| Rank | Model Name | LR | Dropout | Conv Filters | FC Layers | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 29| 0.0005| 0.2| 32,64,128| 512,256| 0.14105|
| 2 | 27| 0.0005| 0.2| 20,50| 512,256| 0.14117|
| 3 | 32| 0.0005| 0.3| 32,64,128| 512,256| 0.14591|
| 4 | 30| 0.0005| 0.3| 20,50| 512,256| 0.14648|

## 5. Key Observations
1.  **Grid Edge Phenomenon:** The top-performing models are consistently at the boundaries of our search grid (specifically the lowest LR and the highest model complexity). This suggests that the "Global Optimum" likely lies further in these directions.
2.  **MLP Depth:** There is a significant performance boost when moving from a single hidden layer `[500]` to a dual-layer architecture `[512, 256]`. The deeper MLP allows the model to process the high-level features extracted by the CNN more effectively.
3.  **The "Slow Learner" (4th Model):** The 4th best model reached the 100-epoch limit without triggering Early Stopping. With a Dropout of 0.3, this model exhibits strong regularization potential. Its performance indicates it was still improving; given more time or a slightly lower learning rate, it could potentially outperform the current leader by finding a flatter, more generalizable local minimum.

## 6. Refined Grid Search (Phase 2)
The objective is to explore the "edges" of the Phase 1 results to find the true peak performance. This grid consists of **16 unique combinations**.

| Hyperparameter | Values Tested | Justification |
| :--- | :--- | :--- |
| **Learning Rate (LR)** | `[0.0005, 0.0001]` | We keep the previous winner ($0.0005$) and test a lower rate ($0.0001$) to see if a slower convergence leads to a better local minimum. |
| **Dropout** | `[0.15, 0.2]` | Since $0.2$ was the best and overfitting was minimal, we test a lower value ($0.15$) to allow the model to capture more complex features. |
| **Conv Filters** | `[[20, 50], [32, 64, 128]]` | The performance gap between these two was negligible (0.00012 in loss) so we keep them. |
| **FC Layers (MLP)** | `[[512, 256], [512, 256, 128]]` | Deeper MLP was better. We now test if adding a third layer (`128`) further helps in compressing tactical features into a single evaluation score. |

This makes 16 combinaitions to be tested. But two of them were already tested in the first grid search (models 29 and 27), so we will only need to test 14 new combinations in the second grid search.

## 7. Results of phase 2

| Rank | Model Name | LR | Dropout | Conv Filters | FC Layers | Epochs | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 46 | 0.0001 | 0.15 | 32, 64, 128 | 512, 256, 128 | 97 | 0.13803 |
| 2 | 44 | 0.0001 | 0.15 | 20, 50 | 512, 256, 128 | 100 | 0.14064 |
| 3 | 49 | 0.0001 | 0.2 | 32, 64, 128 | 512, 256, 128 | 100 | 0.14078 | 
| 4 | 29| 0.0005| 0.2| 32,64,128| 512,256| 73 | 0.14105|

=> for phase 3, we explore in the same direction.

## 8. Refined Grid Search (Phase 3)

| Hyperparameter | Values Tested | Justification |
| :--- | :--- | :--- |
| **Learning Rate (LR)** | `[0.0001, 0.00005]` | We keep the previous winner and keep exploring lower lr|
| **Dropout** | `[0.1, 0.15]` | Again, same logic |
| **Conv Filters** | `[[32, 64, 128]]` | Here we stop [20, 50] because it clearly seems to be less good|
| **FC Layers (MLP)** | `[[512, 256, 128], [1024, 512, 256]]` | Deeper MLP is still better, this time we try an MLP with more neurons but still 3 layers. |

This makes 8 combinaitions. One of them was already tested (model 46). => 7 mores tests.
! Now we see that runs start to reach 100 epochs, let's limit them to 200 and let's increase the patience to 15.

I push in github but i don't have the results yet. I will update this section once I have them.

## 9. Results of phase 3

| Rank | Model Name | LR | Dropout | Conv Filters | FC Layers | Epochs | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 52 | 0.0001 | 0.1  | 32,64,128 | 1024,512,256 | 122 | 0.13362 |
| 2 | 53 | 0.0001 | 0.15 | 32, 64, 128 | 1024, 512, 256 | 86 | 0.13642 |
| 3 | 54 | 0.00005 | 0.1 | 32, 64, 128 | 512, 256, 128 | 151 | 0.13656 |
| 4 | 46 | 0.0001 | 0.15 | 32, 64, 128 | 512, 256, 128 | 97 | 0.13803 |
 


=> Smaller LR don't win this time, so we can stop the tuning of the learning rate. For the dropout, smaller dropout keep being better so we'll test 0.05. For the MLP, wider or more parameters seems to be better. But we don't want to have a too big MLP because training might be too long on the full dataset. 
=> we keep the winning parameters and just with a dropout of 0.05.

Resultats keep being better with a smaller dropout so we try an dropout of 0 (while keeping the others fixed).

## 10 Final Results

| Rank | Model Name | LR | Dropout | Conv Filters | FC Layers | Epochs | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 58 | 0.0001 | 0.05 | 32,64,128 | 1024,512,256 | 108 | 0.13356 |
| 2 | 52 | 0.0001 | 0.1  | 32,64,128 | 1024,512,256 | 122 | 0.13362 |
| 3 | 59 | 0.0001 | 0.0  | 32,64,128 | 1024,512,256 | 165 | 0.13565 |
| 4 | 53 | 0.0001 | 0.15 | 32, 64, 128 | 1024, 512, 256 | 86 | 0.13642 |

=> We can now start the training on the full dataset with parameters of model 58.

## 11. Full training

**Overview**:
* **Dataset Size:** 5 million rows (4.5 million Train / 0.5 million Test).
* **Scheduler:** We use `ReduceLROnPlateau` to adjust the learning rate on plateau, with a patience of 4 epochs and a factor of 0.5.
* **Epochs:** We set a maximum of 100 epochs, with an Early Stopping patience of 10 epoch.

**Results**:

| Model Name | LR | Dropout | Conv Filters | FC Layers | Epochs | Learning Rate | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CNN_5M | 0.0001 | 0.05 | 32, 64, 128 | 1024, 512, 256 | 95 | 0.0000015625 | 0.075707 |


This model can be used for the final comparaison between the CNN and the ViT.

**Analysis**:

The Test Loss is very small. We can use the inverse of the tanh(cp/300) to get an estimate of the centipawn error. This gives us an estimated error of `22.75 centipawns`. 