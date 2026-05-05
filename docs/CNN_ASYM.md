We'll now train our CNN with the asymetric MSE loss. Reminder of the function:
$$L(y, \hat{y}) = \begin{cases} 
(y - \hat{y})^2 & \text{if } \text{sgn}(y) = \text{sgn}(\hat{y}) \\
\alpha (y - \hat{y})^2 & \text{if } \text{sgn}(y) \neq \text{sgn}(\hat{y})
\end{cases}$$

Where $\alpha$ is a hyperparameter that controls the asymmetry of the loss.

For the architecture of the CNN, we'll keep the same architecture as the best model of the CNN with the symmetric MSE loss. Not tuning the architecture of this CNN allows us to compare two same models only trained with differents losses.

The models will be compared by playing against each other. 

### Architecture:
* Conv filters: [64, 128, 256]
* FC layers: [512, 256, 128]
* Conv kernels: [5, 3, 3]
* Batch size: 512
* Optimizer: Adam
* Scheduler: ReduceLROnPlateau with patience of 5 epochs and factor of 0.5
* Early stopping: Patience of 15 epochs
* Max epochs: 200
* Learning rate: 0.00005

But how to choose the value of $\alpha$ ? Let's train 4 models with differents values of $\alpha$ and see which one has the best `test_sign_acc`. We'll test the following values: $\alpha \in \{1.25, 1.5, 1.75, 2\}$.
Larger values of $\alpha$ will destabilize the training.

### Results:

| Rank | Model Name | $\alpha$ |epochs | Test Loss | Test sign acc |final LR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | CNN_ASYM_3 | 1.75 | 200 | 0.069763 | 0.90952 | 0.0000015625 |
| 2 | CNN_ASYM_2 |  1.5 | 200 | 0.064947 | 0.90877 | 1.2207e-8 |
| 3 | CNN_ASYM_4 | 2 | 170 | 0.076592 | 0.90854 | 3.9063e-7 |
| 4 | CNN_ASYM_1 | 1.25 | 200 | 0.05959 | 0.90754 | 2.4414e-8 |

We keep CNN_ASYM_3 as the best model for the asymetric MSE loss. 
This model will be compared to the best CNN with the symetric MSE loss and by the two ViTs.