## Project Description
The objective of this project is to design and implement a deep learning-based chess engine that replaces traditional hand-crafted heuristic evaluations with a neural network.

The core of the system will be a position evaluation model, for which we intend to explore either a Convolutional Neural Network (CNN) or a Vision Transformer (ViT). These architectures are particularly suited for capturing the spatial relationships and long-range dependencies between pieces on the $8 \times 8$ board. 

To facilitate the training process, we will utilize a dataset where positions are labeled with centipawn scores (a centipawn is a unit of measure used by chess engines to evaluate a position, where 100 centipawns represent the strategic value of a single pawn). Since these scores exhibit extremely wide bounds, we will apply a normalization transformation using the hyperbolic tangent function:
$$ \text{target} = \tanh\left(\frac{x}{K}\right) $$
where $x$ is the raw centipawn score and $K$ is a scaling factor (e.g., $K=300$). This mapping compresses decisive material advantages while maintaining high resolution for balanced positions, effectively scaling the labels within a stable $[-1, 1]$ range for the optimizer.