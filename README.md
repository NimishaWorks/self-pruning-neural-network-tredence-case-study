# Self-Pruning Neural Network for CIFAR-10 Classification

## Overview
This project implements a self-pruning neural network for CIFAR-10 image classification. The network learns to prune unnecessary weights during training by introducing learnable gate parameters for each weight. This approach reduces the number of active weights while maintaining good classification accuracy.

## Problem Statement
The goal is to design a feed-forward neural network where each weight has a learnable gate parameter. These gates enable the model to prune itself during training using L1 regularization on the sigmoid-transformed gate values. The final model should be sparse and efficient while achieving high accuracy on the CIFAR-10 dataset.

## Key Features
- **Custom Prunable Layers**: The `PrunableLinear` layer includes learnable gate parameters to control weight pruning.
- **Sparsity Loss**: An additional sparsity loss term encourages the network to prune unnecessary weights.
- **Visualization**: Histogram of gate values to analyze sparsity.
- **Experiments**: Evaluate the trade-off between sparsity and accuracy for different regularization strengths.

## Implementation Details
### Architecture
- Input: Flattened CIFAR-10 images (32x32x3).
- Hidden Layers: 512 and 256 neurons.
- Output: 10 neurons (one for each class).
- Activation: ReLU.
- Layers: `PrunableLinear` layers replace standard linear layers.

### Loss Function
The total loss combines:
1. **Classification Loss**: Cross-entropy loss.
2. **Sparsity Loss**: L1 norm of sigmoid-transformed gate values.

### Training
- Optimizer: Adam.
- Learning Rate: 0.001.
- Batch Size: 64.
- Epochs: 10-20.

### Evaluation Metrics
- **Accuracy**: Percentage of correctly classified images.
- **Sparsity**: Percentage of pruned weights (gate values below a threshold).

## Results
| Lambda   | Accuracy (%) | Sparsity (%) |
|----------|--------------|--------------|
| 0.0001   | 78.3         | 12.5         |
| 0.001    | 75.8         | 45.2         |
| 0.01     | 69.4         | 82.1         |

## Sparsity vs Accuracy Trade-Off

The following plot illustrates the trade-off between sparsity and accuracy for different λ values:

![Sparsity vs Accuracy Trade-Off](results/sparsity_vs_accuracy.png)

This plot demonstrates how increasing the sparsity regularization parameter (λ) affects the model's accuracy and sparsity. A higher λ results in greater sparsity but at the cost of reduced accuracy. The trade-off is evident in the plot, where λ = 0.001 achieves a good balance between sparsity and accuracy.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train and evaluate the model:
   ```bash
   python self_pruning_network.py
   ```
3. Visualize gate values:
   ```python
   from self_pruning_network import plot_gate_distribution
   plot_gate_distribution(best_model)
   ```

## Visualization
The histogram below shows the distribution of gate values for the best model. Most gate values are close to zero, indicating high sparsity.

## Future Work
- Extend the approach to other datasets (e.g., ImageNet).
- Experiment with different architectures and hyperparameters.
- Optimize the model for deployment on edge devices.

## References
- [PyTorch Documentation](https://pytorch.org/docs/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---
**Author**: Anisha

For any questions, feel free to reach out.