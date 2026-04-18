# Self-Pruning Neural Network for Efficient Model Compression

## Overview

Modern neural networks often contain a large number of parameters, many of which contribute minimally to model performance. These redundant parameters increase computational cost, memory usage, and inference latency, making deployment challenging in real-world applications.

This project implements a **self-pruning neural network** that automatically identifies and removes unnecessary weights during training. Instead of applying pruning after training, the model integrates a learnable gating mechanism that enables it to dynamically adapt its architecture while learning.

Each weight in the network is associated with a learnable gate parameter that determines whether the connection remains active or is suppressed. This allows the model to retain only the most important parameters, resulting in a sparse and computationally efficient neural network.

The model is trained and evaluated on the CIFAR-10 dataset using PyTorch, demonstrating how sparsity-aware learning can reduce model complexity while maintaining competitive classification performance.

---

## Methodology

### Learnable Gating Mechanism

Each weight in the neural network is paired with a learnable gate score. These gate scores are passed through a sigmoid activation function to ensure that gate values lie between 0 and 1.

Gate values close to 1 allow the weight to contribute fully, while gate values close to 0 effectively deactivate the connection.

During the forward pass, the effective weight is computed as:

effective weight = weight × sigmoid(gate_score)

This formulation ensures differentiability, allowing gradients to flow through both the weights and gate parameters during backpropagation.

---

### Loss Function

To encourage sparsity, an L1 regularization penalty is applied to the gate values.

The total loss function is defined as:

Total Loss = Classification Loss + λ × Sparsity Loss

Where:

Classification Loss ensures predictive accuracy.

Sparsity Loss penalizes active gate values, encouraging the network to reduce the number of active parameters.

λ is a hyperparameter that controls the strength of sparsity regularization.

A higher value of λ results in stronger pruning, while a lower value preserves more connections.

L1 regularization promotes sparsity because it encourages parameters to become exactly zero rather than just small in magnitude.

---

## Experimental Setup

Dataset: CIFAR-10
Framework: PyTorch

Network Architecture:

Input layer: 3072 features (32 × 32 × 3 image flattened)
Hidden layer 1: 512 neurons
Hidden layer 2: 256 neurons
Output layer: 10 neurons

Training Configuration:

Optimizer: Adam
Learning Rate: 0.001
Batch Size: 64
Epochs: 10

Three different values of λ were evaluated to study the trade-off between sparsity and classification accuracy.

---

## Results

### Sparsity vs Accuracy Trade-off

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
| ---------- | ----------------- | ------------------ |
| 0.0001     | 56.20             | 31.59              |
| 0.001      | 51.87             | 94.45              |
| 0.01       | 42.95             | 99.80              |

---

## Sparsity vs Accuracy Trade-Off

The following plot illustrates the trade-off between sparsity and accuracy for different λ values:

![Sparsity vs Accuracy Trade-Off](results/sparsity_vs_accuracy.png)

This plot demonstrates how increasing the sparsity regularization parameter (λ) affects the model's accuracy and sparsity. A higher λ results in greater sparsity but at the cost of reduced accuracy. The trade-off is evident in the plot, where λ = 0.001 achieves a good balance between sparsity and accuracy.

---

## Analysis

The results demonstrate a clear relationship between sparsity strength and model performance.

For a small λ value (0.0001), the sparsity penalty is weak, allowing most weights to remain active. This results in higher classification accuracy but lower model compression.

For a moderate λ value (0.001), the model achieves an effective balance between sparsity and accuracy. Approximately 94% of weights are pruned while maintaining reasonable classification performance. This indicates that many parameters in the original network are redundant.

For a large λ value (0.01), the sparsity constraint becomes very strong, causing most weights to be pruned. While this produces a highly compact model, classification accuracy decreases due to excessive removal of useful connections.

These observations confirm that sparsity can be effectively controlled through the regularization strength parameter λ.

---

## Gate Value Distribution

The distribution of gate values for the best-performing sparse model (λ = 0.001) shows:

• a large concentration of gate values close to zero
• a smaller cluster of important active connections

This distribution confirms that the network successfully learned which weights were unnecessary and pruned them during training.

Add generated plot in:

results/gate_distribution_lambda_0.001.png

---

## Engineering Considerations

Several design decisions were made to ensure efficient implementation:

A custom PyTorch layer was implemented to support differentiable gating.

Vectorized tensor operations were used to ensure computational efficiency.

The training pipeline was designed to support experimentation with different λ values.

The model architecture was intentionally kept simple to clearly demonstrate the pruning mechanism.

Code structure follows modular design principles for readability and maintainability.

---

## Conclusion

This project demonstrates the effectiveness of integrating learnable gate parameters with L1 regularization to create sparse neural networks.

The self-pruning mechanism enables the model to automatically identify redundant parameters and remove them during training.

Results show that significant reduction in active parameters can be achieved while maintaining acceptable accuracy levels.

Such sparsity-aware approaches are highly useful in practical machine learning systems where computational efficiency and model size are critical considerations.

This method is particularly relevant for deployment in edge devices, real-time systems, and large-scale AI applications.

---

## Future Work

Possible improvements include:

Applying pruning to convolutional neural networks.

Structured pruning at neuron or channel level.

Dynamic adjustment of λ during training.

Combining pruning with quantization techniques.

Evaluating performance on larger datasets.

Exploring reinforcement learning-based pruning strategies.

---

## Author

Nimisha S
B.Tech – Computer Science Engineering (CSBS)
SRM Institute of Science and Technology
2023-2027

