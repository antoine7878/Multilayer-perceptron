# Multilayer Perceptron

**Multilayer Perceptron** is a machine learning project.
It involves implementing a feedforward, fully connected neural network from scratch, without using high-level machine learning libraries.

The project is structured around understanding neural networks at the algorithmic level, while enforcing strict modularity and clean code.
All core neural network logic — including layer definition, weight initialization, activation functions, and learning algorithms — must be implemented manually.

## ⚙️ Features

- Modular Fully Connected network
- Optimizers: `SGD`, `RMSprop`
- Activation functions: `sigmoid`, `tanh`, `ReLU`, `softmax`
- Cost function: `cross-entropy`
- Weight initialization: `normal`, `HeUniform`
- Dataset preprocessing: `train/test splitting`, `normalization`, `one-hot encoding`
- Training  and model evaluation: `loss`, `accuracy`
- Model serialization to disk (weights and topology, activation)
- Fully vectorized NumPy based computations
- Stochastic and mini batch `Gradient Descent`
- Tested on 2 datasets:
  - Wisconsin Breast Cancer Dataset
  - MNIST Digits

Target: x86_64-pc-linux-gnu
