"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/01_intro_to_dl
Exercise: 01_what_is_a_neural_network — Ex 3 of 3
Topic   : Two-layer network forward pass — housing price mini-network
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Scenario: You have 4 input features for a house (normalized size, number of
bedrooms, walkability score, school quality score). A small network with one
hidden layer of 3 neurons predicts a normalized price. Building this forward
pass from scratch — tracking shapes at every step, verifying them with
assertions — is the exact skill you need before implementing any real training
loop. Every deep learning forward pass you will ever write is just this
pattern, repeated across more layers. After this exercise you will be able to
implement a multi-layer forward pass, track matrix shapes, and plot a clean
activation curve — all from memory.

Rules
--------------------------------------------------------------------------
- Use only np.dot and your own relu — no np.maximum, no TensorFlow/Keras.
- Verify all shapes with assert statements — do not skip them.
- Use np.allclose for array comparisons — never ==.
- Save the plot to ../images/01_relu_activation.png.
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# --------------------------------------------------------------------------
# Helper — copy your relu from Ex 1 (do not import it)
# --------------------------------------------------------------------------

def relu(z):
    """ReLU element-wise. No np.maximum allowed."""
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Function definition — layer_forward
# --------------------------------------------------------------------------

def layer_forward(X, W, b):
    """
    Forward pass through one dense layer.
    Computes Z = W @ X + b, then A = relu(Z).
    Returns (A, Z) — Z is cached for backpropagation later.
    X shape: (n_in, m) where m is the number of examples (use 1 here).
    W shape: (n_out, n_in).
    b shape: (n_out, 1).
    A shape: (n_out, m).
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Provided data — use exactly as given, do not change the seed
# --------------------------------------------------------------------------

np.random.seed(0)
W1 = np.random.randn(3, 4) * 0.1   # layer 1: 3 neurons, 4 inputs
b1 = np.zeros((3, 1))
W2 = np.random.randn(1, 3) * 0.1   # layer 2: 1 neuron, 3 inputs from layer 1
b2 = np.zeros((1, 1))
x  = np.array([[0.8], [0.5], [0.3], [0.6]])  # one house, shape (4, 1)


# --------------------------------------------------------------------------
# Task 1 — Layer 1 forward pass
# --------------------------------------------------------------------------
# Compute A1, Z1 = layer_forward(x, W1, b1).
# Assert A1.shape == (3, 1) and Z1.shape == (3, 1).
# Print Z1 and A1 rounded to 4 decimal places.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Layer 2 forward pass
# --------------------------------------------------------------------------
# Compute A2, Z2 = layer_forward(A1, W2, b2).
# Assert A2.shape == (1, 1).
# Print the scalar prediction: A2[0, 0] rounded to 4 decimal places.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — What does A2 represent?
# --------------------------------------------------------------------------
# Print whether A2[0, 0] > 0 (the ReLU ensures non-negative price prediction).
# Add a comment: in a real housing model, how would you scale this output back
# to actual dollars?

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 4 — Shape summary table
# --------------------------------------------------------------------------
# Print a summary of all shapes in this forward pass.
# Format each line as: "Variable | shape | description"
# Variables: x, W1, b1, Z1, A1, W2, b2, Z2, A2

print("\nShape summary:")
print(f"{'Variable':<10} | {'Shape':<12} | Description")
print("-" * 50)
# YOUR CODE HERE
# Example of one line (fill in all 9):
# print(f"{'x':<10} | {str(x.shape):<12} | input features (4 normalized house features)")


# --------------------------------------------------------------------------
# Task 5 — Visualisation: ReLU activation curve
# --------------------------------------------------------------------------
# Plot ReLU for z in [-3, 3].
# Label x-axis "z (pre-activation)", y-axis "a = ReLU(z)".
# Title: "ReLU Activation Function".
# Add a vertical dashed line at z = 0 (color='grey', linestyle='--', alpha=0.5).
# Add a grid with alpha=0.3.
# Save to ../images/01_relu_activation.png with dpi=150, bbox_inches='tight'.

os.makedirs('../images', exist_ok=True)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# You built a two-layer network where both layers use ReLU.
# Why is the output A2 always >= 0? Is this desirable for a price prediction
# problem? What activation function would you use at the output layer instead
# if you wanted the output to be unbounded?
# (Two to three sentences.)

# YOUR ANSWER HERE
