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
    return np.where(z > 0, z, 0.0)


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
    Z = np.dot(W, X) + b
    A = relu(Z)
    return A, Z


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

A1, Z1 = layer_forward(x, W1, b1)

# Verify shapes
assert A1.shape == (3, 1), f"Expected A1 shape (3, 1), got {A1.shape}"
assert Z1.shape == (3, 1), f"Expected Z1 shape (3, 1), got {Z1.shape}"

print("--- Task 1: Layer 1 Output ---")
print(f"Z1:\n{np.round(Z1, 4)}")
print(f"A1:\n{np.round(A1, 4)}\n")

# --------------------------------------------------------------------------
# Task 2 — Layer 2 forward pass
# --------------------------------------------------------------------------
# Compute A2, Z2 = layer_forward(A1, W2, b2).
# Assert A2.shape == (1, 1).
# Print the scalar prediction: A2[0, 0] rounded to 4 decimal places.

A2, Z2 = layer_forward(A1, W2, b2)

# Verify shape
assert A2.shape == (1, 1), f"Expected A2 shape (1, 1), got {A2.shape}"

print("--- Task 2: Layer 2 Output ---")
print(f"Prediction (A2 scalar): {A2[0, 0]:.4f}\n")

# --------------------------------------------------------------------------
# Task 3 — What does A2 represent?
# --------------------------------------------------------------------------
# Print whether A2[0, 0] > 0 (the ReLU ensures non-negative price prediction).
# Add a comment: in a real housing model, how would you scale this output back
# to actual dollars?

print("--- Task 3: A2 Analysis ---")
print(f"Is A2 > 0? {A2[0, 0] > 0}")

# Comment: In a real housing model where prices were normalized (e.g., using z-score normalization), 
# you would un-normalize this prediction by multiplying it by the standard deviation of the 
# original price data and adding the mean of the original price data. 
# If Min-Max scaling was used, you would multiply by the (Max - Min) range and add the Min.
print()

# --------------------------------------------------------------------------
# Task 4 — Shape summary table
# --------------------------------------------------------------------------
# Print a summary of all shapes in this forward pass.
# Format each line as: "Variable | shape | description"
# Variables: x, W1, b1, Z1, A1, W2, b2, Z2, A2

print("\nShape summary:")
print(f"{'Variable':<10} | {'Shape':<12} | Description")
print("-" * 50)
print("--- Task 4: Shape summary ---")
print(f"{'Variable':<10} | {'Shape':<12} | Description")
print("-" * 65)
print(f"{'x':<10} | {str(x.shape):<12} | input features (4 normalized house features)")
print(f"{'W1':<10} | {str(W1.shape):<12} | layer 1 weights (3 neurons, 4 inputs)")
print(f"{'b1':<10} | {str(b1.shape):<12} | layer 1 biases (3 neurons)")
print(f"{'Z1':<10} | {str(Z1.shape):<12} | layer 1 pre-activations")
print(f"{'A1':<10} | {str(A1.shape):<12} | layer 1 activations")
print(f"{'W2':<10} | {str(W2.shape):<12} | layer 2 weights (1 neuron, 3 inputs)")
print(f"{'b2':<10} | {str(b2.shape):<12} | layer 2 biases (1 neuron)")
print(f"{'Z2':<10} | {str(Z2.shape):<12} | layer 2 pre-activations")
print(f"{'A2':<10} | {str(A2.shape):<12} | layer 2 activations (final output)")
print()


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

# Generate values
z_vals = np.linspace(-3, 3, 100)
a_vals = relu(z_vals)

plt.figure(figsize=(8, 5))
plt.plot(z_vals, a_vals, label="ReLU(z)", color="blue", linewidth=2)

# Specific formatting requested
plt.title("ReLU Activation Function")
plt.xlabel("z (pre-activation)")
plt.ylabel("a = ReLU(z)")
plt.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.legend()

# Save the plot
save_path = '../images/01_relu_activation.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"--- Task 5: Plot saved successfully to {save_path} ---")


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# You built a two-layer network where both layers use ReLU.
# Why is the output A2 always >= 0? Is this desirable for a price prediction
# problem? What activation function would you use at the output layer instead
# if you wanted the output to be unbounded?
# (Two to three sentences.)

# A2 is always >= 0 because the final layer passes the weighted sum Z2 through the ReLU function, 
# which strictly clamps any negative numbers to zero. This behavior is highly desirable 
# for predicting house prices, as real estate cannot have a negative value. 
# If we wanted the network to output completely unbounded predictions (both positive and negative), 
# we would simply use a linear activation function (meaning we just output Z2 directly without applying ReLU) 
# at the final layer.