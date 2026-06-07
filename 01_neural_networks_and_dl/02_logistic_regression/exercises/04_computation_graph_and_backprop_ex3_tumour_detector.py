"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 04_computation_graph_and_backprop — Ex 3 of 3
Topic   : Complete backprop pipeline with full gradient check — tumour detector
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Scenario: a medical imaging team has a 6-feature logistic regression model
for binary tumour classification (malignant=1, benign=0). Features are:
tumour size (cm), patient age (norm), texture variance, perimeter (norm),
area (norm), and concavity score. Your job is to implement the complete
single-example forward/backward pass, run a full gradient check over ALL
nₓ weights (not just one), and produce a diagnostic plot comparing
analytical and numerical gradients for every weight. This is the most
rigorous validation of a backprop implementation you can do.

Rules
--------------------------------------------------------------------------
- Use np.isclose(analytical, numerical, atol=1e-5) for every gradient check.
- Loop over weights only in the finite difference check — not in forward/backward.
- Clip log inputs to [1e-9, 1-1e-9] — never log(0).
- Save plot to: ../images/04_computation_graph_backprop_gradient_check.png
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# Task 1 — Dataset and parameter setup
# --------------------------------------------------------------------------
# Provided tumour example (do not modify):
NX = 6
FEATURE_NAMES = [
    "Tumour size",
    "Patient age",
    "Texture var.",
    "Perimeter",
    "Area",
    "Concavity"
]

x_tumour = np.array([[0.82],   # tumour size (normalised)
                      [0.61],   # patient age (normalised)
                      [0.47],   # texture variance
                      [0.73],   # perimeter (normalised)
                      [0.55],   # area (normalised)
                      [0.68]])  # concavity score
                                # shape (6, 1)

w_tumour = np.array([[ 0.5],
                      [-0.2],
                      [ 0.3],
                      [ 0.4],
                      [-0.1],
                      [ 0.6]])  # shape (6, 1)
b_tumour = -0.3
y_tumour = 1    # malignant

# Print x_tumour.shape, w_tumour.shape, and b_tumour.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Forward pass
# --------------------------------------------------------------------------
# Implement forward_pass(x, w, b, y) that returns {'Z', 'A', 'L'}.
# Z = w.T @ x + b  (extract to scalar)
# A = sigmoid(Z)
# L = -[y*log(A + 1e-9) + (1-y)*log(1 - A + 1e-9)]
#
# Run it on (x_tumour, w_tumour, b_tumour, y_tumour).
# Print Z, A, L.
# Also print: "Model is X% confident this tumour is malignant." using A.

def forward_pass(x, w, b, y):
    """
    Forward pass for logistic regression on one example.
    Returns dict with keys 'Z', 'A', 'L' — all scalar floats.
    """
    # YOUR CODE HERE
    pass


# YOUR CODE HERE (call forward_pass, print results)


# --------------------------------------------------------------------------
# Task 3 — Backward pass
# --------------------------------------------------------------------------
# Implement backward_pass(cache, x, y) that returns {'dZ', 'dW', 'db'}.
# dZ = A - y  (scalar)
# dW = dZ * x  (shape NX, 1)
# db = dZ  (scalar)
#
# Run on the tumour example. Print dZ, db, and dW (all 6 values with feature names).
# Assert dW.shape == (NX, 1).

def backward_pass(cache, x, y):
    """
    Backward pass for logistic regression on one example.
    Returns dict with 'dZ' (scalar), 'dW' (shape NX,1), 'db' (scalar).
    """
    # YOUR CODE HERE
    pass


# YOUR CODE HERE (call backward_pass, print results)


# --------------------------------------------------------------------------
# Task 4 — Full gradient check over all NX weights
# --------------------------------------------------------------------------
# For every weight w[i, 0] (i = 0 to NX-1):
#   1. Build w_plus  = w_tumour.copy(), set w_plus[i, 0]  += EPSILON
#   2. Build w_minus = w_tumour.copy(), set w_minus[i, 0] -= EPSILON
#   3. L_plus  = forward_pass(x_tumour, w_plus,  b_tumour, y_tumour)['L']
#   4. L_minus = forward_pass(x_tumour, w_minus, b_tumour, y_tumour)['L']
#   5. fd_dw[i]         = (L_plus - L_minus) / (2 * EPSILON)
#   6. analytical_dw[i] = float(grads['dW'][i, 0])
#
# After the loop:
#   7. Check all NX pairs with np.allclose(analytical_dw, fd_dw, atol=1e-5).
#      Print "All gradient checks passed: True/False".
#   8. Print a table: feature name | analytical | finite diff | match (T/F)

EPSILON = 1e-5
cache_tumour = forward_pass(x_tumour, w_tumour, b_tumour, y_tumour)
grads_tumour = backward_pass(cache_tumour, x_tumour, y_tumour)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
# Two-panel figure:
#
#   Left panel: grouped bar chart — analytical vs finite-difference dW.
#     X-axis: FEATURE_NAMES (6 names, use np.arange(NX) for positions)
#     Y-axis: "Gradient value"
#     Two bars per feature: "Analytical dW" (steelblue) and "Finite diff" (coral, hatched)
#     Title: "Tumour Detector: Analytical vs Finite-Difference Gradients"
#     Add legend. Use bar width = 0.35.
#
#   Right panel: scatter plot of analytical_dw[i] vs fd_dw[i] for i=0..NX-1.
#     X-axis: "Analytical dW"
#     Y-axis: "Finite-difference dW"
#     Draw a diagonal y=x line (dashed grey) as reference.
#     Label each point with its feature name using ax.annotate.
#     Title: "Gradient Check: Analytical vs Numerical (should lie on y=x)"
#
# Save to: ../images/04_computation_graph_backprop_gradient_check.png

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In your gradient check, all NX analytical gradients matched the numerical
# approximations to within 1e-5. What does this confirm about your backprop
# implementation, and why would a mismatch here be the first place you look
# when a model fails to converge during training? Two to three sentences.

# YOUR ANSWER HERE
