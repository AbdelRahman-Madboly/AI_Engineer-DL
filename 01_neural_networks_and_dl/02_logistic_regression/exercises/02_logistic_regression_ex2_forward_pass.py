"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 02_logistic_regression — Ex 2 of 3
Topic   : Logistic regression forward pass on a mini dataset
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
The hypothesis ŷ = σ(wᵀx + b) has two sub-steps: the linear pre-activation
z = wᵀx + b, and the sigmoid squashing. Combining them correctly for a
batch of m examples requires careful shape tracking — w must be a column
vector, X columns must be examples, and the result must be (1, m).
This exercise practises both sub-steps together on a small dataset where
you can verify every value by hand.

Rules
--------------------------------------------------------------------------
- Do not use scipy.special.expit — use your own sigmoid() built from np.exp.
- Use np.isclose for scalar comparisons and np.allclose for arrays.
- Do not use a for-loop over examples — one matrix multiply for the full batch.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# Task 1 — Compute the pre-activation Z for a single example
# --------------------------------------------------------------------------
# Provided parameters and example:
W = np.array([[0.5], [-0.3], [0.8]])   # shape (3, 1)
B = 0.1                                 # scalar bias
X1 = np.array([[1.0], [2.0], [-1.0]])  # single example, shape (3, 1)
#
# Steps:
#   1. Compute Z1 = W.T @ X1 + B  (use @ operator)
#   2. Compute A1 = sigmoid(Z1)
#   3. Manual calculation:
#      z = 0.5*1 + (-0.3)*2 + 0.8*(-1) + 0.1 = ?   (fill in the value)
#      Write the step-by-step arithmetic as a comment below.
#   4. Verify np.isclose(float(Z1), your_manual_z) — print True/False.
#   5. Print Z1, A1, and Z1.shape.
#
# Manual calculation:
# z = 0.5*(1.0) + (-0.3)*(2.0) + 0.8*(-1.0) + 0.1 = ?

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Vectorised forward pass for m examples
# --------------------------------------------------------------------------
# Provided mini dataset:
np.random.seed(42)
NX, M = 3, 6
X_BATCH = np.array([
    [ 1.0,  0.5, -1.0,  2.0,  0.0, -0.5],
    [ 2.0, -1.0,  0.5, -0.5,  1.5,  0.0],
    [-1.0,  1.0,  2.0,  0.5, -1.5,  1.0],
])                                          # shape (3, 6)
Y_BATCH = np.array([[1, 0, 1, 1, 0, 0]])   # shape (1, 6)
#
# Steps:
#   1. Z_batch = W.T @ X_BATCH + B          shape must be (1, M)
#   2. A_batch = sigmoid(Z_batch)            shape must be (1, M)
#   3. Assert Z_batch.shape == (1, M) and A_batch.shape == (1, M).
#   4. Count how many predictions cross the 0.5 decision boundary
#      (i.e. A_batch >= 0.5 implies predicted class 1).
#      Print predicted labels and compare to Y_BATCH.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Verify the first column against the single-example result
# --------------------------------------------------------------------------
# The first column of Z_batch should equal Z1 (from Task 1), because
# X_BATCH[:, 0:1] == X1.
# Verify: np.allclose(Z_batch[:, 0:1], Z1)
# Print True/False.
#
# Hint (read only if stuck)
# --------------------------------------------------------------------------
# Use Z_batch[:, 0:1] (shape 1×1) not Z_batch[:, 0] (rank-1 shape (1,))
# to keep the matrix shape consistent for allclose comparison.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 2 you added the scalar B to the (1, M) matrix W.T @ X_BATCH.
# Explain what NumPy broadcasting does here — what shape does B effectively
# become, and why does this work? Two sentences.

# YOUR ANSWER HERE
