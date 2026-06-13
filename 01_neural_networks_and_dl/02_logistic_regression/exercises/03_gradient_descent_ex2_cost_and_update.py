"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 03_gradient_descent — Ex 2 of 3
Topic   : Cost function and one gradient descent step
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Gradient descent needs two things: a cost to minimise (the average of the
loss over all m examples) and the direction to move (the gradient). This
exercise combines the cost function and the first update step so you can
see the cost actually decrease after one iteration. The gradient formulas
you use here — dW = (1/m)·X·dZᵀ, db = (1/m)·sum(dZ) — are the ones that
every neural network's output layer computes during backprop.

Rules
--------------------------------------------------------------------------
- Clip Y_hat before any np.log call — never log(0).
- Use np.isclose for scalar comparisons and np.allclose for arrays.
- Do not use a for-loop over examples in compute_cost — use np.sum.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# Task 1 — Implement compute_cost over m examples
# --------------------------------------------------------------------------

def compute_cost(Y_hat, Y, m):
    """
    Average binary cross-entropy over m examples.
    Formula: J = -(1/m) · sum(Y·log(Ŷ) + (1-Y)·log(1-Ŷ))
    Y_hat: shape (1, m) predicted probabilities
    Y:     shape (1, m) true binary labels
    m:     number of examples (int)
    Returns: scalar cost value
    Clip Y_hat to [1e-9, 1-1e-9] before log. No for-loops.
    """
    # YOUR CODE HERE
    pass


# Verify: all predictions = 0.5, all labels = 1, m = 4
# Expected J = -log(0.5) = log(2) ≈ 0.6931
M_TEST = 4
Y_TEST     = np.ones((1, M_TEST))
Y_HAT_TEST = np.full((1, M_TEST), 0.5)
J_TEST = compute_cost(Y_HAT_TEST, Y_TEST, M_TEST)

# Print J_TEST and verify with np.isclose(J_TEST, np.log(2))
# Expected: True

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — One full gradient descent step
# --------------------------------------------------------------------------
# Provided data:
np.random.seed(1)
NX, M = 3, 4
X = np.random.randn(NX, M)
Y = np.array([[1, 0, 1, 0]])
W = np.zeros((NX, 1))
B = 0.0
ALPHA = 0.1
#
# Steps:
#   1. Forward pass: Z = W.T @ X + B,  A = sigmoid(Z)
#   2. J_before = compute_cost(A, Y, M)
#   3. Gradients:
#      dW = (1/M) * X @ (A - Y).T    shape: (NX, 1)
#      dB = (1/M) * np.sum(A - Y)    scalar
#   4. Update: W_new = W - ALPHA * dW
#              B_new = B - ALPHA * dB
#   5. Forward pass again with W_new, B_new to get A_new
#   6. J_after = compute_cost(A_new, Y, M)
#   7. Print J_before, J_after, and assert J_after < J_before.
#
# Manual sanity check:
#   With W=zeros and B=0, all Z values are 0, so all A values are 0.5.
#   Confirm J_before ≈ 0.6931 before proceeding.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Verify gradient shapes
# --------------------------------------------------------------------------
# After Task 2, assert the following shapes and print each:
#   dW.shape == (NX, 1)
#   isinstance(dB, float) or dB.shape == ()
#
# Also verify: dW has the same shape as W. Print True/False using
# np.isclose is not needed here — this is a shape check, so == is correct.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# The gradient dW = (1/m)·X·dZᵀ has the same shape as W: (nₓ, 1).
# Explain why dividing by m (the number of examples) is important here.
# What would happen to the scale of the gradient — and therefore to training
# stability — if you omitted the 1/m factor? Two sentences.

# YOUR ANSWER HERE
