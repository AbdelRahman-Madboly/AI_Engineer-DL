"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 04_computation_graph_and_backprop — Ex 1 of 3
Topic   : dZ = A − y derived and verified against finite differences
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
The result dZ = A − y is derived by applying the chain rule through the
cross-entropy loss and the sigmoid activation. Writing it from scratch —
and then verifying it numerically — builds the habit of always checking
your gradient implementation before trusting it. Every deep learning
framework does this check internally; now you will too.

Rules
--------------------------------------------------------------------------
- Implement sigmoid from np.exp — no scipy.
- Verify the gradient with np.isclose(analytical, finite_diff, atol=1e-5).
- Use np.isclose for all scalar comparisons.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

# --------------------------------------------------------------------------
# Task 1 — Implement forward_pass returning Z, A, L
# --------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_pass(x, w, b, y):
    """
    Forward pass for logistic regression on one example.
    x: shape (nx, 1), w: shape (nx, 1), b: scalar, y: int (0 or 1)
    Returns dict: {'Z': float, 'A': float, 'L': float}
    Z = w.T @ x + b  (extract to scalar)
    A = sigmoid(Z)
    L = -(y*log(A + 1e-9) + (1-y)*log(1 - A + 1e-9))
    """
    # YOUR CODE HERE
    pass


# Test:
x_test = np.array([[1.0], [0.5]])
w_test = np.array([[0.3], [-0.1]])
b_test = 0.2
y_test = 1

# Manual calculation:
# Z = 0.3*(1.0) + (-0.1)*(0.5) + 0.2 = ? + ? + ? = ?
# A = sigmoid(Z) = ?

CACHE = forward_pass(x_test, w_test, b_test, y_test)
# Print Z, A, L from CACHE.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Implement backward_pass: dZ, dW, db
# --------------------------------------------------------------------------

def backward_pass(cache, x, y):
    """
    Backward pass for logistic regression on one example.
    Returns dict: {'dZ': float, 'dW': ndarray (nx,1), 'db': float}
    dZ = A - y
    dW = dZ * x    (shape nx, 1)
    db = dZ        (scalar)
    """
    # YOUR CODE HERE
    pass


GRADS = backward_pass(CACHE, x_test, y_test)
# Print dZ, dW (shape), db.
# Assert GRADS['dW'].shape == x_test.shape.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Verify dZ using finite differences on b
# --------------------------------------------------------------------------
# Finite difference approximation for db (= dL/db = dZ since dZ/db = 1):
#   L_plus  = forward_pass(x, w, b + epsilon, y)['L']
#   L_minus = forward_pass(x, w, b - epsilon, y)['L']
#   fd_db   = (L_plus - L_minus) / (2 * epsilon)
#
# Provided values (same as Tasks 1 & 2):
EPSILON = 1e-5
#
# Steps:
#   1. Compute L_plus and L_minus by perturbing b_test by ±EPSILON.
#   2. Compute fd_db.
#   3. Get analytical_db from GRADS['db'].
#   4. Verify: np.isclose(fd_db, analytical_db, atol=1e-5) — print True/False.
#   5. Print both values side by side.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# dZ = A − y is the result of applying the chain rule through both the
# cross-entropy loss and the sigmoid. Explain why this formula is described
# as "elegant" — what two steps of the chain rule combine to produce it,
# and what makes the final form computationally convenient? One sentence.

# YOUR ANSWER HERE
