"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 05_vectorization — Ex 1 of 3
Topic   : Rank-1 array bug and vectorised forward pass
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
The single most common NumPy bug in deep learning implementations is the
rank-1 array — shape (n,) instead of (n, 1) or (1, n). It is silent: the
code runs, the shapes look right at a glance, but the math is wrong. This
exercise makes you reproduce the bug, understand exactly what goes wrong,
fix it, and then use the correct shapes in a vectorised forward pass over m
examples. After this exercise you will never create a weight vector with
np.random.randn(n) again.

Rules
--------------------------------------------------------------------------
- Use assert statements to catch shape bugs — at least one per task.
- Use np.allclose for array comparisons and np.isclose for scalars.
- Do not use a for-loop in the forward pass — one matrix multiply only.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# Task 1 — Reproduce the rank-1 bug
# --------------------------------------------------------------------------
# Provided rank-1 vector:
np.random.seed(0)
a_buggy = np.random.randn(5)    # rank-1: shape (5,) — WRONG for ML
#
# Steps:
#   1. Print a_buggy.shape and a_buggy.T.shape — confirm .T does nothing.
#   2. Compute a_buggy.T @ a_buggy.
#      Print the result and its type/shape.
#      Expected: a scalar float (the inner product) — NOT a (5,5) outer product.
#   3. Fix it: a_col = a_buggy.reshape(5, 1)
#   4. Compute a_col.T @ a_col and a_col @ a_col.T.
#      Print both shapes.
#      Expected: (1,1) for inner product, (5,5) for outer product.
#   5. Write an assert that would have caught the bug:
#      assert a_col.shape == (5, 1), f"Shape error: got {a_col.shape}"

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Vectorised forward pass: Z and A for m examples
# --------------------------------------------------------------------------
# Provided data (correct shapes — no rank-1 arrays):
np.random.seed(7)
NX, M = 4, 50
X = np.random.randn(NX, M)           # shape (NX, M) — examples are columns
W = np.random.randn(NX, 1) * 0.01   # shape (NX, 1) — column vector
B = 0.0                               # scalar bias
Y = np.random.randint(0, 2, (1, M))  # shape (1, M)
#
# Steps:
#   1. Z = W.T @ X + B                 (NO for-loops)
#   2. A = sigmoid(Z)
#   3. Assert Z.shape == (1, M) and A.shape == (1, M). Print both.
#   4. Print A[0, :5] (first 5 predicted probabilities, rounded to 3 dp).
#   5. Verify all values in A are in (0, 1):
#      assert np.all(A > 0) and np.all(A < 1)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# a_buggy = np.random.randn(5) has shape (5,). Calling .T on it returns
# the exact same array — transposing a rank-1 array does nothing.
# Explain why this is dangerous in logistic regression: what specific
# operation produces an incorrect scalar when W has shape (5,) instead
# of (5, 1)? One sentence.

# YOUR ANSWER HERE
