"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 05_vectorization — Ex 2 of 3
Topic   : Vectorised backward pass and speed comparison
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
The vectorised backward pass — dZ = A−Y, dW = (1/m)·X·dZᵀ, db = (1/m)·sum(dZ)
— computes all m gradients simultaneously with no Python for-loop. Combining
this with the vectorised forward pass gives you a complete, loop-free gradient
descent step. This exercise also makes you time the vectorised version against
an explicit loop so the speedup is not just theoretical — you measure it.

Rules
--------------------------------------------------------------------------
- No for-loops in the vectorised versions — only in the loop-based version.
- Use np.allclose to verify results match between vectorised and loop versions.
- Use time.time() for timing — not timeit.
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import time

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# Task 1 — Vectorised backward pass (zero for-loops)
# --------------------------------------------------------------------------
# Provided data and forward pass results:
np.random.seed(3)
NX, M = 5, 100
X = np.random.randn(NX, M)
Y = np.random.randint(0, 2, (1, M))
W = np.random.randn(NX, 1) * 0.01
B = 0.0

# Pre-computed forward pass:
Z = W.T @ X + B
A = sigmoid(Z)   # shape (1, M)

# Steps:
#   1. dZ = A - Y              shape: (1, M)
#   2. dW = (1/M) * X @ dZ.T  shape: (NX, 1)   — NO for-loops
#   3. dB = (1/M) * np.sum(dZ) scalar
#
# Assert all three shapes:
#   dZ.shape == (1, M)
#   dW.shape == (NX, 1)
#   np.isscalar(dB) or dB.shape == ()
#
# Print all shapes and dW[:3] (first 3 weight gradients).

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — One complete vectorised gradient descent step
# --------------------------------------------------------------------------
# Wrap forward + backward + update into one function:

def lr_step_vectorised(X, Y, W, B, alpha):
    """
    One complete gradient descent step for logistic regression.
    Forward pass → cost → backward pass → parameter update.
    No for-loops. Returns (W_new, B_new, cost).
    Clip A to [1e-9, 1-1e-9] before log.
    """
    # YOUR CODE HERE
    pass


# Test: use the same NX, M, X, Y from Task 1.
# Re-initialise W=zeros, B=0, alpha=0.1.
# Run 1 step. Print cost_before (=cost returned) and cost after another step.
# Verify cost decreases: assert step2_cost < step1_cost.

np.random.seed(3)
W_init = np.zeros((NX, 1))
B_init = 0.0
ALPHA  = 0.1

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Speed comparison: vectorised vs loop forward pass
# --------------------------------------------------------------------------
# Setup — large arrays:
np.random.seed(0)
NX_LARGE, M_LARGE = 500, 5000
X_LARGE = np.random.randn(NX_LARGE, M_LARGE)
W_LARGE = np.random.randn(NX_LARGE, 1)
B_LARGE = 0.5

# 1. VECTORISED: Z_vec = W_LARGE.T @ X_LARGE + B_LARGE  (time this)
# 2. LOOP:
#      Z_loop = np.zeros((1, M_LARGE))
#      for i in range(M_LARGE):
#          Z_loop[0, i] = np.dot(W_LARGE.T, X_LARGE[:, i]) + B_LARGE
#      (time this)
#
# 3. Verify: np.allclose(Z_vec, Z_loop) — print True/False.
# 4. Print: vec time (ms), loop time (ms), speedup factor.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 3 you measured a speedup by replacing the Python for-loop with
# a matrix multiply. NumPy's matrix multiply uses BLAS routines compiled
# to native code. Why is this so much faster than a Python for-loop, even
# when both perform the same total number of multiplications and additions?
# Two sentences.

# YOUR ANSWER HERE
