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
W  = np.array([[0.5], [-0.3], [0.8]])   # shape (3, 1)
B  = 0.1                                 # scalar bias
X1 = np.array([[1.0], [2.0], [-1.0]])   # single example, shape (3, 1)

Z1 = W.T @ X1 + B      # shape (1, 1)
A1 = sigmoid(Z1)        # shape (1, 1)

# Manual calculation:
# z = 0.5*(1.0) + (-0.3)*(2.0) + 0.8*(-1.0) + 0.1
#   = 0.5  +  (-0.6)  +  (-0.8)  +  0.1
#   = 0.5 - 0.6 - 0.8 + 0.1
#   = -0.8
manual_z = 0.5 * 1.0 + (-0.3) * 2.0 + 0.8 * (-1.0) + 0.1    # = -0.8

print(f"Z1           : {Z1}")
print(f"A1           : {A1}")
print(f"Z1.shape     : {Z1.shape}")
print(f"manual z     : {manual_z}")
print(f"Z1 == manual : {np.isclose(Z1.item(), manual_z)}")    # Expected True


# --------------------------------------------------------------------------
# Task 2 — Vectorised forward pass for m examples
# --------------------------------------------------------------------------
np.random.seed(42)
NX, M = 3, 6
X_BATCH = np.array([
    [ 1.0,  0.5, -1.0,  2.0,  0.0, -0.5],
    [ 2.0, -1.0,  0.5, -0.5,  1.5,  0.0],
    [-1.0,  1.0,  2.0,  0.5, -1.5,  1.0],
])                                          # shape (3, 6)
Y_BATCH = np.array([[1, 0, 1, 1, 0, 0]])   # shape (1, 6)

Z_batch = W.T @ X_BATCH + B                # shape (1, M)
A_batch = sigmoid(Z_batch)                 # shape (1, M)

assert Z_batch.shape == (1, M), f"Expected (1, {M}), got {Z_batch.shape}"
assert A_batch.shape == (1, M), f"Expected (1, {M}), got {A_batch.shape}"

Y_PRED_BATCH = (A_batch >= 0.5).astype(int)
n_correct    = int(np.sum(Y_PRED_BATCH == Y_BATCH))

print(f"\nZ_batch (3 dp) : {Z_batch.round(3)}")
print(f"A_batch (3 dp) : {A_batch.round(3)}")
print(f"Predicted      : {Y_PRED_BATCH}")
print(f"True labels    : {Y_BATCH}")
print(f"Correct        : {n_correct} / {M}")


# --------------------------------------------------------------------------
# Task 3 — Verify the first column against the single-example result
# --------------------------------------------------------------------------
# X_BATCH[:, 0:1] == X1, so Z_batch[:, 0:1] must equal Z1.

print(f"\nallclose(Z_batch[:, 0:1], Z1): {np.allclose(Z_batch[:, 0:1], Z1)}")   # True


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 2 you added the scalar B to the (1, M) matrix W.T @ X_BATCH.
# Explain what NumPy broadcasting does here — what shape does B effectively
# become, and why does this work? Two sentences.

# NumPy broadcasting treats the scalar B as if it had shape (1, 1), then
# stretches it to match the (1, M) shape of W.T @ X_BATCH by replicating
# the value across all M columns — so each of the M pre-activations gets
# the same bias added without the programmer needing to construct a bias vector.
# This works because the shapes are compatible: every dimension of B (treated
# as 1) can be broadcast to the corresponding dimension of the matrix (1 and M).
