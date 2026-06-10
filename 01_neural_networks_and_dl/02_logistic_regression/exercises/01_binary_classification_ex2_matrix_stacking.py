"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 01_binary_classification — Ex 2 of 3
Topic   : Building X and Y from a batch of images
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
A single feature vector is not useful for training — you need the full
matrix X of shape (nₓ, m) where every column is one example, and Y of
shape (1, m) holding the labels. Building these matrices correctly is the
first step of every image training pipeline and the layout that all
subsequent gradient math assumes. This exercise makes you practice both
the list-of-vectors approach and the one-shot batch-reshape shortcut,
so you understand why they produce the same result.

Rules
--------------------------------------------------------------------------
- Do not use np.flatten() — use .reshape() only.
- Use np.allclose for any array comparison — never a == b.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

# --------------------------------------------------------------------------
# Task 1 — Stack m images as columns to build X
# --------------------------------------------------------------------------
# You have m = 8 simulated 32×32 RGB images.
# Each image should be flattened into a column vector then stacked.
#
# Provided data:
np.random.seed(7)
m = 8
H, W, C = 32, 32, 3
IMAGES_LIST = [np.random.randint(0, 256, size=(H, W, C)) for _ in range(m)]
LABELS = np.array([1, 0, 1, 1, 0, 0, 1, 0])   # ground-truth labels

NX = H * W * C                                          # 3072

# Flatten each image to (NX, 1) and hstack into (NX, m)
X = np.hstack([img.reshape(NX, 1) for img in IMAGES_LIST])
Y = LABELS.reshape(1, -1)

print(f"X shape: {X.shape}")   # Expected (3072, 8)
print(f"Y shape: {Y.shape}")   # Expected (1, 8)


# --------------------------------------------------------------------------
# Task 2 — One-shot batch reshape from a (m, H, W, C) array
# --------------------------------------------------------------------------
# The same X can be built from a 4-D batch array in one line.
#
# Provided batch array:
np.random.seed(7)
BATCH = np.random.randint(0, 256, size=(m, H, W, C))

X_direct = BATCH.reshape(m, -1).T          # shape (NX, m) in one expression

assert X_direct.shape == (NX, m), f"Expected ({NX}, {m}), got {X_direct.shape}"

print(f"allclose(X, X_direct): {np.allclose(X, X_direct)}")   # Expected True


# --------------------------------------------------------------------------
# Task 3 — Retrieve a single example from X
# --------------------------------------------------------------------------
# Access the 3rd training example (index 2) from X as a column vector of
# shape (NX, 1) — NOT as a rank-1 array of shape (NX,).
# Print its shape and verify it is (NX, 1) with an assert.
#
# Hint (read only if stuck)
# --------------------------------------------------------------------------
# To keep the column dimension, use X[:, 2:3] rather than X[:, 2].

x3 = X[:, 2:3]   # shape (NX, 1)
print(f"x3.shape: {x3.shape}")
assert x3.shape == (NX, 1), f"Expected ({NX}, 1), got {x3.shape}"


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# Why does every neural network stack training examples as columns (nₓ × m)
# rather than as rows (m × nₓ)? What specific operation becomes cleaner?
# Two sentences.

# Stacking examples as columns allows the linear forward pass Z = W.T @ X + b
# to be written as a single matrix multiply that naturally broadcasts the bias b
# across all m examples at once, giving Z of shape (n_out, m) without any
# transposition. If examples were rows, every weight matrix and bias operation
# would need extra transposes, complicating both code and gradient derivations.
