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
#
# Steps:
#   1. Compute NX = H × W × C.
#   2. Flatten each image in IMAGES_LIST into a column vector of shape (NX, 1).
#   3. Stack all column vectors horizontally with np.hstack() → X of shape (NX, m).
#   4. Build Y of shape (1, m) from LABELS using .reshape(1, -1).
#   5. Print X.shape and Y.shape.
#
# Expected result: X shape (3072, 8) | Y shape (1, 8)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — One-shot batch reshape from a (m, H, W, C) array
# --------------------------------------------------------------------------
# The same X can be built from a 4-D batch array in one line.
#
# Provided batch array:
np.random.seed(7)
BATCH = np.random.randint(0, 256, size=(m, H, W, C))
#
# Steps:
#   1. Reshape BATCH to shape (m, NX) → then transpose to (NX, m) to get X_direct.
#      Do this in one expression: BATCH.reshape(m, -1).T
#   2. Assert X_direct.shape == (NX, m).
#   3. Verify np.allclose(X, X_direct) — both methods should give the same matrix
#      since IMAGES_LIST was built from the same seed as BATCH.
#   4. Print the allclose result.

# YOUR CODE HERE


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

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# Why does every neural network stack training examples as columns (nₓ × m)
# rather than as rows (m × nₓ)? What specific operation becomes cleaner?
# Two sentences.

# YOUR ANSWER HERE
