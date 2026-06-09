"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 01_binary_classification — Ex 1 of 3
Topic   : Feature vector construction and nx computation
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Binary classification begins before any model is involved — it starts with
turning raw data into a numeric vector the algorithm can consume. For image
models this means unrolling a 3-channel pixel array into a flat column vector.
Writing this from scratch makes concrete why nₓ has the value it does and
why shape discipline matters before a single weight is ever set. After this
exercise you will be able to state the shape of any image's feature vector
and explain why we use (nₓ, 1) rather than (nₓ,).

Rules
--------------------------------------------------------------------------
- Do not use np.flatten() — use .reshape() only.
- Use np.isclose or np.allclose for any numeric verification — never ==.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

# --------------------------------------------------------------------------
# Task 1 — Compute nₓ and unroll a single image into a column vector
# --------------------------------------------------------------------------
# A 64×64 RGB image is stored as a NumPy array of shape (64, 64, 3).
# Your job:
#   1. Create IMAGE using np.random.randint(0, 256, size=(64, 64, 3)).
#   2. Compute NX = total number of pixel values (H × W × C).
#   3. Unroll IMAGE into a column vector X of shape (NX, 1) using .reshape().
#   4. Print IMAGE.shape, NX, and X.shape.
#
# Expected result: IMAGE (64, 64, 3) | NX = 12288 | X shape (12288, 1)

np.random.seed(0)
IMAGE = np.random.randint(0, 256, size=(64, 64, 3))

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Verify the reshape preserves all pixel values
# --------------------------------------------------------------------------
# After unrolling, the total number of elements must not change.
# Use np.isclose to verify that NX == X.size.
# Print: "Element count preserved: True/False"
#
# Also verify X.shape == (12288, 1) with an assert statement.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# Why do we use shape (nₓ, 1) for the feature vector instead of shape (nₓ,)?
# What goes wrong in matrix arithmetic if you use the rank-1 form? One sentence.

# YOUR ANSWER HERE
comment to save the streak 
