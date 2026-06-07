"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 01_binary_classification — Ex 3 of 3
Topic   : Mini dataset assembly pipeline for WaveMamba-DF inference batch
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Scenario: WaveMamba-DF needs to run inference on a batch of video frames.
Each frame is a 48×48 RGB image. Your job is to build the complete data
pipeline — from raw pixel arrays to the (nₓ, m) matrix and (1, m) label
row — that the logistic regression output layer will consume. This mirrors
exactly how a real inference batch is assembled before it enters a neural
network. By the end you will have a reusable pipeline and a plot confirming
that the assembled feature matrix has the distribution you expect.

Rules
--------------------------------------------------------------------------
- Do not use np.flatten() — use .reshape() only.
- Use np.allclose for array comparisons — never a == b.
- All numeric assertions must use np.isclose or np.allclose.
- Save the plot to ../images/01_binary_classification_batch_stats.png
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Task 1 — Build the raw frame batch
# --------------------------------------------------------------------------
# Simulate a batch of m=16 deepfake detection frames (48×48 RGB images).
# Half are labelled deepfake (1), half are real (0).
#
# Provided:
np.random.seed(42)
m = 16
H, W, C = 48, 48, 3
FRAMES = np.random.randint(0, 256, size=(m, H, W, C))    # raw batch
LABELS_RAW = np.array([1]*8 + [0]*8)                     # 8 deepfake, 8 real
#
# Steps:
#   1. Compute NX = H × W × C.
#   2. Reshape FRAMES to X of shape (NX, m) in one expression.
#   3. Reshape LABELS_RAW to Y of shape (1, m).
#   4. Print NX, X.shape, Y.shape.
#
# Expected result: NX = 6912 | X shape (6912, 16) | Y shape (1, 16)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Normalise pixel values to [0, 1]
# --------------------------------------------------------------------------
# Neural networks train faster when inputs are in [0, 1].
# Divide X by 255 to get X_norm.
# Verify: np.allclose(X_norm.min(), 0.0, atol=0.01) and
#         np.allclose(X_norm.max(), 1.0, atol=0.01) — print both checks.
#
# Do not use a for-loop — one vectorised operation.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Verify dataset integrity
# --------------------------------------------------------------------------
# Run the following checks and print each result:
#   a. X_norm.shape == (NX, m)              → assert + print
#   b. Y.shape == (1, m)                    → assert + print
#   c. Number of deepfake labels (Y == 1)   → np.sum, compare to 8
#   d. Number of real labels (Y == 0)       → np.sum, compare to 8
#   e. Per-feature mean across m examples: MEAN_VEC = X_norm.mean(axis=1)
#      MEAN_VEC.shape should be (NX,) — print shape and mean of MEAN_VEC.
#      Expected: mean of MEAN_VEC ≈ 0.5 (random uniform pixels)
#      Verify with np.isclose(MEAN_VEC.mean(), 0.5, atol=0.05).

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 4 — Retrieve and inspect one example
# --------------------------------------------------------------------------
# Extract the 5th example (index 4) as:
#   (a) a column vector x5 of shape (NX, 1) — NOT rank-1
#   (b) its original frame back as a (H, W, C) array x5_frame
#       using .reshape(H, W, C)
# Print x5.shape and x5_frame.shape.
# Assert x5.shape == (NX, 1).
# Verify np.allclose(x5.reshape(H, W, C), x5_frame) and print True/False.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
# Two-panel figure:
#   Left panel: bar chart — class distribution (deepfake vs real).
#     X-axis: "Class", labels ["Deepfake (y=1)", "Real (y=0)"]
#     Y-axis: "Count"
#     Title: "WaveMamba-DF Batch: Class Distribution"
#     Bar colours: ['steelblue', 'coral']
#
#   Right panel: histogram of per-feature means (MEAN_VEC from Task 3e).
#     X-axis: "Feature Mean (normalised pixel)"
#     Y-axis: "Frequency"
#     Title: "Distribution of Feature Means Across nₓ Features"
#     Use 40 bins, colour 'steelblue'.
#     Add a vertical dashed red line at x=0.5 labelled "expected ≈ 0.5".
#
# Save to: ../images/01_binary_classification_batch_stats.png

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 2 you normalised inputs to [0, 1] by dividing by 255.
# Why does this normalisation help gradient descent converge faster?
# What would happen to the gradient updates if pixel values stayed in [0, 255]?
# Two to three sentences.

# YOUR ANSWER HERE
