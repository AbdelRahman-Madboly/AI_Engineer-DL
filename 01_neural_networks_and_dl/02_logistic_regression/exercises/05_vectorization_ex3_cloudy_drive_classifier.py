"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 05_vectorization — Ex 3 of 3
Topic   : Full vectorised logistic regression — CloudyDrive obstacle classifier
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Scenario: CloudyDrive's perception module runs a logistic regression
classifier on 8 features extracted from LiDAR point clouds to decide
whether an object in the path is an obstacle (y=1) or background (y=0).
Features include: distance (norm), height, width, density, reflectivity,
x-velocity, y-velocity, and object aspect ratio — all normalised to [0, 1].
Your job is to implement the complete, loop-free logistic regression
training pipeline: data setup → training loop → accuracy evaluation →
decision boundary plot. This is the culmination of all five logistic
regression notebooks written as a single working classifier.

Rules
--------------------------------------------------------------------------
- Zero for-loops over m examples at any point — forward, backward, and cost
  must all be vectorised.
- Use np.allclose for all array verifications and np.isclose for scalars.
- All assert statements must be in place before committing.
- Save plot to: ../images/05_vectorization_full_lr_classifier.png
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------------------------------------------------------
# Task 1 — Dataset setup
# --------------------------------------------------------------------------
# Provided LiDAR feature dataset for 80 frames (40 obstacle, 40 background):
np.random.seed(12)
NX = 8
M  = 80
FEATURE_NAMES = [
    "Distance", "Height", "Width", "Density",
    "Reflectivity", "x-velocity", "y-velocity", "Aspect ratio"
]

# Obstacle frames (y=1): higher density, reflectivity, velocity features
X_OBS  = np.clip(np.random.randn(NX, 40) * 0.2 + 0.7, 0, 1)
# Background frames (y=0): lower values across features
X_BG   = np.clip(np.random.randn(NX, 40) * 0.2 + 0.3, 0, 1)

X = np.hstack([X_OBS, X_BG])            # shape (NX, M)
Y = np.hstack([np.ones((1, 40)), np.zeros((1, 40))])  # shape (1, M)

# Shuffle columns together so obstacles and background are interleaved:
np.random.seed(12)
SHUFFLE_IDX = np.random.permutation(M)
X = X[:, SHUFFLE_IDX]
Y = Y[:, SHUFFLE_IDX]

# Verify and print:
#   X.shape == (NX, M) — assert
#   Y.shape == (1, M)  — assert
#   Number of obstacle frames (np.sum(Y)) == 40 — verify with np.isclose

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Implement all helper functions (zero for-loops allowed)
# --------------------------------------------------------------------------

def sigmoid(z):
    """σ(z) = 1 / (1 + exp(-z)). Element-wise. No scipy."""
    # YOUR CODE HERE
    pass


def forward_pass_vec(X, W, B):
    """
    Vectorised forward pass for all m examples.
    X: (NX, M), W: (NX, 1), B: scalar
    Returns Z (1, M) and A (1, M).
    No for-loops.
    """
    # YOUR CODE HERE
    pass


def compute_cost_vec(A, Y, M):
    """
    Average binary cross-entropy over M examples.
    Clip A to [1e-9, 1-1e-9] before log. No for-loops.
    Returns scalar cost.
    """
    # YOUR CODE HERE
    pass


def backward_pass_vec(X, A, Y, M):
    """
    Vectorised gradients for all M examples.
    Returns dW (NX, 1) and dB (scalar). No for-loops.
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Task 3 — Training loop
# --------------------------------------------------------------------------
# Initialise W = zeros (NX, 1), B = 0.0, ALPHA = 0.1, N_ITER = 1000.
#
# Loop for N_ITER iterations:
#   1. forward_pass_vec → Z, A
#   2. compute_cost_vec → cost; append to COSTS list
#   3. backward_pass_vec → dW, dB
#   4. Update W, B
#
# After training:
#   - Print: initial cost (COSTS[0]), final cost (COSTS[-1])
#   - Assert COSTS[-1] < COSTS[0]

W = np.zeros((NX, 1))
B = 0.0
ALPHA  = 0.1
N_ITER = 1000
COSTS  = []

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 4 — Evaluate accuracy and speed
# --------------------------------------------------------------------------
# 1. Final predictions:
#      A_final = sigmoid(W.T @ X + B)
#      Y_PRED  = (A_final >= 0.5).astype(int)
# 2. Accuracy = np.mean(Y_PRED == Y)
# 3. Print accuracy, number of correct predictions, confusion breakdown:
#      True positives  (Y==1 and Y_PRED==1)
#      True negatives  (Y==0 and Y_PRED==0)
#      False positives (Y==0 and Y_PRED==1)
#      False negatives (Y==1 and Y_PRED==0)
# 4. Time one full forward pass:
#      tic = time.time()
#      [run forward_pass_vec 1000 times in a loop]
#      toc = time.time()
#    Print average time per call in microseconds.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
# Two-panel figure:
#
#   Left panel: cost curve (COSTS vs iteration).
#     X-axis: "Iteration", Y-axis: "Cost J(w, b)"
#     Title: "CloudyDrive Obstacle Classifier — Training Cost"
#     Line colour: steelblue. Add dashed coral horizontal line at COSTS[-1]
#     labelled f"Final J = {COSTS[-1]:.4f}".
#
#   Right panel: scatter of feature 0 (Distance) vs feature 1 (Height).
#     X-axis: "Distance (norm)", Y-axis: "Height (norm)"
#     Colour each point by Y_PRED: steelblue=obstacle, coral=background.
#     Add a marker 'x' in black for misclassified examples.
#     Title: "Obstacle vs Background: Distance vs Height"
#     Add legend: "Predicted Obstacle", "Predicted Background", "Misclassified"
#
# Save to: ../images/05_vectorization_full_lr_classifier.png

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 3 your training loop has zero for-loops over m examples —
# the forward pass, cost, and backward pass are all matrix operations.
# Yet there is still one for-loop: the outer iteration loop.
# Why can't we also eliminate the iteration loop with vectorisation?
# What would need to be true about gradient descent for it to be
# computable in a single closed-form expression? Two to three sentences.

# YOUR ANSWER HERE
