"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 03_gradient_descent — Ex 3 of 3
Topic   : Full training loop — learning rate sensitivity study
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Scenario: you are training a logistic regression model to classify sensor
readings from FarmLens — two features (leaf colour score, texture score)
predict whether a crop is diseased (y=1) or healthy (y=0). Your job is to
implement the complete training loop, study how the learning rate α affects
convergence, and visualise the cost curves. This is the smallest complete
version of the training pipeline that every neural network uses: forward →
cost → backward → update → repeat.

Rules
--------------------------------------------------------------------------
- Use np.allclose for array comparisons and np.isclose for scalars.
- Clip probabilities to [1e-9, 1-1e-9] before any log.
- No for-loops over examples — all forward/backward steps must be vectorised.
- Save plot to: ../images/03_gradient_descent_lr_sensitivity.png
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# Task 1 — Dataset setup
# --------------------------------------------------------------------------
# Provided crop health dataset (do not modify):
np.random.seed(5)
NX = 2    # features: leaf colour score, texture score
M  = 30   # training examples

X_CROP = np.vstack([
    np.random.randn(1, 15) * 0.5 + 1.5,   # diseased: colour score ≈ 1.5
    np.random.randn(1, 15) * 0.5 - 1.5,   # healthy:  colour score ≈ -1.5
])                                          # shape (2, 30) — note: already (NX, M)

# Replace the vstack with properly shaped arrays:
X_CROP = np.hstack([
    np.vstack([np.random.randn(1, 15) * 0.5 + 1.5,
               np.random.randn(1, 15) * 0.5 + 0.5]),  # diseased
    np.vstack([np.random.randn(1, 15) * 0.5 - 1.5,
               np.random.randn(1, 15) * 0.5 - 0.5]),  # healthy
])  # shape (2, 30)

Y_CROP = np.hstack([np.ones((1, 15)), np.zeros((1, 15))])  # shape (1, 30)

# Print X_CROP.shape and Y_CROP.shape.
# Assert X_CROP.shape == (NX, M) and Y_CROP.shape == (1, M).

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Implement the training loop function
# --------------------------------------------------------------------------

def train_logistic_regression(X, Y, alpha, n_iterations):
    """
    Full gradient descent training loop for logistic regression.
    Initialise W to zeros (shape: nₓ, 1) and B to 0.0.
    Run n_iterations steps of vectorised gradient descent.
    Return: (W, B, costs) where costs is a list of cost values, one per iteration.
    No for-loops over examples — only a for-loop over iterations.
    Clip predictions to [1e-9, 1-1e-9] before log.
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Task 3 — Train with three learning rates and compare
# --------------------------------------------------------------------------
# Train with: alpha = 0.001, 0.01, 0.1
# n_iterations = 500 for all three.
#
# For each alpha:
#   1. Call train_logistic_regression(X_CROP, Y_CROP, alpha, 500)
#   2. Store the returned costs list.
#   3. Print: alpha, initial cost (costs[0]), final cost (costs[-1])
#
# Store results for plotting in Task 5.

ALPHAS = [0.001, 0.01, 0.1]
RESULTS = {}   # key: alpha, value: (W, B, costs)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 4 — Evaluate final model accuracy
# --------------------------------------------------------------------------
# For the best-converging alpha (0.1):
#   1. Run the final forward pass: A_final = sigmoid(W.T @ X_CROP + B)
#   2. Threshold at 0.5 to get binary predictions Y_PRED.
#   3. Compute accuracy = np.mean(Y_PRED == Y_CROP).
#   4. Print Y_PRED (shape 1×30) and accuracy.
#
# Use RESULTS[0.1] to get W and B for the best model.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
# Two-panel figure:
#
#   Left panel: cost curves for all three learning rates on the same axes.
#     X-axis: "Iteration"
#     Y-axis: "Cost J(w, b)"
#     Title: "FarmLens LR: Cost vs Iteration for 3 Learning Rates"
#     One line per alpha, labelled "α = 0.001", "α = 0.01", "α = 0.1"
#     Use colours: ['coral', 'steelblue', 'seagreen']
#     Add a legend.
#
#   Right panel: scatter plot of the training data.
#     X-axis: first feature X_CROP[0, :]  labelled "Leaf Colour Score"
#     Y-axis: second feature X_CROP[1, :] labelled "Texture Score"
#     Colour: steelblue for diseased (y=1), coral for healthy (y=0)
#     Title: "Crop Health Dataset (NX=2, M=30)"
#     Add legend: "Diseased (y=1)" and "Healthy (y=0)"
#
# Save to: ../images/03_gradient_descent_lr_sensitivity.png

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 3 you tested α = 0.001, 0.01, and 0.1. The cost curve for α = 0.001
# likely converged much slower than α = 0.1. Why can't you simply always use a
# very large learning rate to converge faster? What happens to the cost curve
# when α is too large? Two to three sentences.

# YOUR ANSWER HERE
