"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 02_logistic_regression — Ex 3 of 3
Topic   : Full logistic regression prediction pipeline — spam classifier
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
Scenario: you are building a spam classifier. Each email is represented by
nₓ = 5 features (word count, exclamation marks, links, uppercase ratio,
known-spam-word count), all normalised to [0, 1]. You have a pre-trained
weight vector and bias. Your job is to run the complete logistic regression
forward pass on a batch of 10 emails, threshold predictions at 0.5, compute
classification accuracy, and plot the output probability distribution.
This mini-project assembles every piece of notebook 02 into one working
pipeline.

Rules
--------------------------------------------------------------------------
- Do not use scipy.special.expit — implement sigmoid from np.exp.
- Use np.allclose for array comparisons — never a == b.
- All probability outputs must be in (0, 1) — assert this explicitly.
- Save plot to: ../images/02_logistic_regression_predictions.png
- Run top to bottom with zero errors before committing.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Task 1 — Implement sigmoid and build the dataset
# --------------------------------------------------------------------------

def sigmoid(z):
    """σ(z) = 1 / (1 + exp(-z)). Works element-wise. Do not use scipy."""
    # YOUR CODE HERE
    pass


# Provided pre-trained weights and dataset (do not change):
np.random.seed(0)
NX = 5
M  = 10

W = np.array([[ 1.2],
              [-0.8],
              [ 0.5],
              [ 0.3],
              [-1.0]])    # shape (5, 1)
B = -0.4                  # scalar bias

X_EMAILS = np.array([
    [0.9, 0.1, 0.8, 0.2, 0.7, 0.0, 0.6, 0.3, 0.95, 0.05],   # word count (norm)
    [0.8, 0.0, 0.9, 0.1, 0.6, 0.0, 0.7, 0.2, 0.85, 0.0 ],   # exclamations
    [0.7, 0.2, 0.8, 0.3, 0.5, 0.1, 0.6, 0.4, 0.9,  0.1 ],   # links
    [0.6, 0.1, 0.7, 0.2, 0.4, 0.0, 0.5, 0.3, 0.8,  0.05],   # uppercase ratio
    [0.9, 0.0, 0.8, 0.1, 0.7, 0.0, 0.6, 0.2, 0.95, 0.0 ],   # spam words
])  # shape (5, 10)

Y_TRUE = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])   # true labels, shape (1, 10)

# Verify dataset shapes — assert X_EMAILS.shape == (NX, M) and Y_TRUE.shape == (1, M).
# Print both shapes.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Full forward pass
# --------------------------------------------------------------------------
# Steps:
#   1. Z = W.T @ X_EMAILS + B          shape must be (1, M)
#   2. A = sigmoid(Z)                   shape must be (1, M)
#   3. Assert all values in A are in (0, 1):
#      assert np.all(A > 0) and np.all(A < 1)
#   4. Print Z (rounded to 3 dp) and A (rounded to 3 dp).

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Threshold and compute accuracy
# --------------------------------------------------------------------------
# Convert probabilities to binary predictions:
#   Y_PRED = (A >= 0.5).astype(int)    shape (1, M)
#
# Compute accuracy:
#   ACCURACY = np.mean(Y_PRED == Y_TRUE)
#
# Print Y_PRED, Y_TRUE, and ACCURACY.
# Also print the number of correct predictions.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 4 — Manual verification for one example
# --------------------------------------------------------------------------
# For the first email (index 0), verify z and ŷ by hand.
#
# Manual calculation (fill in the arithmetic):
# z = 1.2*0.9 + (-0.8)*0.8 + 0.5*0.7 + 0.3*0.6 + (-1.0)*0.9 + (-0.4)
#   = ? + ? + ? + ? + ? + ?  =  ?
#
# Write the step-by-step arithmetic as a comment.
# Then verify your manual z matches Z[0, 0] using np.isclose.
# Print True/False.

# Manual z for email 0:
# z = 1.2*(0.9) + (-0.8)*(0.8) + 0.5*(0.7) + 0.3*(0.6) + (-1.0)*(0.9) + (-0.4)
#   = ? + ? + ? + ? + ? + ? = ?

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
# Two-panel figure:
#
#   Left panel: horizontal bar chart of predicted probabilities A[0, :].
#     Y-axis: email index labels ["Email 1", "Email 2", ..., "Email 10"]
#     X-axis: "P(spam)"  range [0, 1]
#     Colour each bar: steelblue if Y_TRUE[0,i]==1 else coral
#     Add vertical dashed line at x=0.5 labelled "decision boundary"
#     Title: "Spam Classifier: Predicted P(spam) per Email"
#
#   Right panel: sigmoid curve for the range z ∈ [−4, 4].
#     Plot σ(z) vs z. Mark the 10 email z-values as scatter points.
#     Colour each point: steelblue if true label==1 else coral
#     Horizontal dashed grey line at y=0.5.
#     X-axis: "z = wᵀx + b", Y-axis: "σ(z)", Title: "Sigmoid Output per Email"
#     Add legend: "Spam (y=1)" and "Real (y=0)"
#
# Save to: ../images/02_logistic_regression_predictions.png

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 3, some emails may be misclassified even though your weights are
# reasonable. Why is the decision boundary (threshold at 0.5) not always
# the optimal operating point for a spam classifier? What real-world
# consideration might make you choose a different threshold?
# Two to three sentences.

# YOUR ANSWER HERE
