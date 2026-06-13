"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 03_gradient_descent — Ex 1 of 3
Topic   : Binary cross-entropy loss from scratch
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
The loss function is the signal that drives all of training. Writing binary
cross-entropy from scratch — with proper numerical clipping — makes concrete
why confident wrong predictions are penalised so heavily (logarithm near 0
blows up), and why a clipping step is non-optional. After this exercise you
will be able to implement and verify the loss without looking anything up,
and explain why it is preferred over mean squared error for classification.

Rules
--------------------------------------------------------------------------
- Clip ŷ to [1e-9, 1 − 1e-9] before any np.log call — never log(0).
- Use np.isclose for scalar comparisons — never == on floats.
- Do not use sklearn.metrics.log_loss — implement from the formula.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

# --------------------------------------------------------------------------
# Task 1 — Implement compute_loss for a single example
# --------------------------------------------------------------------------

def compute_loss(y_hat, y):
    """
    Binary cross-entropy loss for one training example.
    Formula: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    Clip ŷ to [1e-9, 1 - 1e-9] before computing log.
    y_hat: scalar predicted probability in (0, 1)
    y: scalar true label, 0 or 1
    Returns: scalar loss value >= 0
    """
    # YOUR CODE HERE
    pass


# Verify against four canonical cases.
# Fill in the expected values by hand before checking.
#
# Case 1: y=1, y_hat=0.99  → L = -log(0.99) ≈ ?
# Case 2: y=1, y_hat=0.01  → L = -log(0.01) ≈ ?
# Case 3: y=0, y_hat=0.01  → L = -log(0.99) ≈ ?
# Case 4: y=0, y_hat=0.99  → L = -log(0.01) ≈ ?
#
# Print all four computed losses.
# Verify Case 1 ≈ 0.010 with np.isclose(compute_loss(0.99, 1), 0.010, atol=1e-3).
# Verify Case 4 ≈ 4.605 with np.isclose(compute_loss(0.99, 0), 4.605, atol=1e-3).

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Verify loss is always non-negative and correct for boundary cases
# --------------------------------------------------------------------------
# Test these inputs and print results:
#   y_hat very close to 0 (=1e-8), y=1  → should be very large (not inf)
#   y_hat very close to 1 (=1-1e-8), y=1 → should be near 0
#   y_hat = 0.5, y=1 → should equal log(2) ≈ 0.6931
#
# Verify the 0.5 case: np.isclose(compute_loss(0.5, 1), np.log(2))
# Print True/False.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# Binary cross-entropy penalises a confident wrong prediction (e.g. y=1, ŷ=0.01)
# with loss ≈ 4.605, but a correct prediction (y=1, ŷ=0.99) with loss ≈ 0.010.
# Why does this asymmetric penalty make cross-entropy a better loss function
# for classification than mean squared error? One sentence.

# YOUR ANSWER HERE
