"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 02_logistic_regression — Ex 1 of 3
Topic   : Sigmoid function and its derivative from scratch
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
The sigmoid function is the mathematical core of logistic regression and
the output layer of any binary classifier. Writing it from scratch — rather
than calling scipy — forces you to internalise both the formula and its
numerical properties. Its derivative, σ(z)·(1 − σ(z)), reappears every
time you implement backpropagation through a sigmoid layer. After this
exercise you will be able to write both from memory and verify them against
a reference implementation.

Rules
--------------------------------------------------------------------------
- Do not use scipy.special.expit — implement using np.exp only.
- Verify all results with np.allclose — never == on floats.
- Run top to bottom with zero errors before committing.
"""

import numpy as np

# --------------------------------------------------------------------------
# Task 1 — Implement sigmoid(z)
# --------------------------------------------------------------------------

def sigmoid(z):
    """
    Compute the sigmoid of z element-wise.
    Formula: σ(z) = 1 / (1 + exp(-z))
    Works on scalars, 1-D arrays, and 2-D arrays.
    Do not use scipy.special.expit — use np.exp only.
    """
    # YOUR CODE HERE
    pass


# Verify against scipy
from scipy.special import expit

TEST_VALUES = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
scratch_out = sigmoid(TEST_VALUES)
library_out = expit(TEST_VALUES)

# Print both arrays and the allclose result.
# Expected: all np.allclose checks print True.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Implement sigmoid_derivative(z)
# --------------------------------------------------------------------------

def sigmoid_derivative(z):
    """
    Compute dσ/dz = σ(z) · (1 − σ(z)) element-wise.
    You may call sigmoid() inside this function.
    Do not re-derive from scratch using np.exp directly.
    """
    # YOUR CODE HERE
    pass


# Test and verify:
#   sigmoid_derivative(0)  → expected exactly 0.25
#   sigmoid_derivative(2)  → expected ~0.1050
#   sigmoid_derivative(-2) → expected ~0.1050  (symmetric)
# Verify each with np.isclose. Print True/False for each.

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# The sigmoid derivative equals σ(z)·(1 − σ(z)), which reaches its maximum
# value of 0.25 at z = 0 and shrinks toward 0 for large |z|.
# Why does this shrinking gradient cause a problem in deep networks that use
# sigmoid activations in every hidden layer? One sentence.

# YOUR ANSWER HERE
