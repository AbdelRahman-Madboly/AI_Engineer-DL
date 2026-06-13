"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/logistic_regression
Exercise: 04_computation_graph_and_backprop — Ex 2 of 3
Topic   : Full forward + backward pass with gradient checking
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
A single-example forward/backward pass is the atomic unit of neural network
training. This exercise puts both together: you run the computation graph
forward, cache intermediate values, then walk it backwards to compute all
gradients. You then verify dW with finite differences — the same check
PyTorch performs internally. This combination of analytical derivation and
numerical verification is a core debugging skill for any deep learning
engineer.

Rules
--------------------------------------------------------------------------
- Use np.isclose(analytical, numerical, atol=1e-5) for gradient checks.
- Do not use a loop over weights in the finite difference check — check the
  first weight w[0, 0] only (full gradient check is for Ex 3).
- Run top to bottom with zero errors before committing.
"""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(x, w, b, y):
    Z = float(w.T @ x + b)
    A = float(sigmoid(Z))
    L = -(y * np.log(A + 1e-9) + (1 - y) * np.log(1 - A + 1e-9))
    return {'Z': Z, 'A': A, 'L': float(L)}

def backward_pass(cache, x, y):
    A  = cache['A']
    dZ = A - y
    dW = dZ * x
    db = dZ
    return {'dZ': dZ, 'dW': dW, 'db': db}


# --------------------------------------------------------------------------
# Task 1 — Run full forward + backward on a 4-feature example
# --------------------------------------------------------------------------
# Provided:
np.random.seed(99)
NX = 4
x = np.random.randn(NX, 1)
w = np.array([[0.2], [-0.1], [0.4], [-0.3]])
b = 0.05
y = 1
#
# Steps:
#   1. cache = forward_pass(x, w, b, y)
#   2. grads = backward_pass(cache, x, y)
#   3. Print: Z, A, L (all from cache), then dZ, dW, db (from grads).
#   4. Assert grads['dW'].shape == (NX, 1).
#   5. Interpret: is dZ large or small? Why? (answer depends on A vs y)

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Finite difference check for the first weight w[0, 0]
# --------------------------------------------------------------------------
# Perturb w[0, 0] by ±EPSILON and compute the numerical gradient.
#
# EPSILON = 1e-5
#
# Steps:
#   1. Build w_plus  = copy of w with w[0,0] increased by EPSILON.
#   2. Build w_minus = copy of w with w[0,0] decreased by EPSILON.
#   3. L_plus  = forward_pass(x, w_plus,  b, y)['L']
#   4. L_minus = forward_pass(x, w_minus, b, y)['L']
#   5. fd_dw0 = (L_plus - L_minus) / (2 * EPSILON)
#   6. analytical_dw0 = float(grads['dW'][0, 0])
#   7. Verify np.isclose(fd_dw0, analytical_dw0, atol=1e-5). Print True/False.
#   8. Print both values side by side (round to 8 dp).
#
# Hint (read only if stuck)
# --------------------------------------------------------------------------
# Use w.copy() before modifying — never modify w in-place when you need the
# original for the second perturbed pass.

EPSILON = 1e-5

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Interpret the gradient
# --------------------------------------------------------------------------
# Print the answer to this question as a comment immediately below.
# If A (the prediction) is 0.8 and y=1, dZ = A − y = −0.2.
# dW = dZ * x means each weight gradient is negative (assuming positive x).
# The update w := w − α·dW will therefore INCREASE those weights.
# Does this make sense? Why should the weights increase when the model
# underpredicts a positive example?
# Write two to three sentences explaining the direction of the update.

# YOUR CODE HERE (print explanation as a comment or string)


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# Finite difference gradient checking costs O(nₓ) forward passes (one per
# weight) — expensive for large networks. Why is it used during development
# but never during production training? Two sentences.

# YOUR ANSWER HERE
