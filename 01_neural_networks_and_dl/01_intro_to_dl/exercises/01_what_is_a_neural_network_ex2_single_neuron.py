"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/01_intro_to_dl
Exercise: 01_what_is_a_neural_network — Ex 2 of 3
Topic   : Single neuron forward pass — weighted sum, bias, ReLU
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
A single neuron does two things: (1) computes a weighted sum of its inputs
plus a bias, and (2) applies an activation function. Writing this from scratch
makes concrete the difference between z (the pre-activation, which can be
any real number) and a (the activation, which is always non-negative after
ReLU). You will also see what happens when z is negative — the neuron outputs
zero, contributing nothing to the next layer. After this exercise you will be
able to implement the forward pass of one neuron in any framework without
looking anything up.

Rules
--------------------------------------------------------------------------
- Do NOT use np.maximum inside neuron_forward(). Use your own relu().
- Use np.dot for the weighted sum — no manual element-wise loops.
- Use np.isclose for scalar float comparisons — never ==.
- Run top to bottom with zero errors before committing.
"""

import numpy as np


# --------------------------------------------------------------------------
# Helper — copy your relu from Ex 1 (do not import it)
# --------------------------------------------------------------------------

def relu(z):
    """
    ReLU applied element-wise. max(0, z). No np.maximum allowed.
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Function definition — neuron_forward
# --------------------------------------------------------------------------

def neuron_forward(x, w, b):
    """
    Forward pass of a single neuron.
    Computes z = wᵀx + b, then a = relu(z).
    Returns (a, z) — both values, because z is needed for backprop later.
    Constraint: use np.dot for the weighted sum; use your relu for activation.
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Task 1 — Case A: positive pre-activation
# --------------------------------------------------------------------------
# Run the neuron on Case A and verify a > 0.
# Print z and a rounded to 4 decimal places.
#
# Expected z: 1.0500   Expected a: 1.0500

x_A = np.array([1.5, -0.5, 2.0])
w_A = np.array([0.3, -0.2, 0.5])
b_A = -0.1

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Case B: negative pre-activation (dead neuron)
# --------------------------------------------------------------------------
# Same x and w, but b = -5.0. Verify a = 0.0 using np.isclose.
# Print z and a. Add a comment explaining what "dead neuron" means here.
#
# Expected z: -3.9500   Expected a: 0.0000

b_B = -5.0

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Shape check
# --------------------------------------------------------------------------
# Create a 5-dimensional input and verify the neuron still works correctly.
# Use any w with shape (5,) and b scalar of your choice.
# Print the shape of z (should be a scalar — use np.ndim or type check).
# Print the output a.

x_5d = np.array([0.5, -1.0, 2.0, 0.0, -0.3])
w_5d = np.array([0.1, 0.4, -0.2, 0.6, 0.3])
b_5d = 0.05

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Case B, a = 0.0 even though the input x is non-zero.
# What does this tell you about the relationship between input magnitude
# and neuron output? (Two sentences.)

# YOUR ANSWER HERE
