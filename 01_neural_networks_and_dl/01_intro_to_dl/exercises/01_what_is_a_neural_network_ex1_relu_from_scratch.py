"""
Repo    : AI_Engineer-DL
Section : 01_neural_networks_and_dl/01_intro_to_dl
Exercise: 01_what_is_a_neural_network — Ex 1 of 3
Topic   : ReLU from scratch — implementation and derivative
Author  : Abdel Rahman Madboly

--------------------------------------------------------------------------
What this exercise is about
--------------------------------------------------------------------------
ReLU (Rectified Linear Unit) is the default activation function for hidden
layers in modern neural networks. Writing it from scratch — without reaching
for np.maximum — forces you to see exactly what it computes: a comparison
and a conditional zero. Understanding this operation cold is critical because
during backpropagation you will compute its derivative by hand, and if you
only ever used np.maximum, that step will feel arbitrary. After this exercise
you will be able to implement both ReLU and its derivative from memory.

Rules
--------------------------------------------------------------------------
- Do NOT use np.maximum inside relu(). Use np.where or a boolean comparison.
- You MAY use np.maximum in the verification step only.
- Use np.allclose for all array comparisons — never ==.
- Run top to bottom with zero errors before committing.
"""

import numpy as np


# --------------------------------------------------------------------------
# Function definition — relu
# --------------------------------------------------------------------------

def relu(z):
    """
    Rectified Linear Unit applied element-wise.
    Formula: max(0, z) for each element.
    Constraint: do NOT use np.maximum inside this function.
    Works on both scalars and NumPy arrays of any shape.
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Function definition — relu_derivative
# --------------------------------------------------------------------------

def relu_derivative(Z):
    """
    Derivative of ReLU applied element-wise.
    Returns 1.0 where Z > 0, else 0.0.
    Constraint: no Python for-loops — must be fully vectorized.
    """
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Task 1 — Test relu on a range of values
# --------------------------------------------------------------------------
# Apply your relu to z_values and verify it matches np.maximum(0, z_values).
# Print both results and whether they match.
#
# Expected relu output: [0.  0.  0.  0.5 1.5 4. ]

z_values = np.array([-3.0, -1.0, 0.0, 0.5, 1.5, 4.0])

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 2 — Test relu_derivative on the same array
# --------------------------------------------------------------------------
# Apply your relu_derivative to z_values.
# Print the result.
# Verify it equals np.array([0., 0., 0., 1., 1., 1.]) using np.allclose.
#
# Expected: [0. 0. 0. 1. 1. 1.]

# YOUR CODE HERE


# --------------------------------------------------------------------------
# Task 3 — Scalar test
# --------------------------------------------------------------------------
# Test relu and relu_derivative on individual scalars.
# For each value below, print relu(val) and relu_derivative(np.array([val])).
#
# Values to test: -2.5, 0.0, 3.7

for val in [-2.5, 0.0, 3.7]:
    # YOUR CODE HERE
    pass


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# Why is the ReLU derivative undefined at exactly z = 0, and why does it
# not matter in practice during neural network training?
# (Two sentences.)

# YOUR ANSWER HERE
