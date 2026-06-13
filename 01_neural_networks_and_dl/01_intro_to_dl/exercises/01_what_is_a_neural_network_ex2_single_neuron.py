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
    # Using np.where to conditionally return z or 0.0
    return np.where(z > 0, z, 0.0)


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
    # 1. Compute the weighted sum + bias
    z = np.dot(w, x) + b
    
    # 2. Apply activation function
    a = relu(z)
    
    return a, z


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

a_A, z_A = neuron_forward(x_A, w_A, b_A)

print("--- Task 1: Positive Pre-activation ---")
print(f"z: {z_A:.4f}  (Expected: 1.0500)")
print(f"a: {a_A:.4f}  (Expected: 1.0500)\n")


# --------------------------------------------------------------------------
# Task 2 — Case B: negative pre-activation (dead neuron)
# --------------------------------------------------------------------------
# Same x and w, but b = -5.0. Verify a = 0.0 using np.isclose.
# Print z and a. Add a comment explaining what "dead neuron" means here.
#
# Expected z: -3.9500   Expected a: 0.0000

b_B = -5.0

a_B, z_B = neuron_forward(x_A, w_A, b_B)

print("--- Task 2: Negative Pre-activation ---")
print(f"z: {z_B:.4f}  (Expected: -3.9500)")
print(f"a: {a_B:.4f}  (Expected: 0.0000)")
print(f"Is 'a' exactly 0.0? {np.isclose(a_B, 0.0)}")

# Comment: A "dead neuron" means the weighted sum plus bias resulted in a 
# negative number (z < 0), causing the ReLU activation to clamp the output to zero. 
# This neuron essentially blocks the signal and contributes nothing to the next layer.
print()


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

a_5d, z_5d = neuron_forward(x_5d, w_5d, b_5d)

print("--- Task 3: 5D Shape Check ---")
print(f"Dimensions of z: {np.ndim(z_5d)} (Type: {type(z_5d)})")
print(f"Output a: {a_5d:.4f}\n")


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Case B, a = 0.0 even though the input x is non-zero.
# What does this tell you about the relationship between input magnitude
# and neuron output? (Two sentences.)

# The output is determined by the linear combination of inputs (the weighted sum) and the bias, 
# not just the raw magnitude of the input data. If negative weights or a strongly negative bias dominate the equation,
# they pull the pre-activation below zero, effectively erasing the input's influence entirely.