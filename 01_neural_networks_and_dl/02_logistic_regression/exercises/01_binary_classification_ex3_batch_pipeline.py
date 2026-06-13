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
np.random.seed(42)
m = 16
H, W, C = 48, 48, 3
FRAMES = np.random.randint(0, 256, size=(m, H, W, C))    # raw batch
LABELS_RAW = np.array([1]*8 + [0]*8)                     # 8 deepfake, 8 real

NX = H * W * C                                  # 6912
X  = FRAMES.reshape(m, -1).T                    # shape (NX, m) — one expression
Y  = LABELS_RAW.reshape(1, m)                   # shape (1, m)

print(f"NX      : {NX}")
print(f"X.shape : {X.shape}")    # Expected (6912, 16)
print(f"Y.shape : {Y.shape}")    # Expected (1, 16)


# --------------------------------------------------------------------------
# Task 2 — Normalise pixel values to [0, 1]
# --------------------------------------------------------------------------
X_norm = X / 255                                 # vectorised, no loop

print(f"X_norm.min() ≈ 0.0 : {np.allclose(X_norm.min(), 0.0, atol=0.01)}")
print(f"X_norm.max() ≈ 1.0 : {np.allclose(X_norm.max(), 1.0, atol=0.01)}")


# --------------------------------------------------------------------------
# Task 3 — Verify dataset integrity
# --------------------------------------------------------------------------
assert X_norm.shape == (NX, m),  f"Expected ({NX}, {m}), got {X_norm.shape}"
print(f"X_norm.shape == ({NX}, {m})   : True")

assert Y.shape == (1, m),        f"Expected (1, {m}), got {Y.shape}"
print(f"Y.shape == (1, {m})         : True")

n_deepfake = int(np.sum(Y == 1))
n_real     = int(np.sum(Y == 0))
print(f"Deepfake labels (Y=1) : {n_deepfake}  (expected 8)")
print(f"Real labels     (Y=0) : {n_real}  (expected 8)")

MEAN_VEC = X_norm.mean(axis=1)                   # shape (NX,)
print(f"MEAN_VEC.shape        : {MEAN_VEC.shape}")
print(f"mean of MEAN_VEC      : {MEAN_VEC.mean():.4f}  (expected ≈ 0.5)")
print(f"MEAN_VEC.mean() ≈ 0.5 : {np.isclose(MEAN_VEC.mean(), 0.5, atol=0.05)}")


# --------------------------------------------------------------------------
# Task 4 — Retrieve and inspect one example
# --------------------------------------------------------------------------
x5       = X_norm[:, 4:5]                        # shape (NX, 1) — column slice
x5_frame = x5.reshape(H, W, C)                  # shape (H, W, C)

print(f"x5.shape       : {x5.shape}")
print(f"x5_frame.shape : {x5_frame.shape}")
assert x5.shape == (NX, 1), f"Expected ({NX}, 1), got {x5.shape}"
print(f"reshape round-trip matches: {np.allclose(x5.reshape(H, W, C), x5_frame)}")


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left panel — class distribution bar chart
axes[0].bar(
    ["Deepfake (y=1)", "Real (y=0)"],
    [n_deepfake, n_real],
    color=["steelblue", "coral"],
)
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].set_title("WaveMamba-DF Batch: Class Distribution")

# Right panel — histogram of per-feature means
axes[1].hist(MEAN_VEC, bins=40, color="steelblue", edgecolor="white")
axes[1].axvline(0.5, color="red", linestyle="--", label="expected ≈ 0.5")
axes[1].set_xlabel("Feature Mean (normalised pixel)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Feature Means Across nₓ Features")
axes[1].legend()

plt.tight_layout()
plt.savefig("../images/01_binary_classification_batch_stats.png", dpi=150)
plt.show()
print("Plot saved to ../images/01_binary_classification_batch_stats.png")


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 2 you normalised inputs to [0, 1] by dividing by 255.
# Why does this normalisation help gradient descent converge faster?
# What would happen to the gradient updates if pixel values stayed in [0, 255]?
# Two to three sentences.

# When pixel values stay in [0, 255] the gradients with respect to the weights
# are proportional to those large input magnitudes, so weight updates become
# enormous and the loss surface is poorly scaled — different features may have
# wildly different gradient scales, forcing a tiny learning rate and slow convergence.
# Normalising to [0, 1] keeps all input dimensions on the same scale, so gradient
# descent takes balanced steps in every direction and converges in far fewer iterations
# without requiring a very small or carefully tuned learning rate.
