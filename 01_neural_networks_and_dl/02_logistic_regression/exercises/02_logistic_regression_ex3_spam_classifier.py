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
    return 1 / (1 + np.exp(-z))


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

assert X_EMAILS.shape == (NX, M), f"Expected ({NX}, {M}), got {X_EMAILS.shape}"
assert Y_TRUE.shape   == (1,  M), f"Expected (1, {M}), got {Y_TRUE.shape}"
print(f"X_EMAILS.shape : {X_EMAILS.shape}")
print(f"Y_TRUE.shape   : {Y_TRUE.shape}")


# --------------------------------------------------------------------------
# Task 2 — Full forward pass
# --------------------------------------------------------------------------
Z = W.T @ X_EMAILS + B     # shape (1, M)
A = sigmoid(Z)              # shape (1, M)

assert np.all(A > 0) and np.all(A < 1), "Probabilities must be in (0, 1)"

print(f"\nZ (3 dp) : {Z.round(3)}")
print(f"A (3 dp) : {A.round(3)}")


# --------------------------------------------------------------------------
# Task 3 — Threshold and compute accuracy
# --------------------------------------------------------------------------
Y_PRED   = (A >= 0.5).astype(int)                  # shape (1, M)
ACCURACY = np.mean(Y_PRED == Y_TRUE)
n_correct = int(np.sum(Y_PRED == Y_TRUE))

print(f"\nY_PRED    : {Y_PRED}")
print(f"Y_TRUE    : {Y_TRUE}")
print(f"Accuracy  : {ACCURACY:.2f}")
print(f"Correct   : {n_correct} / {M}")


# --------------------------------------------------------------------------
# Task 4 — Manual verification for one example
# --------------------------------------------------------------------------
# Manual calculation for email index 0:
# z = 1.2*(0.9) + (-0.8)*(0.8) + 0.5*(0.7) + 0.3*(0.6) + (-1.0)*(0.9) + (-0.4)
#   = 1.08  +  (-0.64)  +  0.35  +  0.18  +  (-0.90)  +  (-0.40)
#   = 1.08 - 0.64 + 0.35 + 0.18 - 0.90 - 0.40
#   = -0.33

manual_z0 = (1.2 * 0.9) + (-0.8 * 0.8) + (0.5 * 0.7) + (0.3 * 0.6) + (-1.0 * 0.9) + (-0.4)
print(f"\nManual z for email 0 : {manual_z0:.4f}")
print(f"Z[0, 0]              : {Z[0, 0]:.4f}")
print(f"Match                : {np.isclose(float(Z[0, 0]), manual_z0)}")   # True


# --------------------------------------------------------------------------
# Task 5 — Visualisation
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

email_labels = [f"Email {i+1}" for i in range(M)]
bar_colors   = ["steelblue" if Y_TRUE[0, i] == 1 else "coral" for i in range(M)]

# Left panel — horizontal bar chart of P(spam)
axes[0].barh(email_labels, A[0, :], color=bar_colors)
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="decision boundary")
axes[0].set_xlabel("P(spam)")
axes[0].set_xlim(0, 1)
axes[0].set_title("Spam Classifier: Predicted P(spam) per Email")
axes[0].legend()
axes[0].invert_yaxis()

# Right panel — sigmoid curve with email z-values overlaid
z_range   = np.linspace(-4, 4, 300)
axes[1].plot(z_range, sigmoid(z_range), color="black", linewidth=1.5, label="σ(z)")
axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1.0)

spam_mask = Y_TRUE[0, :] == 1
axes[1].scatter(Z[0, spam_mask],  A[0, spam_mask],  color="steelblue",
                zorder=5, label="Spam (y=1)")
axes[1].scatter(Z[0, ~spam_mask], A[0, ~spam_mask], color="coral",
                zorder=5, label="Real (y=0)")

axes[1].set_xlabel("z = wᵀx + b")
axes[1].set_ylabel("σ(z)")
axes[1].set_title("Sigmoid Output per Email")
axes[1].legend()

plt.tight_layout()
plt.savefig("../images/02_logistic_regression_predictions.png", dpi=150)
plt.show()
print("Plot saved to ../images/02_logistic_regression_predictions.png")


# --------------------------------------------------------------------------
# Answer in a comment
# --------------------------------------------------------------------------
# In Task 3, some emails may be misclassified even though your weights are
# reasonable. Why is the decision boundary (threshold at 0.5) not always
# the optimal operating point for a spam classifier? What real-world
# consideration might make you choose a different threshold?
# Two to three sentences.

# The threshold at 0.5 treats false positives (real email flagged as spam)
# and false negatives (spam reaching the inbox) as equally costly, but in
# practice those costs differ: a user losing an important email to the spam
# folder is often more disruptive than letting occasional spam through.
# Lowering the threshold raises recall for spam but increases false positives,
# while raising it protects real emails at the cost of missing more spam —
# so the right operating point depends on the relative cost of each error type
# and is typically tuned using a precision-recall or ROC curve on a validation set.
