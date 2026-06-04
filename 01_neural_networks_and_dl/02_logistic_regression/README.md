# Week 2 — Logistic Regression as a Neural Network

**Section path:** `01_neural_networks_and_dl/week_02_logistic_regression/`
**Status:** ⬜ Not started
**Note:** This week is the mathematical core. Every concept here reappears in every week that follows.

---

## Notebooks

| # | File | Core concept | Status |
|---|------|-------------|--------|
| 01 | `01_binary_classification.ipynb` | Feature vectors, training notation, matrix stacking | ⬜ |
| 02 | `02_logistic_regression.ipynb` | Sigmoid, weights, bias, cross-entropy loss | ⬜ |
| 03 | `03_gradient_descent.ipynb` | Cost function, update rule, learning rate, convexity | ⬜ |
| 04 | `04_computation_graph_and_backprop.ipynb` | Forward/backward pass, chain rule, dZ = A-Y | ⬜ |
| 05 | `05_vectorization.ipynb` | Eliminating for-loops, broadcasting, full vectorized LR | ⬜ |

## Key concepts to be able to implement from scratch after this week

- [ ] Sigmoid from scratch and its derivative
- [ ] Binary cross-entropy loss: formula and why not MSE
- [ ] Forward propagation: Z = wᵀX + b, A = σ(Z)
- [ ] Backward propagation: dZ = A - Y, dW = (1/m)X·dZᵀ, db = (1/m)·sum(dZ)
- [ ] Vectorized gradient descent: one iteration, zero for-loops
