# logistic_regression

**Course:** 01_neural_networks_and_dl
**Status:** ⬜ Not started
**Note:** This is the mathematical core. Every concept here reappears in every topic that follows.

---

## Notebooks

| # | File | Core concept | Status |
|---|------|-------------|--------|
| 01 | `01_binary_classification.ipynb` | Feature vectors, training notation, matrix stacking X(nx,m) Y(1,m) | ⬜ |
| 02 | `02_logistic_regression.ipynb` | Sigmoid, weights, bias, cross-entropy loss | ⬜ |
| 03 | `03_gradient_descent.ipynb` | Cost function, update rule, learning rate, convexity | ⬜ |
| 04 | `04_computation_graph_and_backprop.ipynb` | Forward/backward pass, chain rule, dZ = A-Y | ⬜ |
| 05 | `05_vectorization.ipynb` | Eliminating for-loops, broadcasting, full vectorized LR | ⬜ |

## Implement from scratch (benchmark)

- [ ] Sigmoid from scratch and its derivative σ(1-σ)
- [ ] Binary cross-entropy loss — formula and why not MSE
- [ ] Forward pass: Z = wᵀX + b, A = σ(Z)
- [ ] Backward pass: dZ = A-Y, dW = (1/m)X·dZᵀ, db = (1/m)·sum(dZ)
- [ ] One full gradient descent iteration — zero for-loops
