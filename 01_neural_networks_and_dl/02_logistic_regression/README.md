# logistic_regression

**Course:** 01_neural_networks_and_dl
**Status:** ⏳ In progress
**Note:** This is the mathematical core. Every concept here reappears in every topic that follows.

---

## Notebooks

| # | File | Core concept | Status |
|---|------|-------------|--------|
| 01 | `01_binary_classification.ipynb` | Feature vectors, training notation, matrix stacking X(nₓ,m) Y(1,m) | ⬜ |
| 02 | `02_logistic_regression.ipynb` | Sigmoid, weights, bias, hypothesis ŷ = σ(wᵀx + b) | ⬜ |
| 03 | `03_gradient_descent.ipynb` | Cost function, update rule, learning rate, convexity | ⬜ |
| 04 | `04_computation_graph_and_backprop.ipynb` | Forward/backward pass, chain rule, dZ = A−Y | ⬜ |
| 05 | `05_vectorization.ipynb` | Eliminating for-loops, broadcasting, full vectorised LR | ⬜ |

## Notes

| # | File | Status |
|---|------|--------|
| 01 | `01_binary_classification_notes.md` | ⬜ |
| 02 | `02_logistic_regression_notes.md` | ⬜ |
| 03 | `03_gradient_descent_notes.md` | ⬜ |
| 04 | `04_computation_graph_and_backprop_notes.md` | ⬜ |
| 05 | `05_vectorization_notes.md` | ⬜ |

## Exercises

| # | File | Status |
|---|------|--------|
| 01 | `01_binary_classification_ex1_feature_vector.py` | ⬜ |
| 01 | `01_binary_classification_ex2_build_matrices.py` | ⬜ |
| 01 | `01_binary_classification_ex3_dataset_pipeline.py` | ⬜ |
| 02 | `02_logistic_regression_ex1_sigmoid.py` | ⬜ |
| 02 | `02_logistic_regression_ex2_predictions.py` | ⬜ |
| 02 | `02_logistic_regression_ex3_decision_boundary.py` | ⬜ |
| 03 | `03_gradient_descent_ex1_cross_entropy.py` | ⬜ |
| 03 | `03_gradient_descent_ex2_learning_rate.py` | ⬜ |
| 03 | `03_gradient_descent_ex3_train_lr.py` | ⬜ |
| 04 | `04_computation_graph_and_backprop_ex1_dZ.py` | ⬜ |
| 04 | `04_computation_graph_and_backprop_ex2_gradient_check.py` | ⬜ |
| 04 | `04_computation_graph_and_backprop_ex3_training_loop.py` | ⬜ |
| 05 | `05_vectorization_ex1_rank1_bug.py` | ⬜ |
| 05 | `05_vectorization_ex2_vectorised_backprop.py` | ⬜ |
| 05 | `05_vectorization_ex3_end_to_end_lr.py` | ⬜ |

## Docs

| File | Covers | Status |
|------|--------|--------|
| `Deep_Learning_Part_2_Logistic_Regression.docx` | Binary classification → vectorisation — full academic reference | ⬜ |

---

## Implement from scratch (benchmark)

Close this section only when every box is ticked and verified without looking anything up.

- [ ] Sigmoid from scratch and its derivative σ(1−σ)
- [ ] Binary cross-entropy loss — formula and why not MSE
- [ ] Forward pass: Z = wᵀX + b, A = σ(Z)
- [ ] Backward pass: dZ = A−Y, dW = (1/m)·X·dZᵀ, db = (1/m)·sum(dZ)
- [ ] One full gradient descent iteration — zero for-loops

---

## Folder Structure

```
logistic_regression/
├── notebooks/
│   ├── 01_binary_classification.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_gradient_descent.ipynb
│   ├── 04_computation_graph_and_backprop.ipynb
│   └── 05_vectorization.ipynb
├── notes/
│   ├── 01_binary_classification_notes.md
│   ├── 02_logistic_regression_notes.md
│   ├── 03_gradient_descent_notes.md
│   ├── 04_computation_graph_and_backprop_notes.md
│   └── 05_vectorization_notes.md
├── exercises/
│   ├── 01_binary_classification_ex1_feature_vector.py
│   ├── 01_binary_classification_ex2_build_matrices.py
│   ├── 01_binary_classification_ex3_dataset_pipeline.py
│   ├── 02_logistic_regression_ex1_sigmoid.py
│   ├── 02_logistic_regression_ex2_predictions.py
│   ├── 02_logistic_regression_ex3_decision_boundary.py
│   ├── 03_gradient_descent_ex1_cross_entropy.py
│   ├── 03_gradient_descent_ex2_learning_rate.py
│   ├── 03_gradient_descent_ex3_train_lr.py
│   ├── 04_computation_graph_and_backprop_ex1_dZ.py
│   ├── 04_computation_graph_and_backprop_ex2_gradient_check.py
│   ├── 04_computation_graph_and_backprop_ex3_training_loop.py
│   ├── 05_vectorization_ex1_rank1_bug.py
│   ├── 05_vectorization_ex2_vectorised_backprop.py
│   └── 05_vectorization_ex3_end_to_end_lr.py
├── images/
│   ├── 01_binary_classification_data_layout.png
│   ├── 02_logistic_regression_sigmoid.png
│   ├── 03_gradient_descent_cost.png
│   ├── 04_computation_graph_backprop.png
│   └── 05_vectorization.png
├── docs/
│   └── Deep_Learning_Part_2_Logistic_Regression.docx
└── README.md
```

---

## Git Commits — This Section

One commit per artifact type, per notebook. Never `git add .`.

```bash
# Notebook + notes together
git add 01_neural_networks_and_dl/logistic_regression/notebooks/01_binary_classification.ipynb
git add 01_neural_networks_and_dl/logistic_regression/notes/01_binary_classification_notes.md
git commit -m "feat(dl): logistic_regression/01_binary_classification — feature vectors, matrix stacking X and Y"

# Exercises (separate commit, after completing all 3)
git add 01_neural_networks_and_dl/logistic_regression/exercises/01_binary_classification_ex*.py
git commit -m "exercise(dl): logistic_regression/01_binary_classification exercises — all 3 ready"

# Docs (one commit when the full docx is done)
git add 01_neural_networks_and_dl/logistic_regression/docs/Deep_Learning_Part_2_Logistic_Regression.docx
git commit -m "docs(dl): logistic_regression guide — binary classification through vectorisation"

# Update README after each notebook is complete
git add 01_neural_networks_and_dl/logistic_regression/README.md
git commit -m "docs(dl): update logistic_regression README — notebook 01 complete"
```