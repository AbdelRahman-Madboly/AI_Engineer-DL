# Notes — Gradient Descent & the Cost Function
**Repo:** `AI_Engineer-DL` | **Section:** `01_neural_networks_and_dl/logistic_regression`  
**Notebook:** `notebooks/03_gradient_descent.ipynb`  
**Exercises:** `exercises/03_gradient_descent_exN_label.py`

---

## 1. What Is This?

Training a model means finding w and b that minimise prediction error.  
The cost function J(w, b) measures that error as a single number — the average  
binary cross-entropy loss over all m training examples.  
Gradient descent then iteratively nudges w and b in the direction that reduces J,  
using the slope (gradient) of J to determine which way to step.

**The DL connection:** Every model you have built — FarmLens, WaveMamba-DF, CloudyDrive —  
was trained by minimising a cost function with some variant of gradient descent.  
The cross-entropy and update rule here are the mathematical core of all of them.

| Name | What it is |
|------|------------|
| **L(ŷ, y)** | Loss: error on one example |
| **J(w, b)** | Cost: average loss over all m examples |
| **Cross-entropy** | The specific loss function for binary classification |
| **α** | Learning rate: step size per gradient descent iteration |
| **∂J/∂w** | Gradient: direction of steepest ascent in J w.r.t. w |

---

## 2. Core Concepts

### 2.1 — Binary Cross-Entropy Loss (per example)

$$\mathcal{L}(\hat{y}, y) = -\left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]$$

**How to read this:**
- y = 1 → loss = −log(ŷ). Want ŷ → 1 to make loss → 0.
- y = 0 → loss = −log(1−ŷ). Want ŷ → 0 to make loss → 0.

**Worked examples:**

| y | ŷ | Loss |
|---|---|------|
| 1 | 0.99 | −log(0.99) ≈ 0.01 ✓ small |
| 1 | 0.01 | −log(0.01) ≈ 4.61 ✗ large |
| 0 | 0.99 | −log(0.01) ≈ 4.61 ✗ large |
| 0 | 0.01 | −log(0.99) ≈ 0.01 ✓ small |

> ⚠️ **Float safety:** Always clip ŷ before taking log:  
> `y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)`  
> `log(0) = −∞` will crash or produce NaN in your cost.

---

### 2.2 — Why Not MSE?

Squared error L = ½(ŷ − y)² with sigmoid output creates a **non-convex** cost surface —  
multiple local minima where gradient descent can get stuck.  
Cross-entropy produces a **convex** surface with a single global minimum.  
Convexity is the guarantee that gradient descent always finds the optimal parameters.

**Comparison:**

| | MSE | Cross-Entropy |
|---|-----|--------------|
| Cost surface shape | Non-convex (multiple minima) | Convex (one minimum) |
| Gradient descent | May get stuck | Always converges |
| Probability interpretation | None | Maximum likelihood |
| Use for logistic regression | ❌ | ✅ |

---

### 2.3 — Cost Function J(w, b)

$$J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

Average of individual losses. Minimising J over all m examples is the entire training objective.

---

### 2.4 — Gradient Descent Update Rule

Repeat until convergence:

$$w := w - \alpha \frac{\partial J}{\partial w} \qquad b := b - \alpha \frac{\partial J}{\partial b}$$

- Subtract because we go **downhill** (negative gradient direction)
- Both w and b updated simultaneously using the same-timestep gradients
- Convexity guarantees convergence from any starting point

**Learning rate α:**

| α too small | α too large |
|-------------|-------------|
| Converges but very slowly | Overshoots minimum — oscillates or diverges |
| Cost decreases every step | Cost may increase on some steps |
| Safe but impractical | Unstable |

---

## 3. Code Patterns

### Loss for one example (with float safety)
```python
def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
```

### Cost over m examples (vectorised)
```python
def compute_cost(Y_hat, Y, m):
    Y_hat = np.clip(Y_hat, 1e-9, 1 - 1e-9)
    return -(1/m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
```

### One gradient descent step
```python
Z = w.T @ X + b             # forward: (1, m)
A = sigmoid(Z)              # predictions: (1, m)
cost = compute_cost(A, Y, m)
dZ = A - Y                  # (1, m)
dw = (1/m) * X @ dZ.T      # (nx, 1)
db = (1/m) * np.sum(dZ)    # scalar
w  = w - alpha * dw
b  = b - alpha * db
```

---

## 4. Real-World Connections

| Concept | Real-world use case |
|---------|-------------------|
| **Cross-entropy loss** | Standard loss for all classification tasks — used in YOLO, EfficientNet, transformers |
| **Cost function J** | The training objective that PyTorch/TF minimise automatically |
| **Learning rate α** | The most important hyperparameter in any training run — tuned first |
| **Convexity of J** | Only guaranteed for logistic regression. Deep networks are non-convex but SGD still works in practice |
| **Cost curve (J vs iteration)** | The primary diagnostic tool during training — any spike or plateau is a signal |

---

## 5. Revision Corner

**One-sentence definition:**  
The cost function J(w,b) is the average binary cross-entropy over all m examples,  
and gradient descent minimises it by repeatedly subtracting the scaled gradient  
w := w − α·∂J/∂w until convergence.

**Why it exists:**  
Without a cost function there is no signal for learning — you cannot know if your parameters  
are getting better or worse. Cross-entropy is specifically chosen because it is convex  
(guaranteeing one global minimum) and has a clean probabilistic interpretation  
(maximising log-likelihood of the labels given the inputs).

**Gotchas:**

| Question | Answer |
|----------|--------|
| Why is MSE bad for logistic regression? | MSE + sigmoid gives a non-convex cost surface with multiple local minima. Gradient descent can get stuck. Cross-entropy is convex, so one global minimum is guaranteed. |
| What happens if α is too large? | Gradient descent overshoots the minimum. The cost oscillates or diverges — you see the cost going up instead of down. |
| What is the gradient of J w.r.t. w? | dJ/dw = (1/m) · X · dZᵀ, where dZ = A − Y. The (1/m) factor is critical — without it the gradient scales with m and your learning rate breaks. |
| Can J be negative? | No. Cross-entropy = −log(probability). Since −log(p) ≥ 0 for p ∈ (0,1], J ≥ 0 always. |
| What is the difference between loss and cost? | Loss L is the error on one example. Cost J is the average over the full training set. Gradient descent minimises J. |

---

## 6. Exercises

| # | Topic | Status |
|---|-------|--------|
| 1 ⭐ | Implement `compute_loss` and `compute_cost` from scratch — clip before log — verify 6 test cases with `np.isclose` | ⬜ To do |
| 2 ⭐⭐ | Run gradient descent with α ∈ {0.001, 0.1, 1.5} for 300 iterations — plot all three cost curves — identify which diverges | ⬜ To do |
| 3 ⭐⭐⭐ | Train logistic regression on linearly separable 2D tumour data (200 examples, 1000 iterations) — plot cost curve + decision boundary — answer why the boundary is a straight line | ⬜ To do |
