# Notes — Logistic Regression
**Repo:** `AI_Engineer-DL` | **Section:** `01_neural_networks_and_dl/logistic_regression`  
**Notebook:** `notebooks/02_logistic_regression.ipynb`  
**Exercises:** `exercises/02_logistic_regression_exN_label.py`

---

## 1. What Is This?

Logistic regression is the algorithm that turns a linear score into a probability.  
It takes the weighted sum z = wᵀx + b — which can be any real number — and squashes it  
through the sigmoid function to produce ŷ ∈ (0, 1), a valid probability that y = 1.  
Every neural network output layer doing binary classification is logistic regression.

**The DL connection:** WaveMamba-DF's final layer computes exactly ŷ = σ(wᵀ·features + b),  
where features are the deep representations produced by the EfficientNet-B5 backbone.

| Name | What it is |
|------|------------|
| **w** | Weight vector ∈ ℝⁿˣ — one weight per input feature |
| **b** | Bias scalar ∈ ℝ — shifts the decision boundary |
| **z** | Linear pre-activation: z = wᵀx + b |
| **σ(z)** | Sigmoid function — maps ℝ → (0, 1) |
| **ŷ** | Predicted probability P(y=1\|x) |

---

## 2. Core Concepts

### 2.1 — Why Not Linear Regression?

A linear model outputs z = wᵀx + b ∈ ℝ — unbounded, can be negative or > 1.  
Probabilities must live in [0, 1]. Linear regression cannot be interpreted as a probability  
and cannot be trained with a proper probabilistic loss function.

---

### 2.2 — The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

**Key values:**

| z | σ(z) | Meaning |
|---|------|---------|
| → +∞ | → 1 | Model is certain: class 1 |
| 0 | 0.5 | Model is uncertain |
| → −∞ | → 0 | Model is certain: class 0 |

**In NumPy:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

> ⚠️ **Float safety:** For very large negative z (e.g. z = −1000), `np.exp(-z)` overflows to  
> `inf`, making σ(z) = 0. This is numerically correct but can cause log(0) issues  
> downstream. Use `scipy.special.expit` in production for numerical stability.

---

### 2.3 — Sigmoid Derivative

$$\frac{d\sigma}{dz} = \sigma(z)\,(1 - \sigma(z))$$

If a = σ(z): **derivative = a · (1 − a)**.

**Key property:** Maximum at z = 0, value = 0.25. Falls toward 0 for large |z|.  
This is the vanishing gradient problem — hidden layer sigmoids kill gradients in deep nets.

**Comparison:**

| | Sigmoid | ReLU (Week 3) |
|---|---------|--------------|
| Range | (0, 1) | [0, ∞) |
| Derivative max | 0.25 at z=0 | 1 for all z > 0 |
| Vanishing gradient | Yes, for |z| > 5 | No |
| Use in hidden layers | ❌ Avoid | ✅ Default |
| Use in output (binary) | ✅ Yes | ❌ No |

---

### 2.4 — The Logistic Regression Model

$$\hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

- w ∈ ℝⁿˣ — weight vector, same size as input
- b ∈ ℝ — bias scalar
- x ∈ ℝⁿˣ — one input example (column vector)
- ŷ ∈ (0, 1) — predicted probability

**Shapes for one example:**

| Variable | Shape |
|----------|-------|
| x | (nₓ, 1) |
| w | (nₓ, 1) |
| b | scalar |
| z = wᵀx + b | (1, 1) |
| ŷ = σ(z) | (1, 1) |

---

## 3. Code Patterns

### Sigmoid from scratch
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))   # works on scalars, 1-D, 2-D arrays
```

### Sigmoid derivative
```python
def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)             # = σ(z) · (1 − σ(z))
```

### Forward pass: z then ŷ
```python
z    = w.T @ x + b                 # (1,1) — linear score
y_hat = sigmoid(z)                 # (1,1) — predicted probability
```

### Library equivalent
```python
from scipy.special import expit
y_hat = expit(w.T @ x + b)        # numerically stable sigmoid
```

---

## 4. Real-World Connections

| Concept | Real-world use case |
|---------|-------------------|
| **Sigmoid output** | Binary classification output layer in any detector or classifier |
| **w (weights)** | Each weight controls how much one feature (e.g. one pixel or embedding dimension) influences the prediction |
| **Bias b** | Allows the decision boundary to shift — without it, the boundary always passes through the origin |
| **Vanishing sigmoid gradient** | Why ResNets, ReLU, and batch norm exist — sigmoid in deep nets stops gradient flow |
| **ŷ as probability** | Threshold at 0.5 for hard prediction; keep soft for calibrated confidence scores |

---

## 5. Revision Corner

**One-sentence definition:**  
Logistic regression computes ŷ = σ(wᵀx + b), where the sigmoid maps the linear score  
z = wᵀx + b to a probability ŷ ∈ (0, 1) representing P(y=1|x).

**Why it exists:**  
Linear regression cannot produce valid probabilities. The sigmoid wraps the linear model  
to ensure the output is always in (0, 1), interpretable as a probability, and differentiable  
everywhere — enabling gradient-based training with a proper probabilistic loss function.

**Gotchas:**

| Question | Answer |
|----------|--------|
| What is the sigmoid derivative at z = 0? | 0.25. This is the maximum value of the derivative. For all other z it is smaller, approaching 0 for large |z|. |
| Why does sigmoid cause vanishing gradients? | For |z| > 5, the derivative is near 0. Chained through many layers during backprop, near-zero multiplications shrink the gradient to nothing before it reaches early layers. |
| What does the bias b do geometrically? | It shifts the decision boundary (where σ(wᵀx+b) = 0.5, i.e. wᵀx+b = 0) away from the origin. Without b, the boundary must pass through x=0. |
| Can ŷ ever equal exactly 0 or 1? | No. σ(z) ∈ (0,1) — open interval. It approaches but never reaches the endpoints. This matters: log(0) = −∞, so the loss is undefined at the extremes. Always clip ŷ before taking log. |
| What is the shape of w for nₓ=12288? | (12288, 1) — a column vector. Never (12288,) — that is a rank-1 array and will cause silent shape bugs. |

---

## 6. Exercises

| # | Topic | Status |
|---|-------|--------|
| 1 ⭐ | Implement `sigmoid(z)` and `sigmoid_derivative(z)` — verify against `scipy.special.expit` with `np.allclose` on scalars, 1-D and 2-D inputs | ⬜ To do |
| 2 ⭐⭐ | Compute ŷ for a 6-example mini dataset using matrix form (no loop) — threshold at 0.5 — compute accuracy | ⬜ To do |
| 3 ⭐⭐⭐ | Generate 2-class 2-feature data, train w and b manually for 200 steps, visualise scatter + decision boundary, answer what doubling w does | ⬜ To do |
