# Notes — Computation Graph & Backpropagation
**Repo:** `AI_Engineer-DL` | **Section:** `01_neural_networks_and_dl/logistic_regression`  
**Notebook:** `notebooks/04_computation_graph_and_backprop.ipynb`  
**Exercises:** `exercises/04_computation_graph_and_backprop_exN_label.py`

---

## 1. What Is This?

A computation graph maps each operation in the forward pass as a node, showing  
what feeds into what. Backpropagation walks this graph right-to-left, applying  
the chain rule at each node to compute gradients.  
This is what PyTorch's autograd does automatically — but understanding it manually  
is what lets you debug gradient issues, design custom layers, and reason about  
why deep networks train the way they do.

**The DL connection:** When CloudyDrive's YOLO makes a wrong detection, the error  
propagates backward through the computation graph — detection head → neck → backbone —  
layer by layer, using the chain rule you implement here.

| Name | What it is |
|------|------------|
| **Forward pass** | Left-to-right computation: x,w,b → Z → A → L |
| **Cache** | Intermediate values (Z, A) saved during forward pass for use in backprop |
| **Chain rule** | ∂L/∂Z = (∂L/∂A)·(∂A/∂Z) — compose gradients through operations |
| **dZ = A − y** | Gradient of loss w.r.t. the pre-activation — the key backprop result |
| **dW, db** | Gradients w.r.t. parameters — used in the update step |

---

## 2. Core Concepts

### 2.1 — The Computation Graph (Forward Pass)

For one example, logistic regression has three nodes:

$$x,w,b \;\xrightarrow{\;Z=w^Tx+b\;}\; Z \;\xrightarrow{\;A=\sigma(Z)\;}\; A \;\xrightarrow{\;\mathcal{L}(A,y)\;}\; \mathcal{L}$$

**What to cache:** Z and A. Backprop needs both.

---

### 2.2 — The Chain Rule

If output depends on Z which depends on w:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial w}$$

Each arrow in the computation graph introduces one multiplicative factor.  
Going backwards (right-to-left), you multiply the incoming gradient by the local derivative at each node.

---

### 2.3 — Deriving dZ = A − y

Apply the chain rule at the Z node:

$$\frac{\partial \mathcal{L}}{\partial Z} = \underbrace{\frac{\partial \mathcal{L}}{\partial A}}_{= -\frac{y}{A}+\frac{1-y}{1-A}} \cdot \underbrace{\frac{\partial A}{\partial Z}}_{= A(1-A)}$$

Expanding and simplifying:

$$= \left(-\frac{y}{A} + \frac{1-y}{1-A}\right) \cdot A(1-A) = -y(1-A) + (1-y)A = A - y$$

**Result: dZ = A − y** — the cross-entropy loss and sigmoid derivative cancel beautifully.

---

### 2.4 — Gradients for W and b (single example)

Once dZ is known:

$$\frac{\partial \mathcal{L}}{\partial w_j} = dZ \cdot x_j \quad \Rightarrow \quad dW = dZ \cdot x \quad \text{(shape: } n_x \times 1\text{)}$$

$$\frac{\partial \mathcal{L}}{\partial b} = dZ \quad \text{(scalar)}$$

Averaged over m examples:

$$dW = \frac{1}{m} X \cdot dZ^T \qquad db = \frac{1}{m} \sum dZ$$

---

## 3. Code Patterns

### Forward pass (cache everything)
```python
def forward_pass(x, w, b, y):
    Z = w.T @ x + b                                # (1,1)
    A = sigmoid(Z)                                 # (1,1)
    L = -(y*np.log(A+1e-9) + (1-y)*np.log(1-A+1e-9))
    return {'Z': Z, 'A': A, 'L': L}
```

### Backward pass (single example)
```python
def backward_pass(cache, x, y):
    A  = cache['A']
    dZ = A - y                   # scalar — the key result
    dW = dZ * x                  # (nx, 1)
    db = dZ                      # scalar
    return {'dZ': dZ, 'dW': dW, 'db': db}
```

### Finite difference gradient check (verify dL/db)
```python
eps = 1e-5
L_plus  = forward_pass(x, w, b + eps, y)['L']
L_minus = forward_pass(x, w, b - eps, y)['L']
fd_db   = (L_plus - L_minus) / (2 * eps)
# Compare with analytical db using np.isclose(fd_db, db_analytical, atol=1e-5)
```

---

## 4. Real-World Connections

| Concept | Real-world use case |
|---------|-------------------|
| **Computation graph** | PyTorch builds a dynamic graph every forward pass — `.backward()` traverses it right-to-left |
| **Cache (Z, A)** | In deep networks, every layer's Z and A are stored in GPU memory during forward pass for use in backward |
| **dZ = A − y** | The output layer gradient in any binary classifier — the starting point of every backprop |
| **Chain rule** | What makes arbitrarily deep networks trainable — gradients compose multiplicatively through layers |
| **Gradient checking** | Used in framework debugging; finite differences confirm autograd is implemented correctly |

---

## 5. Revision Corner

**One-sentence definition:**  
Backpropagation applies the chain rule right-to-left through the computation graph,  
computing how much each parameter contributed to the loss so that gradient descent  
can update them efficiently.

**Why it exists:**  
Without backprop, computing every gradient independently from scratch would require  
one full forward pass per parameter — O(n) passes for n parameters.  
Backprop reuses cached intermediate values so that all gradients are computed in  
a single backward pass — O(1) passes regardless of parameter count.

**Gotchas:**

| Question | Answer |
|----------|--------|
| What is dZ in logistic regression and how is it derived? | dZ = A − y. It comes from the chain rule: (∂L/∂A)·(∂A/∂Z) = (−y/A + (1−y)/(1−A)) · A(1−A), which simplifies exactly to A − y. |
| Why does the forward pass cache Z and A? | Backprop needs Z to compute σ'(Z) = A(1−A) and uses A directly in dZ = A − y. Without caching, you'd recompute the entire forward pass for every gradient. |
| What does finite difference gradient checking verify? | That your analytical gradient matches [L(θ+ε) − L(θ−ε)]/(2ε). If they disagree by more than ~1e-5, your backprop has a bug. |
| What is the shape of dW for nₓ=100? | (100, 1) — same shape as W. Every weight gets its own gradient. |
| Why must w and b be updated simultaneously? | If you update w first and use the new w to compute the gradient for b, you are no longer taking the gradient of J at the same point. Both must be updated from gradients computed at the current (w, b). |

---

## 6. Exercises

| # | Topic | Status |
|---|-------|--------|
| 1 ⭐ | Implement `backward_single(A, y, x)` returning dZ, dW, db — verify dZ with finite differences for 4 (A, y) combinations — all `np.isclose` checks pass | ⬜ To do |
| 2 ⭐⭐ | Full gradient check on all elements of W (nₓ=4) using finite differences — verify with `np.allclose` | ⬜ To do |
| 3 ⭐⭐⭐ | Build full training loop for a 4-feature deepfake detector (50 examples, 500 iterations) — forward + backward + update all from scratch — plot cost curve — comment on what forgetting (1/m) does | ⬜ To do |
