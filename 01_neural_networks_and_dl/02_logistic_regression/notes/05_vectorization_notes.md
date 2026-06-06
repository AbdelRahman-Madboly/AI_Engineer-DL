# Notes — Vectorization
**Repo:** `AI_Engineer-DL` | **Section:** `01_neural_networks_and_dl/logistic_regression`  
**Notebook:** `notebooks/05_vectorization.ipynb`  
**Exercises:** `exercises/05_vectorization_exN_label.py`

---

## 1. What Is This?

Vectorization replaces explicit Python for-loops over training examples with NumPy  
matrix operations that run in compiled, SIMD-parallel C code.  
The result is typically 100–300× faster on CPU, and the same principle is what  
makes GPU training possible — a GPU is just vectorization taken to extreme parallelism.  
A correct vectorised implementation of logistic regression has zero explicit for-loops.

**The DL connection:** CloudyDrive's YOLO processes 32 frames simultaneously per forward  
pass. Each layer is a matrix multiply over the entire batch — pure vectorization.  
Without it, training a 12-layer network on 50,000 images would be impractical.

| Name | What it is |
|------|------------|
| **Vectorization** | Replacing loops with matrix ops — 100–300× faster |
| **Broadcasting** | NumPy auto-expanding smaller arrays to match larger ones |
| **Rank-1 array** | Shape (n,) — neither row nor column — source of silent bugs |
| **SIMD** | CPU-level parallelism that NumPy exploits for speed |
| **Vectorised LR** | Full logistic regression in 3 lines, zero loops |

---

## 2. Core Concepts

### 2.1 — From Loop to Matrix: Forward Pass

**Loop version:**
```python
for i in range(m):
    z[i] = w.T @ x[:, i] + b
    a[i] = sigmoid(z[i])
```

**Vectorised version:**
```python
Z = w.T @ X + b    # (1, m) — all m examples at once
A = sigmoid(Z)     # (1, m)
```

Speed: the vectorised version is ~200× faster on large m. Same result — verified with `np.allclose`.

---

### 2.2 — Vectorised Backward Pass

$$dZ = A - Y \quad \text{shape } (1, m)$$
$$dW = \frac{1}{m} X \cdot dZ^T \quad \text{shape } (n_x, 1)$$
$$db = \frac{1}{m} \sum dZ \quad \text{scalar}$$

No loop over examples anywhere.

**Shape tracking (nₓ=5, m=100):**

| Step | Operation | Shape |
|------|-----------|-------|
| Z = wᵀX + b | (1,5)@(5,100) + scalar | (1,100) |
| A = σ(Z) | element-wise | (1,100) |
| dZ = A − Y | element-wise | (1,100) |
| dW = (1/m)·X·dZᵀ | (5,100)@(100,1) | (5,1) |
| db = (1/m)·sum(dZ) | sum all | scalar |

---

### 2.3 — The Rank-1 Array Bug

```python
a = np.random.randn(5)      # shape (5,)  ← DANGER
a.T.shape                    # still (5,) — transpose does nothing!
a @ a                        # scalar — silent inner product
```

**Fix:**
```python
a = np.random.randn(5, 1)   # shape (5,1) — proper column vector
a.T.shape                    # (1,5) — transpose works
a @ a.T                      # (5,5) outer product ✓
a.T @ a                      # (1,1) inner product ✓
```

**Rule:** After every initialisation, add:
```python
assert w.shape == (nx, 1), f"w shape error: {w.shape}"
assert Y.shape == (1, m),  f"Y shape error: {Y.shape}"
```

---

### 2.4 — Broadcasting Rules

Two shapes are broadcastable when dimensions match right-to-left, with 1 allowed on either side:

| A shape | B shape | Result | Notes |
|---------|---------|--------|-------|
| (1, m) | scalar | (1, m) | b broadcasts to every element |
| (nₓ, m) | (nₓ, 1) | (nₓ, m) | column vector adds to every column |
| (3, 4) | (1, 4) | (3, 4) | row vector adds to every row |
| (3, 4) | (3,) | **ERROR** | rank-1 arrays break broadcasting |

> ⚠️ **Always avoid rank-1 arrays in neural network code.**  
> Use `np.sum(dZ, axis=1, keepdims=True)` not `np.sum(dZ, axis=1)` to preserve (1,1) shape.

---

## 3. Code Patterns

### Complete vectorised logistic regression — one step (zero loops)
```python
def lr_step(X, Y, w, b, alpha):
    m  = X.shape[1]
    Z  = w.T @ X + b                  # (1, m)
    A  = sigmoid(Z)                   # (1, m)
    J  = -(1/m) * np.sum(
           Y * np.log(np.clip(A, 1e-9, 1-1e-9)) +
           (1-Y) * np.log(np.clip(1-A, 1e-9, 1-1e-9)))
    dZ = A - Y                        # (1, m)
    dw = (1/m) * X @ dZ.T            # (nx, 1)
    db = (1/m) * float(np.sum(dZ))   # scalar
    w  = w - alpha * dw
    b  = b - alpha * db
    return w, b, J
```

### Speed comparison template
```python
import time
t0 = time.time(); Z_vec = w.T @ X + b; t_vec = time.time() - t0
t0 = time.time()
for i in range(m): Z_loop[0,i] = np.dot(w.T, X[:,i]) + b
t_loop = time.time() - t0
print(f"Speedup: {t_loop/t_vec:.1f}×")
```

---

## 4. Real-World Connections

| Concept | Real-world use case |
|---------|-------------------|
| **Matrix multiply for all examples** | Every layer in a neural network — the core GPU operation |
| **Broadcasting** | Adding bias to a layer output: (n,m) + (n,1) — used in every forward pass |
| **Rank-1 bug** | A common source of silent shape errors in production ML code — causes wrong results, not crashes |
| **Vectorised backward pass** | Backprop in PyTorch is vectorised over the batch — same equations, auto-differentiated |
| **Batch processing** | mini-batch SGD is just vectorization with m = batch_size (e.g. 32, 64, 256) |

---

## 5. Revision Corner

**One-sentence definition:**  
Vectorization replaces for-loops over m training examples with matrix operations  
Z = wᵀX + b and A = σ(Z) that process all m examples simultaneously in compiled  
NumPy code, achieving 100–300× speedup.

**Why it exists:**  
Python loops are slow because each iteration is interpreted separately.  
NumPy passes entire arrays to optimised C code that uses CPU SIMD instructions  
to process multiple data elements per clock cycle. Without vectorization, training  
on any realistic dataset would be orders of magnitude too slow.

**Gotchas:**

| Question | Answer |
|----------|--------|
| What is a rank-1 array and why is it dangerous? | Shape (n,). The `.T` attribute does nothing — transpose returns the same shape. `a @ a` gives a scalar instead of an outer product. Always use (n,1) or (1,n). The bug is silent — no error is thrown. |
| Write the vectorised backward pass equations. | dZ = A − Y (shape 1,m). dW = (1/m)·X·dZᵀ (shape nₓ,1). db = (1/m)·np.sum(dZ) (scalar). Zero loops anywhere. |
| Why use keepdims=True in np.sum? | Without it, summing axis=1 of shape (1,m) gives shape (1,) not (1,1). This breaks broadcasting in the update step. keepdims=True preserves the number of dimensions. |
| How does broadcasting handle Z = wᵀX + b? | wᵀX has shape (1,m). b is a scalar. NumPy treats the scalar as shape (1,1) and broadcasts it to (1,m), adding it to every element. This is equivalent to a for-loop over m but runs in one C operation. |
| What is the vectorised cost function? | J = −(1/m) · np.sum(Y·log(A_clip) + (1−Y)·log(1−A_clip)). One np.sum over the entire (1,m) matrix — no loop over examples. |

---

## 6. Exercises

| # | Topic | Status |
|---|-------|--------|
| 1 ⭐ | Demonstrate the rank-1 bug in 4 steps — show `.T` does nothing — fix with `.reshape(n,1)` — verify outer and inner products have correct shapes | ⬜ To do |
| 2 ⭐⭐ | Implement vectorised forward+backward (nₓ=3, m=8) — verify all gradients match a loop-based version with `np.allclose` | ⬜ To do |
| 3 ⭐⭐⭐ | Train full logistic regression end-to-end on a cat/non-cat proxy dataset (nₓ=20, m_train=200, m_test=50, 2000 iterations) — print train+test accuracy — plot cost curve — comment on why test accuracy may differ | ⬜ To do |
