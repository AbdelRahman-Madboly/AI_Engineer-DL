# Notes — Binary Classification
**Repo:** `AI_Engineer-DL` | **Section:** `01_neural_networks_and_dl/logistic_regression`  
**Notebook:** `notebooks/01_binary_classification.ipynb`  
**Exercises:** `exercises/01_binary_classification_exN_label.py`

---

## 1. What Is This?

Binary classification maps an input x to one of two labels y ∈ {0, 1}.  
Before any model can learn, the raw data — images, signals, text — must be encoded as  
fixed-length numeric vectors. For images, each pixel channel value becomes one feature.  
The full training set is then packed into two matrices X and Y whose shapes encode  
the column convention that makes every downstream matrix operation efficient.

**The DL connection:** The (nₓ, m) shape of X is used by every neural network  
in this repo — from logistic regression here through YOLO in Course 4.

| Name | What it is |
|------|------------|
| **x⁽ⁱ⁾** | Feature vector for the i-th training example — shape (nₓ, 1) |
| **y⁽ⁱ⁾** | Binary label for example i — scalar ∈ {0, 1} |
| **nₓ** | Number of input features (e.g. 64×64×3 = 12,288) |
| **m** | Number of training examples |
| **X** | All examples stacked as columns — shape (nₓ, m) |
| **Y** | All labels in one row — shape (1, m) |

---

## 2. Core Concepts

### 2.1 — Image as a Feature Vector

An RGB image of height H, width W has three colour channels.  
Flatten all channels into one column vector:

$$x \in \mathbb{R}^{n_x}, \quad n_x = H \times W \times C$$

For a 64×64 RGB image: nₓ = 64 × 64 × 3 = **12,288**.  
Each row of x is one pixel-channel value (0–255).

**In NumPy:** `x = image.reshape(-1, 1)` — the `-1` infers the length automatically.  
Always verify: `x.shape == (nx, 1)`, never `(nx,)`.

---

### 2.2 — Training Set Notation

A single example: **(x⁽ⁱ⁾, y⁽ⁱ⁾)** where superscript **(i)** indexes the example.  
Full training set: `{(x⁽¹⁾, y⁽¹⁾), …, (x⁽ᵐ⁾, y⁽ᵐ⁾)}`.

> ⚠️ **Notation rule:** Parentheses (i) = training example index. Square brackets [l] = layer index.  
> These two are never confused — one is data indexing, the other is architecture indexing.

---

### 2.3 — Matrix X: examples as columns

$$X = \begin{bmatrix} x^{(1)} & x^{(2)} & \cdots & x^{(m)} \end{bmatrix} \in \mathbb{R}^{n_x \times m}$$

Shape: **(nₓ, m)** — examples are **columns**, features are **rows**.

**Why columns, not rows?** Because W·X then computes predictions for all m examples  
in one matrix multiply — no loop. If examples were rows, every computation needs an  
extra transpose and the code becomes harder to read.

**In NumPy:** `X = np.hstack(list_of_column_vectors)` or `batch.reshape(m, -1).T`.

---

### 2.4 — Matrix Y: labels in a row

$$Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} \in \mathbb{R}^{1 \times m}$$

Shape: **(1, m)**. Always keep this as a 2-D array — never shape (m,).

**In NumPy:** `Y = labels.reshape(1, m)`.

---

## 3. Code Patterns

### Unroll image to column vector
One-liner that produces shape (nₓ, 1) from any image array.
```python
x = image.reshape(-1, 1)        # (H*W*C, 1)
assert x.shape == (nx, 1)
```

### Build X from a batch array (shape m, H, W, C)
```python
X = batch.reshape(m, -1).T      # (nx, m) — one line, no loop
assert X.shape == (nx, m)
```

### Build Y from a labels array (shape m,)
```python
Y = labels.reshape(1, m)        # (1, m)
assert Y.shape == (1, m)
```

### Access the i-th example (preserve column shape)
```python
x_i = X[:, i:i+1]              # shape (nx, 1) — NOT X[:, i] which gives (nx,)
```

---

## 4. Real-World Connections

| Concept | Real-world use case |
|---------|-------------------|
| **Feature vector x** | Any CNN takes a flattened or spatially organised version of raw pixel data |
| **X matrix (nₓ, m)** | Every mini-batch in PyTorch/TF is X with m = batch size |
| **nₓ** | Determines the first-layer weight matrix size — directly impacts parameter count |
| **Binary label y** | Detection confidence score thresholded at 0.5 in YOLO post-processing |
| **Column convention** | Consistent across logistic regression, shallow NN, deep NN, CNNs |

---

## 5. Revision Corner

**One-sentence definition:**  
Binary classification maps input x ∈ ℝⁿˣ to a label y ∈ {0,1}, and the m training  
examples are organised as matrix X of shape (nₓ, m) with examples as columns  
and labels as matrix Y of shape (1, m).

**Why it exists:**  
A neural network processes data in parallel across many examples simultaneously.  
The column layout of X is the prerequisite for this — without it, every operation  
over examples would require an explicit for-loop, and training would be 100–300× slower.

**Gotchas:**

| Question | Answer |
|----------|--------|
| Why is X shaped (nₓ, m) not (m, nₓ)? | With examples as columns, W·X computes predictions for all m examples in one matrix multiply. The (m, nₓ) layout forces an extra transpose at every step and makes code harder to read. |
| What is nₓ for a 32×32 RGB image? | 32 × 32 × 3 = 3,072. Always multiply height × width × channels. Greyscale (1 channel) gives 1,024. |
| Can Y have shape (m,) instead of (1, m)? | No. Shape (m,) is a rank-1 array — broadcasting behaves differently and gradient computations produce wrong shapes. Always enforce (1, m). |
| What does x⁽ⁱ⁾ refer to? | The i-th training example's feature vector. The parentheses distinguish it from layer indices [l]. |
| How do you preserve the column shape when slicing X? | Use `X[:, i:i+1]` (shape nx, 1) not `X[:, i]` (shape nx,). The slice preserves the second dimension. |

---

## 6. Exercises

| # | Topic | Status |
|---|-------|--------|
| 1 ⭐ | Write `image_to_vector(image)` using `.reshape()` only — test on (28,28) and (32,32,3) — verify shapes with assertions | ⬜ To do |
| 2 ⭐⭐ | Build X (nₓ, m) and Y (1, m) from a (m, H, W, C) batch in zero loops — verify X[:,i] recovers example i | ⬜ To do |
| 3 ⭐⭐⭐ | Simulate a binary "deepfake vs real" frame dataset — build X, Y — plot histogram of mean pixel values per class — answer why raw pixel means would fail a real detector | ⬜ To do |
