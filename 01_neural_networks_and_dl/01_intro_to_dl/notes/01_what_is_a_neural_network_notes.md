# Notes — What Is a Neural Network?
**Repo:** `AI_Engineer-DL` | **Section:** `01_neural_networks_and_dl/01_intro_to_dl`
**Notebook:** `notebooks/01_what_is_a_neural_network.ipynb`
**Exercises:** `exercises/01_what_is_a_neural_network_ex1_relu_from_scratch.py`

---

## 1. What Is This?

A neural network is a function that maps an input to an output by passing it through layers of simple computing units called neurons. Each neuron takes a weighted sum of its inputs, adds a bias, applies a nonlinearity, and produces a single number. Stack enough neurons together in layers and a network can learn to approximate almost any function — from predicting house prices to detecting objects in images.

Deep learning is the practice of training these networks when they have many layers. The "deep" refers to depth — not complexity of thought, just the number of layers between input and output.

**Real-world connection — CloudyDrive:** Every layer in the YOLOv11 backbone running inside CloudyDrive is a stack of neurons computing a weighted sum and applying ReLU. Understanding one neuron from scratch is understanding the atomic unit of your autonomous vehicle's perception system.

| Name | What it is |
|------|------------|
| **Neuron** | A single computing unit: takes inputs, applies weights + bias, applies a nonlinearity, outputs one number |
| **ReLU** | Rectified Linear Unit — the most common nonlinearity: output = max(0, z) |
| **Hidden layer** | A layer of neurons between input and output whose values are not directly observed |
| **Dense layer** | A layer where every neuron receives every input — fully connected |
| **Deep network** | A network with more than one hidden layer |

---

## 2. Core Concepts

### 2.1 The Neuron as a Function

A single neuron takes a vector of inputs **x** ∈ ℝⁿ and produces a scalar output:

$$z = \mathbf{w}^\top \mathbf{x} + b$$

$$a = g(z)$$

- **w** — weight vector (same shape as **x**): how much each input contributes
- **b** — bias (scalar): shifts the activation threshold
- **z** — pre-activation: the raw weighted sum before the nonlinearity
- **g(·)** — activation function: introduces nonlinearity so the network can learn curved decision boundaries
- **a** — activation / output of this neuron

The weight vector **w** and bias **b** are the parameters the network learns during training. Before training, they are initialized (often randomly). After training, they encode everything the network has learned from data.

**Worked example:** Suppose n = 2, **w** = [0.5, -1.0], b = 0.3, **x** = [2.0, 1.0].

$$z = (0.5)(2.0) + (-1.0)(1.0) + 0.3 = 1.0 - 1.0 + 0.3 = 0.3$$

Then g(z) = ReLU(0.3) = max(0, 0.3) = 0.3.

### 2.2 ReLU — Rectified Linear Unit

The most widely used activation function in hidden layers:

$$\text{ReLU}(z) = \max(0,\ z)$$

Its derivative:

$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Why ReLU instead of sigmoid in hidden layers?**

| Property | Sigmoid σ(z) | ReLU max(0,z) |
|----------|-------------|--------------|
| Output range | (0, 1) | [0, ∞) |
| Gradient at large \|z\| | ≈ 0 (vanishes) | 1 (constant) |
| Computation | exp required | single comparison |
| Learning speed | Slow in deep nets | Fast |

The vanishing gradient problem: sigmoid saturates near 0 and 1 — gradient becomes nearly zero, so parameters in early layers stop learning. ReLU gradient is 1 for all positive inputs, so the signal travels cleanly backward through many layers.

### 2.3 Stacking Neurons — Hidden Layers

A layer is a collection of neurons that all receive the same input and produce outputs in parallel. For a layer with n[1] neurons receiving input **x** ∈ ℝⁿ⁰:

$$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$

$$\mathbf{A}^{[1]} = g(\mathbf{Z}^{[1]})$$

- **W**[1] has shape (n[1], n[0]) — one weight row per neuron
- **b**[1] has shape (n[1], 1) — one bias per neuron
- **Z**[1] has shape (n[1], 1) — one pre-activation per neuron
- g is applied element-wise

The output **A**[1] becomes the input to the next layer. Stack L layers and you get a deep network.

**Why depth works:** Each layer learns a different level of abstraction. In image recognition, early layers detect edges, middle layers detect shapes, deep layers detect objects. A single layer would need exponentially more neurons to represent the same function.

### 2.4 Why Deep Learning Is Taking Off Now

Three forces combined in the last decade:

**Data scale:** Digitization of society produced massive labeled datasets. Traditional algorithms (SVM, logistic regression) plateau as data grows. Neural networks keep improving.

**Compute scale:** GPUs and TPUs enabled training networks with millions of parameters in hours instead of years.

**Algorithmic improvements:** Switching from sigmoid to ReLU in hidden layers was one of the most impactful changes — it solved the vanishing gradient problem and made training deep networks practical. Better initialization, batch normalization, and the Adam optimizer followed.

The scale relationship: more data + larger network + more compute = better performance. This relationship holds reliably — one of the few things in ML that almost always works.

---

## 3. Code Patterns

### Single neuron forward pass
Compute z = wᵀx + b, then apply ReLU. This is the building block of every layer.
```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def neuron_forward(x, w, b):
    z = np.dot(w, x) + b   # weighted sum + bias
    a = relu(z)             # activation
    return a, z             # return both — z is needed for backprop later
```

### One dense layer forward pass
Vectorize across all neurons in the layer using matrix multiply.
```python
def layer_forward(X, W, b):
    """
    X: (n_in,)  or (n_in, m) for a batch of m examples
    W: (n_out, n_in)
    b: (n_out, 1)
    Returns A of shape (n_out,) or (n_out, m)
    """
    Z = np.dot(W, X) + b   # (n_out, m)
    A = relu(Z)             # element-wise
    return A, Z
```

### ReLU derivative (needed for backprop)
```python
def relu_derivative(Z):
    return (Z > 0).astype(float)   # 1 where Z > 0, else 0
```

---

## 4. Real-World Connections

| Concept | Real-world use case |
|---------|-------------------|
| **Single neuron** | Every unit in a fully-connected layer of a classification head |
| **ReLU activation** | Default nonlinearity in convolutional and dense layers in nearly all modern architectures |
| **Dense layer** | Final classification layers in image models, text encoders, and recommendation systems |
| **Deep vs shallow** | ResNet-50 has 50 layers; each layer builds on representations from the layer before |
| **Data scale drives performance** | ImageNet (1.2M images) unlocked modern CV; GPT-4 trained on trillions of tokens |

---

## 5. Revision Corner

**One-sentence definition:**
A neural network is a parameterized function composed of layers of neurons — each computing a weighted sum of its inputs, adding a bias, and applying a nonlinearity — trained to approximate a target mapping from input to output.

**Why it exists:**
Most real-world mappings (image → label, audio → text, features → price) are not linear and cannot be captured by a single equation. Neural networks provide a universal approximator: given enough neurons and data, they can represent any continuous function. The layered structure allows them to learn hierarchical features automatically from data, without hand-engineering.

**Gotchas:**

| Question | Answer |
|----------|--------|
| Is a deeper network always better? | No. Deeper networks are more expressive but also harder to train and more likely to overfit if data is limited. Depth is only useful when you have enough data and compute to take advantage of it. |
| Does every neuron in a hidden layer learn a separate concept? | Not in general. Neurons in practice learn distributed representations — a single concept may be spread across many neurons, and a single neuron may respond to multiple concepts. |
| Why can't you use linear activations in all layers? | A composition of linear functions is linear. A 10-layer network with linear activations is mathematically equivalent to a 1-layer network — adding depth adds nothing. |
| Is ReLU always the right choice? | For hidden layers, ReLU is the default. For the output layer: sigmoid for binary classification (outputs a probability), softmax for multi-class, linear for regression. |
| Does `np.dot(w, x)` work when x is 1D? | Yes, but shapes must match: if w is (n,) and x is (n,), you get a scalar. Use (n_out, n_in) @ (n_in, 1) = (n_out, 1) for batched layer operations to avoid subtle shape bugs. |

---

## 6. Exercises

| # | Topic | Status |
|---|-------|--------|
| 1 ⭐ | Implement `relu(z)` from scratch using only `max()` or comparison operators — no `np.maximum`. Test on a range of positive, negative, and zero inputs. Verify with `np.allclose` against `np.maximum(0, z_array)`. | ⬜ To do |
| 2 ⭐⭐ | Implement a single neuron forward pass: given **w** (shape (3,)), **b** (scalar), and **x** (shape (3,)), compute z = wᵀx + b and a = ReLU(z) from scratch. Then replace your ReLU with `np.maximum` and verify both outputs match using `np.isclose`. | ⬜ To do |
| 3 ⭐⭐⭐ | Scenario — Housing price mini-network: build a 4-input → 3-hidden → 1-output network forward pass from scratch using only `np.dot` and your own `relu`. Provided: W1 (3,4), b1 (3,1), W2 (1,3), b2 (1,1), x (4,1). Compute A1 and the final output. Verify layer shapes at each step. Plot the ReLU activation curve for z ∈ [-3, 3] and save to `../images/01_relu_activation.png`. | ⬜ To do |
