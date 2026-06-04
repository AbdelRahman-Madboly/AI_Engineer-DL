# ROADMAP.md — AI_Engineer-DL
# Last updated: June 2026

---

## Learning Path

```
AI_Engineer-ML (prerequisite)
  └── 01_math_foundations/calculus       ← chain rule, gradient descent
  └── 01_math_foundations/linear_algebra ← matrix multiply, eigenvectors
  └── 02_data_tools                      ← NumPy, vectorization
  └── 03_linear_regression               ← cost functions, gradient descent
         ↓
AI_Engineer-DL (this repo)
  └── 01_neural_networks_and_dl
  └── 02_improving_deep_nn
  └── 03_structuring_ml_projects
  └── 04_convolutional_neural_networks
  └── 05_sequence_models
```

---

## Course Build Order

### Course 1 — Neural Networks and Deep Learning

**intro_to_dl** (1 notebook)
- What is a neural network? Why deep learning now?
- ReLU, hidden layers, structured vs unstructured data, data scale.

**logistic_regression** (5 notebooks)
This section is the mathematical core. Every concept here reappears everywhere.
- Binary classification, feature vectors, matrix notation
- Logistic regression: sigmoid, weights, bias
- Cost function: cross-entropy loss, why not MSE
- Gradient descent: update rule, learning rate
- Computation graphs: chain rule, forward/backward pass
- Vectorization: eliminating for-loops with NumPy broadcasting

**shallow_neural_network** (3 notebooks)
- Layer notation, forward pass shapes
- Activation functions: sigmoid, tanh, ReLU, leaky ReLU
- Backpropagation through a 2-layer network: full derivation

**deep_neural_network** (3 notebooks)
- L-layer forward propagation: matrix dimensions tracked at every step
- General backpropagation: L-layer derivation
- Parameters vs hyperparameters: what you tune vs what you learn

**Capstone project:** `cat_classifier` — end-to-end binary image classifier from scratch.

---

### Course 2 — Improving Deep Neural Networks

**practical_aspects** (3 notebooks)
- Train/dev/test splits in the DL era
- Bias and variance: diagnosing underfitting vs overfitting
- Regularization: L2, dropout, data augmentation
- Vanishing/exploding gradients: Xavier and He initialization

**optimization_algorithms** (2 notebooks)
- Mini-batch gradient descent
- Exponentially weighted averages, momentum
- RMSprop, Adam, learning rate decay

**hyperparameter_tuning** (2 notebooks)
- Random search, coarse-to-fine, log-scale sampling
- Batch normalization: why it works, where to place it
- Softmax regression
- TensorFlow introduction

---

### Course 3 — Structuring ML Projects (notes only)

**ml_strategy_1**
- Orthogonalization, single evaluation metrics
- Train/dev/test distributions, human-level performance, Bayes error

**ml_strategy_2**
- Error analysis, mismatched train/dev data
- Transfer learning, multi-task learning, end-to-end deep learning

---

### Course 4 — Convolutional Neural Networks

**foundations** (2 notebooks)
- Convolution operation: filter, stride, padding, output shape
- Pooling: max/average, full CNN architecture

**deep_cnn_models** (2 notebooks)
- LeNet, AlexNet, VGG: why depth matters
- ResNets: skip connections; Inception: 1×1 convolutions

**object_detection** (2 notebooks)
- YOLO: unified detection, anchor boxes, NMS — directly explains CloudyDrive YOLOv11
- Bounding box prediction: IoU, mAP

**special_applications** (2 notebooks)
- Face recognition: one-shot learning, Siamese networks, triplet loss
- Neural style transfer: content and style cost functions

---

### Course 5 — Sequence Models

**recurrent_neural_networks** (2 notebooks)
- RNN forward/backward pass
- GRU and LSTM: solving vanishing gradients

**nlp_word_embeddings** (2 notebooks)
- Word2Vec, GloVe, embedding matrix
- Sentiment classification, debiasing

**sequence_attention** (2 notebooks)
- Encoder-decoder architecture, beam search
- Attention mechanism: the core idea behind transformers

**transformers** (2 notebooks)
- Self-attention, multi-head attention, positional encoding
- HuggingFace: pre-trained models, NER, QA

---

## Projects Plan

| Project | After | What it applies |
|---------|-------|----------------|
| `cat_classifier` | Course 1 | Binary classification, full DL pipeline from scratch |
| `digit_recognizer` | Course 2 | Regularization, optimization, full training pipeline |
| `object_detector_lite` | Course 4 | CNN + detection, connects to CloudyDrive |
| `sentiment_analyzer` | Course 5 | RNN/LSTM, text classification |
| `mini_transformer` | Course 5 | Attention + transformer from scratch |