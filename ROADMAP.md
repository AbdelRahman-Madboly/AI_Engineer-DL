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
  └── 01_neural_networks_and_dl          ⏳ In progress
  └── 02_improving_deep_nn               ⬜ Not started
  └── 03_structuring_ml_projects         ⬜ Not started
  └── 04_convolutional_neural_networks   ⬜ Not started
  └── 05_sequence_models                 ⬜ Not started
```

---

## Course Build Order

### Course 1 — Neural Networks and Deep Learning

**01_intro_to_dl** (1 notebook) ✅ Complete
- What is a neural network? Why deep learning now?
- ReLU, hidden layers, structured vs unstructured data, data scale.

**02_logistic_regression** (5 notebooks) ⏳ Exercises remaining
- Binary classification, feature vectors, matrix notation — notebooks ✅ notes ✅ docs ✅
- Logistic regression: sigmoid, weights, bias
- Cost function: cross-entropy loss, why not MSE
- Gradient descent: update rule, learning rate, convexity
- Computation graphs: chain rule, forward/backward pass, dZ = A−Y
- Vectorization: eliminating for-loops with NumPy broadcasting
- **Next:** write 15 exercise scaffolds (3 per notebook)

**03_shallow_neural_network** (3 notebooks) ⬜ Not started
- Layer notation, forward pass shapes
- Activation functions: sigmoid, tanh, ReLU, leaky ReLU — derivatives and when to use each
- Backpropagation through a 2-layer network: full derivation, random initialisation

**04_deep_neural_network** (3 notebooks) ⬜ Not started
- L-layer forward propagation: matrix dimensions tracked at every step
- General backpropagation: L-layer derivation, caching
- Parameters vs hyperparameters: what you tune vs what you learn

**Capstone project:** `cat_classifier` — end-to-end binary image classifier from scratch.

---

### Course 2 — Improving Deep Neural Networks

**01_practical_aspects** (3 notebooks)
- Train/dev/test splits in the DL era
- Bias and variance: diagnosing underfitting vs overfitting
- Regularization: L2, dropout, data augmentation
- Vanishing/exploding gradients: Xavier and He initialization

**02_optimization_algorithms** (2 notebooks)
- Mini-batch gradient descent
- Exponentially weighted averages, momentum
- RMSprop, Adam, learning rate decay

**03_hyperparameter_tuning** (2 notebooks)
- Random search, coarse-to-fine, log-scale sampling
- Batch normalization: why it works, where to place it
- Softmax regression
- TensorFlow introduction

---

### Course 3 — Structuring ML Projects (notes only)

**01_ml_strategy_1**
- Orthogonalization, single evaluation metrics
- Train/dev/test distributions, human-level performance, Bayes error

**02_ml_strategy_2**
- Error analysis, mismatched train/dev data
- Transfer learning, multi-task learning, end-to-end deep learning

---

### Course 4 — Convolutional Neural Networks

**01_foundations** (2 notebooks)
- Convolution operation: filter, stride, padding, output shape
- Pooling: max/average, full CNN architecture

**02_deep_cnn_models** (2 notebooks)
- LeNet, AlexNet, VGG: why depth matters
- ResNets: skip connections; Inception: 1×1 convolutions

**03_object_detection** (2 notebooks)
- YOLO: unified detection, anchor boxes, NMS — directly explains CloudyDrive YOLOv11
- Bounding box prediction: IoU, mAP

**04_special_applications** (2 notebooks)
- Face recognition: one-shot learning, Siamese networks, triplet loss
- Neural style transfer: content and style cost functions

---

### Course 5 — Sequence Models

**01_recurrent_neural_networks** (2 notebooks)
- RNN forward/backward pass
- GRU and LSTM: solving vanishing gradients

**02_nlp_word_embeddings** (2 notebooks)
- Word2Vec, GloVe, embedding matrix
- Sentiment classification, debiasing

**03_sequence_attention** (2 notebooks)
- Encoder-decoder architecture, beam search
- Attention mechanism: the core idea behind transformers

**04_transformers** (2 notebooks)
- Self-attention, multi-head attention, positional encoding
- HuggingFace: pre-trained models, NER, QA

---

## Projects Plan

| Project | After | What it applies | Status |
|---------|-------|----------------|--------|
| `cat_classifier` | Course 1 | Binary classification, full DL pipeline from scratch | ⬜ |
| `digit_recognizer` | Course 2 | Regularization, optimization, full training pipeline | ⬜ |
| `object_detector_lite` | Course 4 | CNN + detection, connects to CloudyDrive | ⬜ |
| `sentiment_analyzer` | Course 5 | RNN/LSTM, text classification | ⬜ |
| `mini_transformer` | Course 5 | Attention + transformer from scratch | ⬜ |