# ROADMAP.md — AI_Engineer-DL
# The plan: what to build, in what order, and why.
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
  └── 01_neural_networks_and_dl          ← foundations, the full picture
  └── 02_improving_deep_nn               ← making NNs actually work well
  └── 03_structuring_ml_projects         ← strategy, train/dev/test, transfer learning
  └── 04_convolutional_neural_networks   ← images, detection, style transfer
  └── 05_sequence_models                 ← RNNs, LSTMs, attention, transformers
```

---

## Course Build Order

### Course 1 — Neural Networks and Deep Learning
Build in strict week order. Each week depends on the previous.

**Week 1 — Introduction (1 notebook)**
- What is a neural network? Why deep learning now? Supervised learning overview.
- Concepts: ReLU, hidden layers, structured vs unstructured data, data scale.

**Week 2 — Logistic Regression as a Neural Network (5 notebooks)**
This is the mathematical core. Everything in deep learning builds on this week.
- Binary classification, feature vectors, matrix notation
- Logistic regression: sigmoid, weights, bias
- Cost function: cross-entropy loss, why not MSE
- Gradient descent: the update rule, learning rate
- Derivatives and computation graphs: chain rule, forward/backward pass
- Vectorization: eliminating for-loops with NumPy broadcasting

**Week 3 — Shallow Neural Network (3 notebooks)**
- Neural network representation: input → hidden → output layers
- Activation functions: sigmoid, tanh, ReLU, leaky ReLU — when to use each
- Backpropagation through a 2-layer network: full derivation

**Week 4 — Deep Neural Network (3 notebooks)**
- L-layer network: forward propagation with matrix dimensions tracked
- Backpropagation: general L-layer derivation
- Parameters vs hyperparameters: what you tune and what you learn

**Capstone project:** Cat classifier — end-to-end binary image classifier from scratch.

---

### Course 2 — Improving Deep Neural Networks
Build after Course 1 is complete.

**Week 1 — Practical Aspects**
- Train/dev/test splits in the DL era (not 70/30 anymore)
- Bias and variance — diagnosing underfitting vs overfitting
- Regularization: L2 weight decay, dropout, data augmentation, early stopping
- Vanishing/exploding gradients and weight initialization (Xavier, He)

**Week 2 — Optimization Algorithms**
- Mini-batch gradient descent
- Exponentially weighted averages
- Momentum, RMSprop, Adam — understanding all three
- Learning rate decay and local optima

**Week 3 — Hyperparameter Tuning, Batch Norm, Frameworks**
- Hyperparameter search: random vs grid, coarse-to-fine
- Batch normalization: why it works, where to place it
- Softmax regression
- Introduction to TensorFlow

---

### Course 3 — Structuring ML Projects
Concepts-only course — notes and docs, no notebooks for most topics.

**Week 1 — ML Strategy I**
- Orthogonalization: one knob at a time
- Single evaluation metrics: precision vs recall → F1
- Train/dev/test distributions: must come from same distribution
- Comparing to human-level performance: Bayes error, avoidable bias

**Week 2 — ML Strategy II**
- Error analysis: manually examining dev set mistakes
- Mismatched train/dev data: what to do
- Transfer learning: when and how
- Multi-task learning
- End-to-end deep learning: when to use it and when not to

---

### Course 4 — Convolutional Neural Networks
Build after Course 2 is complete. Directly relevant to CloudyDrive and FarmLens.

**Week 1 — CNN Foundations**
- Convolution operation: filter, stride, padding
- Pooling: max pooling, average pooling
- Full CNN architecture: conv → pool → flatten → dense

**Week 2 — Deep CNN Architectures**
- LeNet, AlexNet, VGG: why depth matters
- ResNets: skip connections, solving vanishing gradients
- Inception network: 1×1 convolutions

**Week 3 — Object Detection**
- Object localization: bounding box prediction
- YOLO: unified detection, anchor boxes, non-max suppression
- This directly explains how CloudyDrive's YOLOv11 works

**Week 4 — Special Applications**
- Face recognition: one-shot learning, Siamese networks, triplet loss
- Neural style transfer: content and style cost functions

---

### Course 5 — Sequence Models
Build after Course 3 is complete.

**Week 1 — Recurrent Neural Networks**
- RNN forward/backward pass
- Vanishing gradients in RNNs
- GRU and LSTM: solving the vanishing gradient problem

**Week 2 — NLP and Word Embeddings**
- Word2Vec, GloVe
- Sentiment classification, debiasing word embeddings

**Week 3 — Sequence Models and Attention**
- Encoder-decoder architecture
- Attention mechanism: the core idea behind transformers
- Speech recognition, trigger word detection

**Week 4 — Transformer Network**
- Self-attention, multi-head attention
- Transformer architecture
- HuggingFace: applying pre-trained transformers (NER, QA)

---

## Projects Plan

| Project | After | What it applies |
|---------|-------|----------------|
| `cat_classifier` | Course 1 | Binary classification, full DL pipeline from scratch |
| `digit_recognizer` | Course 2 | Regularization, optimization, full training pipeline |
| `object_detector_lite` | Course 4 | CNN + detection concepts, connects to CloudyDrive |
| `sentiment_analyzer` | Course 5 | RNN/LSTM, text classification |
| `mini_transformer` | Course 5 | Attention + transformer from scratch |
