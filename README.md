# AI_Engineer-DL

**Author:** Abdel Rahman Madboly
**GitHub:** https://github.com/AbdelRahman-Madboly/AI_Engineer-DL
**Local:** `D:\AI_Engineer-DL`
**Environment:** `dl_env` — TensorFlow + PyTorch + CUDA in one environment
**Companion repo:** [AI_Engineer-ML](https://github.com/AbdelRahman-Madboly/AI_Engineer-ML) — complete that first

---

## What This Repo Is

A deep learning learning repository built by implementing every concept from scratch in Jupyter
notebooks. Every notebook is written by Abdel Rahman — not AI-generated. Every concept is
explained before it is coded. The goal is to understand deep learning deeply enough to implement
architectures from scratch, explain every design decision in an interview, and apply them to
real projects.

This repo follows the same standards as `AI_Engineer-ML` — the same 7-section notebook format,
the same notes style, the same exercise scaffold pattern.

---

## Prerequisites

| Topic | Repo section |
|-------|-------------|
| Derivatives, gradient descent, chain rule | `AI_Engineer-ML / 01_math_foundations / calculus` |
| Matrix operations, eigenvectors, PCA | `AI_Engineer-ML / 01_math_foundations / linear_algebra` |
| NumPy, Pandas, Matplotlib | `AI_Engineer-ML / 02_data_tools` |
| Cost functions, regression pipelines | `AI_Engineer-ML / 03_linear_regression` |

---

## Repo Structure

```
AI_Engineer-DL/
│
├── 01_neural_networks_and_dl/               ⏳ In progress
│   ├── 01_intro_to_dl/                      ✅ Complete
│   │   ├── notebooks/
│   │   ├── notes/
│   │   ├── exercises/
│   │   └── images/
│   ├── 02_logistic_regression/              ⏳ In progress — exercises remaining
│   │   ├── notebooks/
│   │   ├── notes/
│   │   ├── exercises/
│   │   ├── images/
│   │   └── docs/
│   ├── 03_shallow_neural_network/           ⬜ Not started
│   ├── 04_deep_neural_network/              ⬜ Not started
│   ├── docs/                                ← section-level DOCX guides
│   └── README.md
│
├── 02_improving_deep_nn/                    ⬜ Not started
│   ├── 01_practical_aspects/
│   ├── 02_optimization_algorithms/
│   ├── 03_hyperparameter_tuning/
│   └── docs/
│
├── 03_structuring_ml_projects/              ⬜ Not started
│   ├── 01_ml_strategy_1/                    ← notes + images only (no notebooks)
│   ├── 02_ml_strategy_2/
│   └── docs/
│
├── 04_convolutional_neural_networks/        ⬜ Not started
│   ├── 01_foundations/
│   ├── 02_deep_cnn_models/
│   ├── 03_object_detection/
│   ├── 04_special_applications/
│   └── docs/
│
├── 05_sequence_models/                      ⬜ Not started
│   ├── 01_recurrent_neural_networks/
│   ├── 02_nlp_word_embeddings/
│   ├── 03_sequence_attention/
│   ├── 04_transformers/
│   └── docs/
│
├── projects/                                ⬜ Planned
├── docs/                                    ← repo-level reference guides
│
├── _prompts/                                ← gitignored — local only
│   ├── DL_REPO_SKILL.md
│   ├── HOW_TO_USE.md
│   ├── skills/
│   │   ├── LESSON_SKILL.md
│   │   ├── NOTES_SKILL.md
│   │   ├── NOTEBOOKS_SKILL.md
│   │   ├── EXERCISES_SKILL.md
│   │   ├── DOCX_SKILL.md
│   │   └── REPO_SKILL.md
│   └── courses/
│       ├── COURSE_01.md
│       ├── COURSE_02.md
│       ├── COURSE_03.md
│       ├── COURSE_04.md
│       └── COURSE_05.md
│
├── _course_content/                         ← gitignored — subtitle files (local only)
│
├── .gitignore
├── environment.yml
├── PROGRESS.md
├── ROADMAP.md
└── README.md
```

Each topic folder follows this pattern:
```
NN_topic_name/
├── notebooks/    NN_topic.ipynb
├── notes/        NN_topic_notes.md
├── exercises/    NN_topic_exN_label.py
├── images/       NN_topic_description.png
└── README.md     section tracker
```

Course 3 (`structuring_ml_projects`) is concepts-only — notes and images, no notebooks.

---

## Environment Setup

```bash
# 1. Create the environment (run once — takes ~10 min with GPU packages)
conda env create -f environment.yml

# 2. Activate
conda activate dl_env

# 3. Register as a Jupyter kernel
python -m ipykernel install --user --name dl_env --display-name "Python (dl_env)"

# 4. Verify GPU is visible for both frameworks
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('TF GPU:', tf.config.list_physical_devices('GPU'))"

# 5. Launch
cd D:\AI_Engineer-DL
jupyter notebook
```

Always select **Python (dl_env)** as the kernel before running any notebook.

---

## Notebook Standard — 7 Sections Every Time

```
# NN — Topic Name
## 1. What Is This?        Plain language + real-world project connection
## 2. The Math             Formulas with full symbol explanations + worked examples
## 3. Build From Scratch   🔮 Predict → # YOUR CODE HERE → 💡 Reflect → ✅ Self-check
## 4. Library Version      TensorFlow/Keras or NumPy — verify scratch matches
## 5. Visualisation        Pre-filled — saves to ../images/
## 6. Revision Corner      Definition · Why it exists · 3-column gotcha table
## 7. Exercises            ⭐ Recall · ⭐⭐ Apply · ⭐⭐⭐ Scenario — solutions in exercises/
```

---

## Real-World Connections

| Concept | Project |
|---------|---------|
| Sigmoid / binary classification | WaveMamba-DF: final layer outputs deepfake probability |
| Matrix multiplication | CloudyDrive: every YOLO layer is Wᵀx + b |
| Backprop through layers | FarmLens: gradients flow back through EfficientNet |
| Vectorization across batch | Any training loop: process all images in a batch at once |
| ReLU activation | CloudyDrive: all hidden layers use ReLU |
| Dropout regularization | WaveMamba-DF: prevents overfitting on deepfake dataset |
| Convolution | CloudyDrive: YOLOv11 feature extraction is pure convolution |
| LSTM | Nexus: sequence modeling for multi-turn agent memory |
| Attention | WaveMamba-DF: Boundary Attention Module |
| Transformer | Nexus: HuggingFace models for text understanding |

---

## Progress

| Course | Notebooks | Notes | Exercises | Docs | Status |
|--------|-----------|-------|-----------|------|--------|
| `01_neural_networks_and_dl` | 6 / 12 | 6 / 12 | 3 / 12 | 3 / 4 | ⏳ In progress |
| `02_improving_deep_nn` | 0 / 7 | 0 / 7 | 0 / 7 | 0 / 3 | ⬜ Not started |
| `03_structuring_ml_projects` | — | 0 / 4 | — | 0 / 2 | ⬜ Not started |
| `04_convolutional_neural_networks` | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 4 | ⬜ Not started |
| `05_sequence_models` | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 4 | ⬜ Not started |

### Course 01 detail

| Section | Notebooks | Notes | Exercises | Docs | Status |
|---------|-----------|-------|-----------|------|--------|
| `01_intro_to_dl` | 1 / 1 | 1 / 1 | 3 / 3 | 1 / 1 | ✅ Complete |
| `02_logistic_regression` | 5 / 5 | 5 / 5 | 0 / 15 | 2 / 2 | ⏳ Exercises remaining |
| `03_shallow_neural_network` | 0 / 3 | 0 / 3 | 0 / 9 | 0 / 1 | ⬜ Not started |
| `04_deep_neural_network` | 0 / 3 | 0 / 3 | 0 / 9 | 0 / 1 | ⬜ Not started |

---

## Git Workflow

```bash
# After every session — specific files only, never git add .
git add 01_neural_networks_and_dl/02_logistic_regression/exercises/01_binary_classification_ex1_feature_vector.py
git commit -m "exercise(dl): logistic_regression/01_binary_classification exercises — all 3 ready"
git push
```

Commit types: `feat` · `exercise` · `fix` · `refactor` · `docs` · `scaffold`