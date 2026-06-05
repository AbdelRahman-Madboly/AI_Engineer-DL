# AI_Engineer-DL

**Author:** Abdel Rahman Madboly
**GitHub:** https://github.com/AbdelRahman-Madboly/AI_Engineer-DL
**Local:** `D:\AI_Engineer-DL`
**Environment:** `dl_env` (see `environment.yml`)

---

## What This Repo Is

A deep learning learning repository built by implementing every concept from scratch in Jupyter notebooks. Every notebook is written by Abdel Rahman — not AI-generated. Every concept is explained before it is coded. The goal is to understand deep learning deeply enough to implement architectures from scratch, explain every design decision in an interview, and apply them to real projects.

This repo follows the same structure and standards as `AI_Engineer-ML` — the same notebook format (7 sections), the same notes style, the same exercise scaffold pattern.

**Companion repo:** `AI_Engineer-ML` (classical ML, math foundations) — complete that first.

---

## Repo Structure

```
AI_Engineer-DL/
│
├── 01_neural_networks_and_dl/          ⏳ In progress
│   ├── week_01_intro_to_dl/
│   ├── week_02_logistic_regression/
│   ├── week_03_shallow_neural_network/
│   ├── week_04_deep_neural_network/
│   └── docs/
│
├── 02_improving_deep_nn/               ⏳ Not started
│   ├── week_01_practical_aspects/
│   ├── week_02_optimization_algorithms/
│   ├── week_03_hyperparameter_tuning/
│   └── docs/
│
├── 03_structuring_ml_projects/         ⏳ Not started
│   ├── week_01_ml_strategy_1/
│   ├── week_02_ml_strategy_2/
│   └── docs/
│
├── 04_convolutional_neural_networks/   ⏳ Not started
│   ├── week_01_foundations/
│   ├── week_02_deep_cnn_models/
│   ├── week_03_object_detection/
│   ├── week_04_special_applications/
│   └── docs/
│
├── 05_sequence_models/                 ⏳ Not started
│   ├── week_01_recurrent_neural_networks/
│   ├── week_02_nlp_word_embeddings/
│   ├── week_03_sequence_attention/
│   ├── week_04_transformers/
│   └── docs/
│
├── projects/                           ⏳ Planned
├── data/
├── docs/                               ← Repo-level reference guides
├── environment.yml                     ← dl_env spec — always commit this
├── PROGRESS.md
├── ROADMAP.md
└── README.md
```

Each week folder follows this pattern:
```
week_NN_topic_name/
├── notebooks/      NN_topic_name.ipynb
├── notes/          NN_topic_name_notes.md
├── exercises/      NN_topic_name_exN_short_label.py
└── images/         NN_topic_short_desc.png
```

---

## Prerequisites

Before starting this repo, make sure you have completed or are comfortable with:

- `AI_Engineer-ML / 01_math_foundations / linear_algebra` — matrix operations, eigenvalues, PCA
- `AI_Engineer-ML / 01_math_foundations / calculus` — derivatives, gradient descent, chain rule
- `AI_Engineer-ML / 02_data_tools` — NumPy, Pandas, Matplotlib, sklearn basics
- `AI_Engineer-ML / 03_linear_regression` — cost functions, gradient descent, regression pipelines

The math behind backpropagation is calculus. The math behind weight matrices is linear algebra. You need both before this repo makes sense.

---

## Environment Setup

```bash
# Create the environment (run once)
conda env create -f environment.yml

# Activate
conda activate dl_env

# Register as a Jupyter kernel
python -m ipykernel install --user --name dl_env --display-name "Python (dl_env)"

# Launch
cd D:\AI_Engineer-DL
jupyter notebook
```

Always select **Python (dl_env)** as the kernel before running any notebook.

---

## Notebook Standard — 7 Sections Every Time

Every notebook in this repo follows the exact same structure:

```
# NN — Topic Name

## 1. What Is This?
   Plain language. No code. What, why, where in deep learning.
   Real-world connection — one of Abdel Rahman's projects.
   Concept orientation table.

## 2. The Math
   LaTeX formulas with word-for-word explanation of every symbol.
   Worked examples showing the arithmetic, not just the result.
   Derivations where the concept requires it (e.g. backprop).

## 3. Build From Scratch
   One sub-concept per markdown+code cell pair.
   🔮 Predict before you run.
   # YOUR CODE HERE — all task cells.
   💡 Reflect after non-obvious outputs.
   ✅ Self-check at natural breakpoints.

## 4. Library / Framework Version
   TensorFlow/Keras or NumPy equivalent.
   Verify scratch matches the framework.

## 5. Visualisation
   Pre-filled. Save to ../images/NN_topic_short_description.png

## 6. Revision Corner
   One-sentence definition (interview answer).
   Why it exists — what problem it solves.
   3-column gotcha table: Question | Common mistake | The reality.

## 7. Exercises
   3 exercises: ⭐ Recall · ⭐⭐ Apply · ⭐⭐⭐ Scenario.
   Solutions in exercises/ — never in the notebook.
```

---

## Progress

| Course | Notebooks | Notes | Exercises | Docs | Status |
|--------|-----------|-------|-----------|------|--------|
| `01_neural_networks_and_dl` | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 4 | ⏳ |
| `02_improving_deep_nn` | 0 / 7 | 0 / 7 | 0 / 7 | 0 / 3 | ⏳ |
| `03_structuring_ml_projects` | — | 0 / 4 | — | 0 / 2 | ⏳ |
| `04_convolutional_neural_networks` | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 4 | ⏳ |
| `05_sequence_models` | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 4 | ⏳ |

Status: ⬜ Not started · 🟡 In progress · ✅ Complete

---

## Git Rules

```
<type>(dl): <file or section name> — <one line of what it covers>

Types:
  feat      → new notebook content
  exercise  → exercise scaffold or solutions
  fix       → mistake corrected
  refactor  → restructured without content change
  docs      → README, notes, PROGRESS, or DOCX update
  scaffold  → new folder structure created
```

**Examples:**
```bash
feat(dl): add 01_binary_classification — vectorized logistic regression, sigmoid from scratch
feat(dl): add 02_shallow_nn — forward/backward pass, activation functions from scratch
exercise(dl): scaffold week02 exercises — all 3 ready
docs(dl): update PROGRESS after course 1 week 2 complete
```

**Workflow after every session:**
```bash
git add 01_neural_networks_and_dl/week_02_logistic_regression/notebooks/01_binary_classification.ipynb
git add 01_neural_networks_and_dl/week_02_logistic_regression/notes/01_binary_classification_notes.md
git commit -m "feat(dl): add 01_binary_classification — ..."
git push
```

Never `git add .` and never `git commit -m "update"`.
