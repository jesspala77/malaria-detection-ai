# Malaria Detection AI

A portfolio-ready machine learning project for malaria microscopy classification built with TensorFlow and Streamlit.

## Overview

This repository packages a fine-tuned ResNet50 workflow for classifying blood smear cell images as either parasitized or uninfected. The main deliverable is an interactive Streamlit app that presents:

- a polished inference workflow for uploaded or sample microscopy images
- conservative uncertainty handling for safer review-oriented predictions
- model evaluation visuals including confusion matrix, training curves, ROC/PRC, and probability analysis
- supporting notebooks that document the progression from baseline CNNs to the final transfer learning model

## Repository Layout

```text
malaria-detection-ai/
├── model/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   ├── assets/
│   ├── models/
│   ├── notebooks/
│   └── samples/
└── scripts/
    └── convert_model.py
```

## Featured App

The main application lives in [`model/`](./model) and includes:

- a clinical-style prediction interface
- automatic discovery of evaluation visuals from exported chart files
- compatibility handling for newer TensorFlow and Keras model-loading behavior
- downloadable prediction summaries for demo and portfolio use

For app-specific details, setup steps, and usage instructions, see [`model/README.md`](./model/README.md).

## Quick Start

From the `model/` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Recommended Python version: `3.10` or `3.11`.

## Streamlit Community Cloud

Deploy this app with the following settings:

- Repository: `jesspala77/malaria-detection-ai`
- Branch: `codex/polish-portfolio-app` until the PR is merged, then `main`
- Entrypoint file: `model/app.py`
- Python version: `3.11`

Important:

- Do not use the default newest Python version in Community Cloud for this app.
- The current TensorFlow requirement (`>=2.16,<2.19`) does not install on Python 3.14, which causes deployment to fail before the app starts.
- If the app was already deployed with the wrong Python version, delete it and redeploy with Python `3.11`.

## Project Highlights

- Transfer learning with ResNet50 for binary medical image classification
- Streamlit app redesigned as a polished portfolio demo
- Evaluation artifacts surfaced directly in the interface
- Jupyter notebooks documenting iterative model development
- Cleaner environment setup and repository organization for GitHub presentation

## Notes

- This project is for research, educational, and portfolio purposes only.
- It is not intended for clinical diagnosis or real-world medical decision making.

## Author

Jessica Palacio
