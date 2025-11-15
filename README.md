[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)
# IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

## 📌 Purpose of this Template

This repository is the **starter** for your End Semester Project submission in GitHub Classroom. You can implement your solution and push your work in this repository. Please free to edit this README.md file as per your requirements.
## Repository Overview

This repository contains the final implementation of the Hybrid OvR classification system
developed for the Machine Learning End Semester Project. The goal was to build a multi-class
classifier for handwritten digits (MNIST) **without using external ML libraries such as sklearn,
TensorFlow, or PyTorch**, and implement the core logic fully from scratch using NumPy.

The solution uses a **single unified model: `ovr_variant`**, which integrates:
- One-vs-Rest Gradient Boosting (implemented manually)
- Epsilon-triggered KNN refinement for ambiguous predictions
- PCA dimensionality reduction applied globally for speed and noise reduction

This hybrid design significantly improves robustness and reduces misclassification among visually
similar digits (e.g., 4 vs 9), outperforming the baseline pure OvR model.

Only the `ovr_variant` model is required and used in this project — all training, prediction,
and validation is performed through this final model.


**Important Note:** 
1. TAs will evaluate using the `.py` file only.
2. All your reports, plots, visualizations, etc pertaining to your solution should be uploaded to this GitHub repository

---

## 📁 Repository Structure


### Summary of Code Organization
- **Model Logic** is fully implemented in `Hybrid_algorithm.py` with no external ML libraries.
- **Only `ovr_variant` should be used**—it encapsulates boosting, PCA, and KNN refinement in one model.
- **Running `main.py`** trains and evaluates the hybrid model from the command line.
- **endsem_Report** provides detailed explanation of methodology and results.



---

## 📦 Installation & Dependencies

### Requirements
To run this project, you need the following installed on your system:
- Python 3.8 or above
- pip (Python package manager)
- numpy
- pandas
- counter from collections

### Python Dependencies
The project does not use external ML libraries such as sklearn, TensorFlow, or PyTorch.
Only basic scientific computing packages are required: pip install numpy pandas matplotlib

---

## ▶️ Running the Code

All experiments should be runnable from the command line **and** reproducible in the notebook.

### A. Command-line (recommended for grading)
- python main.py 
* Mention the instructions to run you .py files.

Place the following CSV files in the same directory as main.py:
- MNIST_train.csv
- MNIST_validation.csv

Output :
- Training Progress will be shown
- prints accuracy score (95.5 - 96.02) and f1 score(95.5 - 96.02)
- Due to the stochastic nature of boosting and data ordering, accuracy fluctuates slightly between runs.

---

## 🧾 Authors

**Aakash,DA24B028**, IIT Madras (2025–26)


## Best Practices:
* Keep commits with meaningful messages.
* Please do not write all code on your local machine and push everything to GitHub on the last day. The commits in GitHub should reflect how the code has evolved during the course of the assignment.
* Collaborations and discussions with other students is strictly prohibited.
* Code should be modularized and well-commented.

