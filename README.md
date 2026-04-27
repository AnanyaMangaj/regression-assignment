# 📈 Regression Analysis — Machine Learning Assignment

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()

> A comprehensive implementation of **5 regression algorithms** in Machine Learning — from simple linear regression to regularization techniques — with hands-on datasets, evaluation metrics, and a full model comparison.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Algorithms Covered](#-algorithms-covered)
- [Datasets Used](#-datasets-used)
- [Project Structure](#-project-structure)
- [Notebooks Summary](#-notebooks-summary)
- [Model Performance Results](#-model-performance-results)
- [Installation & Setup](#-installation--setup)
- [Technologies Used](#-technologies-used)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

## 📖 Project Overview

This assignment explores five core **regression algorithms** in supervised machine learning using Python and scikit-learn. Each algorithm is implemented in a dedicated Jupyter Notebook with a relevant dataset, step-by-step code, visualizations, and performance evaluation.

**Goals:**
- Understand how linear, polynomial, and regularized regression models work
- Apply models to real-world style datasets (salary, housing, job levels)
- Compare model performance using **R² Score** and **Mean Squared Error (MSE)**
- Identify when to use regularization (Ridge / Lasso) to combat overfitting

---

## 🤖 Algorithms Covered

| # | Algorithm | Type | Dataset Used | Key Concept |
|---|---|---|---|---|
| 1 | Simple Linear Regression | Linear | Salary (YearsExperience → Salary) | Single feature, straight-line fit |
| 2 | Multiple Linear Regression | Linear | Housing (area, bedrooms, bathrooms → price) | Multiple features, hyperplane fit |
| 3 | Polynomial Regression | Non-linear | Job Level → Salary | Degree-2 feature transformation |
| 4 | Ridge Regression | Regularized | Housing dataset | L2 penalty, shrinks coefficients |
| 5 | Lasso Regression | Regularized | Housing dataset | L1 penalty, performs feature selection |

---

## 📊 Datasets Used

### 1. `salary_dataset.csv` — Simple Linear Regression
| Feature | Description |
|---|---|
| `YearsExperience` | Years of professional experience |
| `Salary` | Annual salary (target variable) |

25 records mapping experience to salary — ideal for a straight-line regression.

### 2. `housing_dataset.csv` — Multiple Linear / Ridge / Lasso
| Feature | Description |
|---|---|
| `area` | Property area (sq ft) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `price` | Property price (target variable) |

20 records of housing transactions used across notebooks 2, 4, and 5.

### 3. `polynomial_dataset.csv` — Polynomial Regression
| Feature | Description |
|---|---|
| `Level` | Job level (1–10) |
| `Salary` | Salary at each level (target variable) |

10 records showing exponential salary growth by job level — cannot be fit by a straight line.

---

## 📁 Project Structure

```
regression-assignment/
│
├── data/
│   ├── salary_dataset.csv          ← Simple linear regression data
│   ├── housing_dataset.csv         ← Multiple / Ridge / Lasso data
│   └── polynomial_dataset.csv      ← Polynomial regression data
│
├── notebooks/
│   ├── 1_simple_linear_regression.ipynb
│   ├── 2_multiple_linear_regression.ipynb
│   ├── 3_polynomial_regression.ipynb
│   ├── 4_ridge_lasso_regression.ipynb
│   └── 5_model_comparison.ipynb
│
├── results/
│   ├── graphs/
│   │   └── model_comparison.png    ← Bar chart: R² scores across all models
│   └── model_scores/
│       └── model_comparison.csv    ← Final comparison table (exported)
│
└── README.md
```

---

## 📓 Notebooks Summary

### 1. Simple Linear Regression (`1_simple_linear_regression.ipynb`)
- **Dataset:** `salary_dataset.csv`
- **Features:** `YearsExperience` → `Salary`
- **Steps:** Load data → train-test split (80/20) → fit `LinearRegression` → evaluate → scatter + regression line plot
- **Output:** R² Score and MSE on test set; regression line plotted over raw data

### 2. Multiple Linear Regression (`2_multiple_linear_regression.ipynb`)
- **Dataset:** `housing_dataset.csv`
- **Features:** `area`, `bedrooms`, `bathrooms` → `price`
- **Steps:** Load → EDA → split → train `LinearRegression` → evaluate
- **Output:** R² Score: **0.9944** | MSE: 17,605,263,177

### 3. Polynomial Regression (`3_polynomial_regression.ipynb`)
- **Dataset:** `polynomial_dataset.csv`
- **Features:** `Level` → `Salary` (non-linear, exponential relationship)
- **Steps:** Apply `PolynomialFeatures(degree=2)` → transform features → fit `LinearRegression` on transformed data
- **Output:** Curved fit line over exponential salary growth; visual comparison vs straight-line fit

### 4. Ridge & Lasso Regression (`4_ridge_lasso_regression.ipynb`)
- **Dataset:** `housing_dataset.csv`
- **Models:**
  - `Ridge(alpha=1.0)` — L2 regularization → R² Score: **0.9952**
  - `Lasso(alpha=0.1)` — L1 regularization → R² Score: **0.9944**
- **Output:** Side-by-side comparison of Ridge vs Lasso vs plain Linear Regression

### 5. Model Comparison (`5_model_comparison.ipynb`)
- Consolidates all 4 models (Linear, Polynomial, Ridge, Lasso) on the housing dataset in one notebook
- Builds a final comparison DataFrame sorted by R² Score
- Exports results to `results/model_scores/model_comparison.csv`
- Saves a bar chart to `results/graphs/model_comparison.png`

---

## 📈 Model Performance Results

All models evaluated on the **Housing Dataset** with 80/20 train-test split (`random_state=42`):

| Rank | Model | R² Score | MSE |
|---|---|---|---|
| 🥇 1 | **Ridge Regression** | **0.9952** | 1.51 × 10¹⁰ |
| 🥈 2 | Polynomial Regression | 0.9946 | 1.72 × 10¹⁰ |
| 🥉 3 | Lasso Regression | 0.9944 | 1.76 × 10¹⁰ |
| 4 | Multiple Linear Regression | 0.9944 | 1.76 × 10¹⁰ |

> **Best model: Ridge Regression** achieved the highest R² Score (0.9952), confirming that L2 regularization slightly improves generalization even on a small, clean dataset.

---

## ⚙ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/regression-assignment.git
cd regression-assignment

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Launch Jupyter Notebook
jupyter notebook notebooks/
```

Open any notebook from the `notebooks/` folder and run all cells top to bottom.

---

## 🛠 Technologies Used

| Library | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **pandas** | Data loading and wrangling |
| **NumPy** | Numerical operations and array handling |
| **scikit-learn** | `LinearRegression`, `Ridge`, `Lasso`, `PolynomialFeatures`, `train_test_split`, `r2_score`, `mean_squared_error` |
| **Matplotlib** | Scatter plots and regression line visualizations |
| **Seaborn** | Model comparison bar charts |
| **Jupyter Notebook** | Interactive development environment |

---

## 💡 Key Learnings

- **Simple Linear Regression** works well when a single feature has a strong linear relationship with the target variable — YearsExperience is a strong predictor of Salary.
- **Multiple Linear Regression** improves predictions when several features collectively explain variance in the target — housing price depends on area, bedrooms *and* bathrooms together.
- **Polynomial Regression** captures non-linear trends. Salary growth by job level is exponential, not linear, and degree-2 transformation fits it much better than a straight line.
- **Ridge Regression (L2)** shrinks all coefficients proportionally — useful when all features contribute and mild regularization can prevent overfitting without discarding any feature.
- **Lasso Regression (L1)** can shrink coefficients to exactly zero, performing **automatic feature selection** — ideal when some features may be irrelevant or redundant.
- On the housing dataset, **Ridge slightly outperformed** all other models (R² = 0.9952), confirming that even a small regularization boost can improve generalization on real-world data.

---

## 👤 Author

**Keertiraj Kamble**  
B.E. in Artificial Intelligence & Data Science  
KLE College of Engineering and Technology (VTU), Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-Keertiraj2004-181717?style=flat-square&logo=github)](https://github.com/Keertiraj2004)
[![Email](https://img.shields.io/badge/Email-keertirajkamble023%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:keertirajkamble023@gmail.com)

---

## 📄 License

This project is submitted as an academic machine learning assignment. Code is free to use for learning and reference purposes.

---

> ⭐ If this helped you understand regression algorithms, give the repo a star!
