# Solving Fintech Company Problems

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-green.svg)
![Data Science](https://img.shields.io/badge/Data%20Science-Analysis-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**A comprehensive data science project for analyzing and predicting user enrollment in fintech applications**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Connect With Me](#-connect-with-me)
- [License](#-license)

---

## 🎯 Overview

This project addresses a critical business challenge in the fintech industry: **predicting user enrollment behavior**. By analyzing user interaction patterns, screen navigation data, and temporal features, this solution helps fintech companies optimize their user onboarding process and improve conversion rates.

The project employs advanced data preprocessing techniques, exploratory data analysis (EDA), feature engineering, and machine learning algorithms to build a predictive model with high accuracy.

---

## ✨ Features

- **Data Preprocessing & Cleaning**
  - Automated data type conversion and validation
  - Handling missing values and data inconsistencies
  - Custom screen list parsing and counting

- **Exploratory Data Analysis (EDA)**
  - Comprehensive histogram visualizations for all numeric variables
  - Correlation matrix with custom heatmaps
  - Statistical summaries and data distribution analysis

- **Advanced Feature Engineering**
  - Temporal feature extraction (enrollment time difference)
  - Screen interaction funneling (Credit, Loan, Saving, CC screens)
  - Top screens identification and categorization
  - Custom feature aggregation

- **Machine Learning Model**
  - Logistic Regression with L1 regularization
  - Train-test split with 80:20 ratio
  - Feature scaling using StandardScaler
  - 10-fold cross-validation for robust evaluation

- **Model Evaluation**
  - Confusion matrix visualization
  - Accuracy scoring
  - Classification report (Precision, Recall, F1-Score)
  - Cross-validation statistics with mean and standard deviation

---

## 🛠 Technologies Used

### Programming Language
- **Python 3.7+**

### Data Analysis & Manipulation
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Data Visualization
- **Matplotlib** - Basic plotting and visualization
- **Seaborn** - Statistical data visualization

### Machine Learning
- **scikit-learn** - Machine learning algorithms and tools
  - LogisticRegression
  - StandardScaler
  - train_test_split
  - cross_val_score
  - confusion_matrix
  - accuracy_score
  - classification_report

### Utility Libraries
- **python-dateutil** - Date parsing and manipulation

---

## 📦 Installation

### Prerequisites
Ensure you have Python 3.7 or higher installed on your system.

### Clone the Repository
```bash
git clone https://github.com/ketsar28/solving-fintech-company-problems.git
cd solving-fintech-company-problems
```

### Install Required Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn python-dateutil
```

Or create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn python-dateutil
```

---

## 🚀 Usage

### Running the Analysis

1. Ensure all required CSV files are in the project directory:
   - `data_fintech.csv` - Main dataset
   - `top_screens.csv` - Top screens reference data

2. Execute the main script:
```bash
python kasus_fintech.py
```

### Output
The script will generate:
- Multiple histogram plots for numeric variables
- Correlation matrices with custom color schemes
- Time difference distribution plots
- Confusion matrix heatmap
- Model accuracy metrics and classification report
- Cross-validation results

---

## 📁 Project Structure

```
solving-fintech-company-problems/
│
├── kasus_fintech.py          # Main analysis script
├── data_fintech.csv           # Primary dataset (~11.6MB)
├── top_screens.csv            # Top screens reference data
└── README.md                  # Project documentation
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Parse and clean screen list data
- Extract hour information from timestamps
- Convert date strings to datetime objects

### 2. Exploratory Data Analysis
- Generate histograms for all numeric features
- Create correlation matrices to identify feature relationships
- Analyze target variable distribution

### 3. Feature Engineering
- **Temporal Features**: Calculate time difference between first open and enrollment
- **Screen Funneling**: Group related screens into categories:
  - Credit screens (Credit1, Credit2, Credit3, etc.)
  - Loan screens (Loan, Loan2, Loan3, Loan4)
  - Saving screens (Saving1-Saving10)
  - Credit Card screens (CC1, CC1Category, CC3)
- **Binary Features**: Create indicator variables for top screens

### 4. Data Filtering
- Apply business rule: Users who enrolled after 48 hours are marked as not enrolled

### 5. Model Training
- Split data into training (80%) and testing (20%) sets
- Apply standardization to ensure uniform feature scaling
- Train Logistic Regression model with L1 penalty
- Perform 10-fold cross-validation

### 6. Model Evaluation
- Generate confusion matrix
- Calculate accuracy, precision, recall, and F1-score
- Report cross-validation mean and standard deviation

---

## 📊 Results

The Logistic Regression model demonstrates strong performance in predicting user enrollment:

- **Standardized Features**: All features scaled for optimal model performance
- **Regularization**: L1 penalty prevents overfitting
- **Cross-Validation**: 10-fold CV ensures model generalization
- **Comprehensive Evaluation**: Multiple metrics provide complete performance picture

### Model Insights
- Identified key screens that correlate with enrollment decisions
- Temporal patterns reveal optimal enrollment windows
- Feature importance helps prioritize UI/UX improvements

---

## 🌐 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-ketsar28-181717?style=for-the-badge&logo=github)](https://github.com/ketsar28/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ketsarali-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ketsarali/)
[![Instagram](https://img.shields.io/badge/Instagram-ketsar.aaw-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/ketsar.aaw/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ketsar-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ketsar)
[![Streamlit](https://img.shields.io/badge/Streamlit-ketsar28-FF4B4B?style=for-the-badge&logo=streamlit)](https://share.streamlit.io/user/ketsar28)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-Contact-25D366?style=for-the-badge&logo=whatsapp)](https://api.whatsapp.com/send/?phone=6285155343380&text=Hello!%20I%27m%20interested%20in%20your%20fintech%20project)

</div>

---

## 📄 License

```
Copyright © 2025 Ketsar Ali

All rights reserved.

This project and its contents are the intellectual property of Ketsar Ali.
Unauthorized copying, distribution, or use of this project, via any medium,
is strictly prohibited without explicit permission from the copyright holder.

For permissions, collaborations, or inquiries, please contact via:
- Email: Available upon request
- LinkedIn: https://www.linkedin.com/in/ketsarali/
- WhatsApp: https://api.whatsapp.com/send/?phone=6285155343380
```

---

<div align="center">

**Made with ❤️ by [Ketsar Ali](https://github.com/ketsar28/)**

⭐ If you find this project useful, please consider giving it a star!

</div>
