# Credit-Card-Fraud-Detection
 Credit Card Fraud Detection Using Machine Learning

This repository contains the implementation of a machine learning project aimed at detecting fraudulent credit card transactions. The project demonstrates end-to-end data preprocessing, model building, and evaluation, using a highly imbalanced dataset.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Project Workflow](#project-workflow)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Feature Engineering](#2-feature-engineering)
    - [3. Model Training](#3-model-training)
    - [4. Evaluation](#4-evaluation)
5. [Key Results](#key-results)
6. [Future Enhancements](#future-enhancements)
7. [License](#license)

---

## Project Overview
Credit card fraud detection is a critical problem in the financial industry. This project focuses on building machine learning models to classify transactions as fraudulent or legitimate. The key challenge is dealing with the imbalanced nature of the dataset, where legitimate transactions vastly outnumber fraudulent ones.

The project includes:
- Cleaning and preprocessing the data.
- Addressing class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
- Building and evaluating multiple machine learning models.
- Deploying the best-performing model in a simulated environment.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Techniques**:
    - SMOTE for class imbalance
    - Logistic Regression, Random Forest, Gradient Boosted Trees
- **Tools**: Jupyter Notebook

---

## Setup and Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/Credit-Card-Fraud-Detection.git
    cd Credit-Card-Fraud-Detection
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset and place it in the `data/` folder.

---

## Project Workflow

### **1. Data Preprocessing**
- Handled missing values and outliers.
- Scaled numerical features using StandardScaler.
- Applied SMOTE to oversample the minority (fraudulent) class.

### **2. Feature Engineering**
- Engineered transaction-based features, such as:
    - Transaction amount scaling
    - Time-based features (e.g., transaction hour)
- Analyzed feature importance to identify key predictors of fraud.

### **3. Model Training**
- Trained multiple models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosted Trees
- Fine-tuned hyperparameters using GridSearchCV for optimal performance.

### **4. Evaluation**
- Evaluated models using:
    - **F1-Score**: To balance precision and recall.
    - **ROC-AUC**: To measure overall model performance.
    - Confusion Matrix for a detailed error analysis.
- Random Forest provided the best results with high precision and recall.

---

## Key Results
- The best-performing model achieved:
    - **F1-Score**: 0.92
    - **ROC-AUC**: 0.97
- Significant reduction in false negatives, ensuring better fraud detection accuracy.

---

## Future Enhancements
- Experiment with deep learning models such as LSTMs for sequential transaction data.
- Use real-time streaming data with tools like Apache Kafka for real-world applicability.
- Explore advanced ensemble methods for further accuracy improvements.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
