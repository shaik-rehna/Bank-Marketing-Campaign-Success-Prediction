# Bank Marketing Campaign Success Prediction

## Overview

This project develops machine learning models to predict whether a bank marketing campaign will successfully convert a customer into a term-deposit subscriber. The objective is to identify customers who are more likely to respond positively to a campaign, enabling more targeted marketing and efficient allocation of campaign resources.

## Dataset

The project uses the [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data) from Kaggle.

- **Number of records:** 11,162
- **Number of features:** 17
- **Task:** Binary classification
- **Target variable:** `deposit`
  - `0` → Customer did not subscribe
  - `1` → Customer subscribed

The dataset contains customer demographic information, financial attributes, and details of previous marketing interactions.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure and predictive characteristics of the dataset.

The analysis included:

- Examining distributions of continuous variables
- Analyzing correlations between features and the target
- Checking variance of numerical variables
- Studying the distribution of categorical and binary variables
- Identifying highly imbalanced categorical features
- Investigating potentially redundant or low-information features

Based on the EDA, relevant features were retained while features with very low information content and extreme imbalance were considered for removal.

## Data Preprocessing

The preprocessing pipeline included:

- Feature selection based on EDA and target relationships
- Log transformation of highly skewed numerical variables
- Standardization of numerical features where required
- Encoding categorical variables into numerical representations
- Grouping related categorical values where appropriate
- Separating the dataset into training and held-out test sets

The test set was kept separate from model selection and hyperparameter tuning to provide an unbiased estimate of final model performance.

## Model Development

Three classification approaches were implemented and compared:

### 1. RBF Kernel SVM

An SVM with an RBF kernel was used to model non-linear relationships between customer attributes and campaign outcomes.

Hyperparameters tuned using 5-fold GridSearchCV:

- `C` — regularization parameter
- `gamma` — RBF kernel width parameter

### 2. Neural Network

A feed-forward neural network with a single ReLU hidden layer was implemented.

Hyperparameters tuned using 5-fold GridSearchCV:

- Number of neurons in the hidden layer
- Weight decay (`alpha`)

### 3. Random Forest

A Random Forest classifier was trained to capture non-linear feature interactions and relationships.

Hyperparameters tuned using 5-fold GridSearchCV:

- Maximum tree depth (`max_depth`)
- Maximum number of features considered at each split (`max_features`)

## Hyperparameter Optimization

Five-fold cross-validation with `GridSearchCV` was used to systematically evaluate different hyperparameter combinations.

This ensured that hyperparameters were selected using only the training data, while the held-out test set was reserved for final evaluation.

## Results

| Model | Test Accuracy |
|---|---:|
| **Random Forest** | **85.85%** |
| Neural Network | 85.62% |
| SVM (RBF) | 85.35% |

The Random Forest achieved the highest test accuracy of **85.85%**, followed closely by the Neural Network and RBF-SVM.

## Key Findings

- All three models achieved comparable performance, with test accuracy above 85%.
- Random Forest provided the best overall test accuracy.
- The results indicate that non-linear classification models can effectively capture relationships between customer characteristics and marketing campaign outcomes.
- Feature analysis and preprocessing played an important role in preparing the dataset for model training.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- GridSearchCV
