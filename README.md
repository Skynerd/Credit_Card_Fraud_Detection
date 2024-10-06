# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using various machine learning models. It utilizes the **Credit Card Fraud Detection Dataset** from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized transaction features and labels indicating fraud or non-fraud.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [Random Forest](#random-forest)
  - [XGBoost](#xgboost)
  - [Neural Network](#neural-network)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Conclusion](#conclusion)

## Project Overview

The primary objective of this project is to identify fraudulent credit card transactions using machine learning models. The dataset is highly imbalanced, with only 0.17% of transactions labeled as fraudulent. We use different techniques to handle this imbalance, including **SMOTE** for oversampling the minority class.

The models used for this project include:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Neural Network**

We compare the performance of these models based on metrics such as accuracy, precision, recall, and F1-score.

## Dataset

The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, consisting of 284,807 transactions and 31 features, including the target variable `Class` (1 for fraud, 0 for non-fraud).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

## Exploratory Data Analysis (EDA)

We begin by analyzing the dataset, checking for missing values, and visualizing the distribution of key features (`Amount`, `Time`). Here are some of the visualizations:

- **Histogram and Density Plots for `Amount` and `Time`**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.histplot(df['Amount'], bins=50)
    plt.title('Transaction Amount Distribution')
    plt.show()

    sns.kdeplot(df['Time'], shade=True)
    plt.title('Transaction Time Distribution')
    plt.show()
    ```

## Data Preprocessing

1. **Feature Scaling**: Since `Amount` and `Time` are on different scales, we apply **Standard Scaling** to normalize these features.
![VisualizeData](https://github.com/Skynerd/Credit_Card_Fraud_Detection/blob/main/DemoPlots/VisualizeData.png)

2. **Handling Imbalanced Data**: We use **SMOTE** to oversample the minority class (fraudulent transactions).
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    ```

## Modeling

### Random Forest
We first train a Random Forest classifier with 100 trees.
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 
```

### XGBoost
Next, we train an XGBoost classifier.
```python
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
```

### Neural Network
Finally, we build a simple feedforward neural network using **Keras**.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential()
nn_model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## Evaluation

We evaluate all three models using accuracy, precision, recall, and F1-score. Since the dataset is imbalanced, precision and recall are the key metrics.

- **Confusion Matrix and Metrics**:
    ```python
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1
    ```

The confusion matrices for each model are visualized using **Seaborn**.
![ModelEvaluation](https://github.com/Skynerd/Credit_Card_Fraud_Detection/blob/main/DemoPlots/ModelEvaluation.png)

## Feature Importance

We examine the importance of features for each model:

- **Random Forest**:
    ```python
    importances_rf = rf_model.feature_importances_
    sns.barplot(x=importances_rf, y=features)
    plt.title('Feature Importance - Random Forest')
    ```

- **XGBoost**:
    ```python
    importances_xgb = xgb_model.feature_importances_
    sns.barplot(x=importances_xgb, y=features)
    plt.title('Feature Importance - XGBoost')
    ```

- **Neural Network** (using **permutation importance**):
    ```python
    from sklearn.inspection import permutation_importance
    perm_importance_nn = permutation_importance(nn_model, X_test, y_test)
    sns.barplot(x=perm_importance_nn.importances_mean, y=features)
    plt.title('Feature Importance - Neural Network')
    ```

![ModelInterpretation&Visualization](https://github.com/Skynerd/Credit_Card_Fraud_Detection/blob/main/DemoPlots/ModelInterpretation%26Visualization.png)

## Conclusion

In this project, we successfully built three different machine learning models to detect fraudulent transactions. While all models performed well, further tuning and experimentation could help improve their precision and recall. The Random Forest and XGBoost models provide the most interpretable results through feature importance visualization.
