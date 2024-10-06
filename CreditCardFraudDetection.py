 ## Loading Data
import pandas as pd

df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.info())
print(df.isnull().sum()) 
print(df['Class'].value_counts(normalize=True))  



## Visualize Data 
import seaborn as sns
import matplotlib.pyplot as plt

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot histogram for 'Amount'
sns.histplot(df['Amount'], bins=50, kde=False, color='blue', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Transaction Amounts', fontsize=16)
axes[0, 0].set_xlabel('Transaction Amount', fontsize=14)
axes[0, 0].set_ylabel('Frequency', fontsize=14)
axes[0, 0].grid(True)

# Plot histogram for 'Time'
sns.histplot(df['Time'], bins=50, kde=False, color='green', ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Transaction Time', fontsize=16)
axes[0, 1].set_xlabel('Time (seconds from first transaction)', fontsize=14)
axes[0, 1].set_ylabel('Frequency', fontsize=14)
axes[0, 1].grid(True)

# Plot density plot for 'Amount'
sns.kdeplot(df['Amount'], color='blue', shade=True, ax=axes[1, 0])
axes[1, 0].set_title('Density Plot of Transaction Amounts', fontsize=16)
axes[1, 0].set_xlabel('Transaction Amount', fontsize=14)
axes[1, 0].set_ylabel('Density', fontsize=14)
axes[1, 0].grid(True)

# Plot density plot for 'Time'
sns.kdeplot(df['Time'], color='green', shade=True, ax=axes[1, 1])
axes[1, 1].set_title('Density Plot of Transaction Time', fontsize=16)
axes[1, 1].set_xlabel('Time (seconds from first transaction)', fontsize=14)
axes[1, 1].set_ylabel('Density', fontsize=14)
axes[1, 1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()



## Data Preprocessing
# Feature Scaling
from sklearn.preprocessing import StandardScaler

df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)
 
# Handle Imbalanced Data
from imblearn.over_sampling import SMOTE

X = df.drop('Class', axis=1)
y = df['Class']

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
 
# Train-Test Split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)



## Model
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 

# XGBoost Classifier
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
 
# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential()
nn_model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32)
 


## Evaluation

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Define a function to evaluate the model and return confusion matrix and scores
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return cm, accuracy, precision, recall, f1

# Evaluate RandomForest model
cm_rf, acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(rf_model, X_test, y_test)

# Evaluate XGBoost model
cm_xgb, acc_xgb, prec_xgb, rec_xgb, f1_xgb = evaluate_model(xgb_model, X_test, y_test)

# Evaluate Neural Network model (note: for neural networks you might need to use predict_classes if using TensorFlow/Keras)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32")
cm_nn = confusion_matrix(y_test, y_pred_nn)

acc_nn = accuracy_score(y_test, y_pred_nn)
prec_nn = precision_score(y_test, y_pred_nn)
rec_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)

# Plot all confusion matrices in a 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot RandomForest Confusion Matrix
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title(f'Random Forest\nAccuracy: {acc_rf:.2f}, Precision: {prec_rf:.2f}, Recall: {rec_rf:.2f}, F1: {f1_rf:.2f}', fontsize=12)
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_xticks([0.5, 1.5])
axes[0].set_xticklabels(['Non-Fraud', 'Fraud'])
axes[0].set_yticks([0.5, 1.5])
axes[0].set_yticklabels(['Non-Fraud', 'Fraud'])

# Plot XGBoost Confusion Matrix
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
axes[1].set_title(f'XGBoost\nAccuracy: {acc_xgb:.2f}, Precision: {prec_xgb:.2f}, Recall: {rec_xgb:.2f}, F1: {f1_xgb:.2f}', fontsize=12)
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_xticks([0.5, 1.5])
axes[1].set_xticklabels(['Non-Fraud', 'Fraud'])
axes[1].set_yticks([0.5, 1.5])
axes[1].set_yticklabels(['Non-Fraud', 'Fraud'])

# Plot Neural Network Confusion Matrix
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[2])
axes[2].set_title(f'Neural Network\nAccuracy: {acc_nn:.2f}, Precision: {prec_nn:.2f}, Recall: {rec_nn:.2f}, F1: {f1_nn:.2f}', fontsize=12)
axes[2].set_xlabel('Predicted Labels')
axes[2].set_ylabel('True Labels')
axes[2].set_xticks([0.5, 1.5])
axes[2].set_xticklabels(['Non-Fraud', 'Fraud'])
axes[2].set_yticks([0.5, 1.5])
axes[2].set_yticklabels(['Non-Fraud', 'Fraud'])

plt.tight_layout()
plt.show()



## Model Interpretation and Visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# Define a custom scoring function for the Keras model
def keras_score(model, X, y):
    y_pred = (model.predict(X) > 0.5).astype("int32")
    return accuracy_score(y, y_pred)

# Get feature importances for RandomForest
importances_rf = rf_model.feature_importances_

# Get feature importances for XGBoost
importances_xgb = xgb_model.feature_importances_

# Get feature importances for Neural Network using permutation importance
perm_importance_nn = permutation_importance(nn_model, X_test, y_test, n_repeats=30, random_state=0, 
                                            scoring=keras_score)
importances_nn = perm_importance_nn.importances_mean

# Prepare data for plotting
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Random Forest': importances_rf,
    'XGBoost': importances_xgb,
    'Neural Network': importances_nn
})

# Melt the DataFrame for easier plotting
importance_df_melted = importance_df.melt(id_vars='Feature', var_name='Model', value_name='Importance')

# Create a combined plot
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df_melted, x='Importance', y='Feature', hue='Model', palette='viridis')
plt.title('Feature Importance for Random Forest, XGBoost, and Neural Network', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.legend(title='Model', fontsize=12)
plt.grid(axis='x')
plt.show()
 