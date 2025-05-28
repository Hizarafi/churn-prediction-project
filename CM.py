import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# Read and clean data
df = pd.read_csv("C:/Users/MY BOOK/Downloads/churn-prediction-project/Telecomdata.csv", dtype={"Churn": "Int64"})
df = df.dropna(subset=['Churn'])  # Remove rows with missing target values

# Reduce dataset size for faster testing (optional for debugging)
df = df.sample(1000, random_state=42)  # For testing, use a smaller sample

duplicates = df.duplicated().sum()
print(f"Duplicate entries in dataset: {duplicates}")

# Encoding categorical columns
label = LabelEncoder()
df['Contract'] = label.fit_transform(df['Contract'])
df['TechSupport'] = label.fit_transform(df['TechSupport'])
df['InternetService'] = label.fit_transform(df['InternetService'])

# Features and target variable
X = df.drop(['CustomerID', 'Signup_Date', 'Churn_Date', 'PaymentMethod', 'StreamingTV', 'SeniorCitizen', 'Churn'], axis="columns")
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
predict_log = log.predict(X_test)
log_accuracy = accuracy_score(y_test, predict_log)
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

log_precision = precision_score(y_test, predict_log)
log_recall = recall_score(y_test, predict_log)
log_f1 = f1_score(y_test, predict_log)

print(f"Logistic Regression Precision: {log_precision:.2f}")
print(f"Logistic Regression Recall: {log_recall:.2f}")
print(f"Logistic Regression F1 Score: {log_f1:.2f}")


# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, predict_log)
print("Logistic Regression Confusion Matrix:")
print(cm_log)

# Saving Seaborn Heatmap for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('logistic_regression_confusion_matrix.png')  # Save to file
plt.close()  # Close to avoid unnecessary pop-ups

# ➕ Plotly Heatmap for Logistic Regression
fig_log = go.Figure(data=go.Heatmap(
    z=cm_log,
    x=['Predicted: No Churn', 'Predicted: Churn'],
    y=['Actual: No Churn', 'Actual: Churn'],
    colorscale='Blues',
    hoverongaps=False
))
fig_log.update_layout(
    title='Logistic Regression Confusion Matrix (Plotly)',
    xaxis_title='Predicted Label',
    yaxis_title='True Label'
)
fig_log.write_html("logistic_regression_confusion_matrix_plotly.html")  # Save Plotly output

# Random Forest Model
forest_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced trees for speed
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)
forest_accuracy = accuracy_score(y_test, forest_pred)
print(f"Random Forest Accuracy: {forest_accuracy:.2f}")

forest_precision = precision_score(y_test, forest_pred)
forest_recall = recall_score(y_test, forest_pred)
forest_f1 = f1_score(y_test, forest_pred)

print(f"Random Forest Precision: {forest_precision:.2f}")
print(f"Random Forest Recall: {forest_recall:.2f}")
print(f"Random Forest F1 Score: {forest_f1:.2f}")

# Random Forest Confusion Matrix
cm_forest = confusion_matrix(y_test, forest_pred)
print("Random Forest Confusion Matrix:")
print(cm_forest)

# Saving Seaborn Heatmap for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(cm_forest, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('random_forest_confusion_matrix.png')  # Save to file
plt.close()  # Close to avoid unnecessary pop-ups

# ➕ Plotly Heatmap for Random Forest
fig_rf = go.Figure(data=go.Heatmap(
    z=cm_forest,
    x=['Predicted: No Churn', 'Predicted: Churn'],
    y=['Actual: No Churn', 'Actual: Churn'],
    colorscale='Greens',
    hoverongaps=False
))
fig_rf.update_layout(
    title='Random Forest Confusion Matrix (Plotly)',
    xaxis_title='Predicted Label',
    yaxis_title='True Label'
)
fig_rf.write_html("random_forest_confusion_matrix_plotly.html")  # Save Plotly output

# Sample Prediction
sample1 = pd.DataFrame([[24, 800, 32000, 2, 1, 1]], columns=X.columns)
logprediction = log.predict(sample1)
print(f"Prediction using Logistic Regression: {logprediction[0]}")

# Random Forest Prediction
forest_custom_pred = forest_model.predict(sample1)
print(f"Prediction using Random Forest: {forest_custom_pred[0]}")

# Comparison of Predictions
comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'LogisticPredicted': predict_log[:10],
    'RandomPredicted': forest_pred[:10]
})
print("\nPrediction Comparison :")
print(comparison)

# Probabilities
log_proba = log.predict_proba(sample1)
print(f"Logistic Regression Probability: {log_proba}")
forest_proba = forest_model.predict_proba(sample1)
print(f"Random Forest Probability: {forest_proba}")

# ➕ Plotly Bar Chart for Accuracy Comparison
fig_acc = go.Figure([go.Bar(
    x=['Logistic Regression', 'Random Forest'],
    y=[log_accuracy, forest_accuracy],
    marker_color=['royalblue', 'seagreen']
)])
fig_acc.update_layout(
    title='Model Accuracy Comparison',
    yaxis_title='Accuracy',
    xaxis_title='Model'
)
fig_acc.write_html("model_accuracy_comparison.html")  # Save Plotly output

# ➕ Plotly Feature Importance Chart for Random Forest
importances = forest_model.feature_importances_
features = X.columns
indices = np.argsort(importances)

fig_feat = go.Figure(go.Bar(
    x=importances[indices],
    y=features[indices],
    orientation='h',
    marker_color='forestgreen'
))
fig_feat.update_layout(
    title='Feature Importance (Random Forest)',
    xaxis_title='Importance Score',
    yaxis_title='Features'
)
fig_feat.write_html("random_forest_feature_importance.html")  # Save Plotly output

