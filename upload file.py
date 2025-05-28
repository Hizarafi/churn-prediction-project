import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Telecomdata.csv", dtype={"Churn": "Int64"})
df = df.dropna(subset=['Churn'])

df['Churn'] = df['Churn'].astype(int)
df['Contract'] = pd.factorize(df['Contract'])[0]
df['TechSupport'] = pd.factorize(df['TechSupport'])[0]
df['InternetService'] = pd.factorize(df['InternetService'])[0]

X = df.drop(['CustomerID', 'Signup_Date', 'Churn_Date', 'PaymentMethod',
             'StreamingTV', 'SeniorCitizen', 'Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
joblib.dump(log_model, 'logistic_model.pkl')
log_acc = accuracy_score(y_test, log_model.predict(X_test))

forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
joblib.dump(forest_model, 'random_forest_model.pkl')
forest_acc = accuracy_score(y_test, forest_model.predict(X_test))

joblib.dump(X.columns.tolist(), 'input_features.pkl')

sample = pd.DataFrame([[24, 800, 32000, 2, 1, 1]], columns=X.columns)
log_prob = log_model.predict_proba(sample)[0][1]
forest_prob = forest_model.predict_proba(sample)[0][1]

print("Logistic Regression churn probability:", format(log_prob * 100, ".2f"), "%")
print("Random Forest churn probability:", format(forest_prob * 100, ".2f"), "%")
print("Logistic Regression Accuracy:", format(log_acc * 100, ".2f"), "%")
print("Random Forest Accuracy:", format(forest_acc * 100, ".2f"), "%")
