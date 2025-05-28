from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models and input features
logistic_model = joblib.load('logistic_model.pkl')
forest_model = joblib.load('random_forest_model.pkl')
input_features = joblib.load('input_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['monthly_charges'])
    total_charges = float(request.form['total_charges'])
    contract = int(request.form['contract'])
    tech_support = int(request.form['tech_support'])
    internet_service = int(request.form['internet_service'])

    # Create a sample DataFrame for prediction
    sample = pd.DataFrame([[tenure, monthly_charges, total_charges, contract, tech_support, internet_service]], columns=input_features)

    # Make predictions
    log_prediction = logistic_model.predict(sample)[0]
    log_prob = logistic_model.predict_proba(sample)[0][1] * 100

    forest_prediction = forest_model.predict(sample)[0]
    forest_prob = forest_model.predict_proba(sample)[0][1] * 100

    return render_template('index.html',
                           log_prediction='Churn' if log_prediction == 1 else 'No Churn',
                           log_probability=f"{log_prob:.2f}%",
                           forest_prediction='Churn' if forest_prediction == 1 else 'No Churn',
                           forest_probability=f"{forest_prob:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)