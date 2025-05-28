# 📊 Customer Churn Prediction with Data Visualization

### 👩‍💻 By: Hiza Rafi  
**Register Number:** 22BDACC101  
**Course:** BCA (Big Data, Cloud Computing, Cyber Security with IBM)  
**Project Guide:** Ms. Bhoomika  

---

## 📌 Project Overview

This project focuses on predicting customer churn in a telecom company using machine learning models, and presenting key business insights using Power BI dashboards. The main goal is to help the company identify patterns leading to customer churn and take proactive measures.

---

## 🧠 Key Features

- 🔍 **Exploratory Data Analysis (EDA)** using Python (Seaborn, Matplotlib)
- 🤖 **ML Models Used:** Logistic Regression & Random Forest
- 💾 Model trained and serialized using `joblib`
- 📈 Interactive **Power BI Dashboard** with:
  - Churn rates
  - Demographics
  - Financial indicators (MonthlyCharges, Tenure, etc.)
  - Filters/Slicers for dynamic exploration
- 🌐 Deployed web app for churn prediction using trained models
- 📊 Output CSV from model integrated into dashboard

---
# 📊 Customer Churn Prediction and Data Visualization

This project predicts customer churn in the telecom sector using machine learning models — Logistic Regression and Random Forest — and visualizes the results using Seaborn and Plotly. The aim is to identify at-risk customers to help businesses reduce churn rates.

---

## 📁 Dataset

The dataset used is `Telecomdata.csv`, which contains customer demographics, service details, and churn status.

Key attributes include:
- Tenure
- Monthly & Total Charges
- Contract Type
- Tech Support Availability
- Internet Service Type
- Churn Indicator (Target)

---

## 🧹 Data Preprocessing

- Removed rows with missing `Churn` values.
- Dropped irrelevant columns: `CustomerID`, `Signup_Date`, `Churn_Date`, `PaymentMethod`, `StreamingTV`, `SeniorCitizen`.
- Encoded categorical variables: `Contract`, `TechSupport`, `InternetService`.

---

## 🔢 Sample Input for Prediction

```python
sample = pd.DataFrame([[24, 800, 32000, 2, 1, 1]], columns=X.columns)


## 🧪 How It Works

1. Data is cleaned and preprocessed (handling nulls, encoding, etc.).
2. EDA is done to understand churn patterns.
3. Models are trained to predict churn.
4. Predictions are saved in CSV and used in Power BI.
5. Dashboard provides business-friendly insights.
6. Web interface allows live prediction using input fields.

---

## 📎 Tools & Technologies

- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Power BI
- GitHub
- VS Code
- Streamlit (for web interface)
- Joblib (for model saving)

---

## 📌 Project Objectives

- Identify customers likely to churn
- Visualize patterns and risk segments
- Provide actionable business insights through a dashboard

---

## 🔮 Future Enhancements

- Integrate real-time model predictions directly into Power BI
- Add email alert system for high-risk customers
- Expand to multi-class churn segmentation (early churn, mid, loyal)

---

## ✅ Status: Completed (May 2025)

> This project was submitted as part of the final semester major project for BCA (Big Data, Cloud Computing, Cyber Security with IBM) under the guidance of Ms. Bhoomika.

---

