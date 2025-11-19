# Customer Churn Prediction | Machine Learning (Python)

## Overview
Built a machine learning model to predict customers likely to churn and provide actionable insights for retention strategy.

## What I Did
- Built an end-to-end **Random Forest pipeline** with preprocessing, training, and evaluation.
- Performed **data cleaning, feature engineering, and one-hot encoding** for categorical features.
- Identified **top churn drivers** using feature importance to guide business decisions.

## Results
- **Accuracy:** 0.89  
- **ROC-AUC:** 0.92  
- **Top Features Driving Churn:**
  1. Contract type (Month-to-month)  
  2. Monthly charges  
  3. Tenure  
  4. Contract type (One year)  
  5. Internet service (Fiber optic)

## Files
- `churn_model.py` → Python script with preprocessing, model training, evaluation, and feature importance.  
- `churn_sample_data.csv` → Sample dataset for testing the model.
