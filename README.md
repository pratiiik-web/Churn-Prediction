# Customer Churn Prediction

Predict which telecom customers are likely to cancel their subscription.

## Problem
Telecom companies lose revenue when customers leave. This model identifies 
high-risk customers early so the business can take action.

## Dataset
Telco Customer Churn — 7,043 customers, 21 features  
Source: Kaggle (blastchar/telco-customer-churn)

## Key Findings
- Month-to-month contract customers churn 3x more than 2-year customers
- New customers (tenure < 12 months) are highest risk
- Fiber optic + high monthly charges = strong churn signal
- Longer tenure is the single strongest predictor of retention

## Models Compared
| Model               | Accuracy | Churn Recall |
|---------------------|----------|--------------|
| Logistic Regression | 81%      | 57%          |
| Decision Tree       | 74%      | 49%          |
| Random Forest       | 79%      | 49%          |

**Winner: Logistic Regression with class_weight='balanced'**  
Final recall: 78% — correctly identifies 78% of customers who will churn.

## How to Run
pip install -r requirements.txt  
streamlit run apps/app.py

## Tech Stack
Python · Pandas · Seaborn · Scikit-learn · Streamlit

## Project Structure
Churn Prediction/
├── Data/
├── Models/
├── Notebooks/
│   └── 01_eda.ipynb
└── apps/
    └── app.py

## Demo
![App Screenshot](<img width="2880" height="1704" alt="Screenshot 2026-06-05 235204" src="https://github.com/user-attachments/assets/86d4bdbf-8709-49d3-bceb-3f200ca18496" />
)    
