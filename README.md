# Customer Churn Prediction System

I built this project to answer a simple question a telecom company would care about: **which customers are likely to leave, why, and what could we do about it?**

It started as a standard model-training notebook, but I wanted it to feel like something a retention team could actually use. So the final result is a small app: you enter a customer's details and get a churn probability, a risk level, a plain explanation of what drove the prediction, and some suggested retention actions.


**Live demo:**https://churn-prediction-sy.streamlit.app


## What it does

- Predicts the probability that a customer will churn
- Flags the customer as likely to churn or stay
- Places them in one of four risk levels
- Shows which features pushed the prediction up or down (SHAP)
- Suggests retention actions based on the customer's profile
- Lets you download the result as a CSV

## The data

I used the Telco Customer Churn dataset: 7,043 customers and 21 columns, with about 27% of customers churning (1,869 of 7,043).

Cleaning steps:

- Dropped `customerID`, since it identifies a customer but doesn't help predict anything
- Fixed blank values in `TotalCharges`
- Removed 22 duplicate rows

That left **7,010 rows and 20 columns**.

## What I found in the data

A few patterns stood out during EDA:

| Customer group | Churn rate |
| --- | --- |
| Month-to-month contract | 42.6% |
| One-year contract | 11.3% |
| Two-year contract | 2.9% |
| Fiber optic internet | 41.8% |
| Electronic check payment | 45.2% |

Churned customers also had a much lower average tenure (about 18 months versus 38) and slightly higher monthly charges ($74.60 versus $61.39).

These are patterns in this dataset. They don't prove that any of these things cause churn.

## Model and results

I compared Logistic Regression with a tuned XGBoost model and went with XGBoost. Their ROC-AUC scores were almost identical (0.845 versus 0.847), but at my chosen threshold XGBoost did slightly better on every other metric.

Test set results at a 0.35 threshold:

| Metric | Tuned XGBoost | Logistic Regression |
| --- | ---: | ---: |
| Recall | 70.62% | 69.27% |
| Precision | 58.22% | 56.36% |
| Accuracy | 78.82% | 77.67% |
| F1 score | 63.82% | 62.15% |
| ROC-AUC | 0.8450 | 0.8472 |

### Why a 0.35 threshold?

The usual cutoff is 0.50, but only about 1 in 4 customers churns, so a 0.50 cutoff misses a lot of them. For retention work, missing someone who is about to leave usually costs more than sending an unnecessary offer. Lowering the threshold to 0.35 catches more churners at the cost of more false alarms.

I chose it by looking at the precision-recall trade-off. It isn't based on real business costs, which I didn't have.

## Explainability

I used SHAP so the model isn't a black box. In my notebook I looked at global feature importance, where contract type, tenure, monthly charges and internet service were among the most influential features. In the app, each prediction shows the top features pushing that specific customer toward or away from churn.

SHAP describes how the model behaves. It doesn't show what causes churn in real life.

## Risk levels and recommendations

| Churn probability | Risk level |
| --- | --- |
| Below 20% | Low |
| 20% to 35% | Medium |
| 35% to 60% | High |
| 60% and above | Very high |

The 35% boundary matches the model's threshold. The 20% and 60% boundaries are my own choices to make the output easier to read, not optimized values.

Recommendations come from simple rules, not from the model. For example, a month-to-month customer is offered a long-term contract incentive, and a customer without online security is offered that plan. They're sensible starting points for a retention team, not guaranteed fixes.

In short, XGBoost makes the prediction, SHAP explains it, rules suggest actions, and Streamlit presents it all.

## Project structure

```text
ccp/
├── app/
│   └── Streamlit.py
├── artifacts/
│   ├── xgb_model.pkl
│   ├── preprocessor.pkl
│   └── feature_names.pkl
├── data/
│   ├── raw/
│   └── processed/
├── notebook/
│   ├── 01.eda.ipynb
│   ├── 02.feature_engineering_model_training.ipynb
│   └── 03.SHAP Explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── Customer_Risk_Segmentation.py
│   └── Retention_Recommendations.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run it locally

```bash
git clone <your-repo-url>
cd ccp
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run app/Streamlit.py
```

The saved model and preprocessor depend on specific library versions, so `requirements.txt` pins scikit-learn, XGBoost and SHAP. Using different versions can cause errors when loading the `.pkl` files.

## Tech used

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Altair, Matplotlib, Seaborn and Joblib.

## Limitations

- The model gives a probability, not a certainty.
- Results come from one public dataset and may not hold on a different company's customers.
- The risk boundaries and recommendations are rule-based, not learned from data.
- I haven't measured whether the suggested actions actually reduce churn.
