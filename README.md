# FraudGuard — Credit Card Fraud Detection Demo

An interactive Streamlit demo for screening credit-card transactions using a Random Forest pipeline.

## Dataset
Expected columns:
- `TransactionID`
- `TransactionDate`
- `Amount`
- `MerchantID`
- `TransactionType`
- `Location`
- `IsFraud` (label: 1 = fraud, 0 = legitimate)

The included dataset has 2,000 rows and a 50/50 class balance. The current holdout metrics are close to chance, which suggests these fields may not contain a reliable fraud signal. This is a learning prototype, not a production fraud detector. Do not use its scores to approve, reject, or block real payments.

## 1. Train in Google Colab
1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook.
3. Upload `Fraud_Detection_Colab.ipynb` (File → Upload notebook), or copy the notebook cells into a new notebook.
4. Run the cells in order.
5. When prompted, upload `credit_card_fraud_balanced.csv`.
6. The notebook trains/evaluates the model and downloads `fraud_model.joblib` and `metrics.json`.

## 2. Put the project on GitHub
1. Create a new repository on https://github.com/
2. Upload `app.py`, `train_model.py`, `requirements.txt`, `README.md`, `fraud_model.joblib`, `metrics.json`, and `credit_card_fraud_balanced.csv`.
3. Keep the filenames unchanged because the app expects these names.

## 3. Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app** and choose your repository, branch, and `app.py` as the main file.
3. Deploy. Streamlit installs packages from `requirements.txt`.
4. If you retrain in Colab, replace the model and metrics files in GitHub with the newly downloaded versions.

## Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## UI features
- Dashboard overview and charts
- Single transaction risk score
- Batch CSV scoring with downloadable results
- Adjustable alert threshold
- Model evaluation metrics and confusion matrix

## Important model notes
Accuracy alone can be misleading in fraud detection. Review precision, recall, F1, ROC-AUC, average precision, and false positives/negatives. For a meaningful model, use representative real-world data, a time-based train/test split, and validate on data that was not used for model development. Avoid leakage and do not assume a balanced dataset reflects real-world fraud prevalence.
