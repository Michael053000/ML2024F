# Machine Learning Income Prediction

## Description
This project predicts whether a resident's yearly income exceeds $50,000 based on attributes derived from the 1994 Census database. It is based on a machine learning competition evaluating the Area Under ROC (AUC) curve.

### Data
- **train_final.csv**: Training dataset containing labeled data.
- **test_final.csv**: Unlabeled dataset for predictions.
- **Submission_sample.csv**: Sample submission format.

### Methodology
- Data preprocessing: Handled missing values, encoded categorical variables, and normalized numerical features.
- Model training: Used Random Forest, XGBoost, and LightGBM classifiers.
- Evaluation metric: AUC-ROC to measure model performance.

### Results
- Final AUC-ROC: 0.92731 (private score).

### Getting Started
Install library: `pip install pandas scikit-learn xgboost lightgbm numpy matplotlib seaborn`
