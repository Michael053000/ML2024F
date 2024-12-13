# Machine Learning Income Prediction

## Description
This project predicts whether a resident's yearly income exceeds $50,000 based on attributes derived from the 1994 Census database. It is based on a machine learning competition evaluating the Area Under ROC (AUC) curve.

### Data
- **Train.csv**: Training dataset containing labeled data.
- **Test.csv**: Unlabeled dataset for predictions.
- **Submission_sample.csv**: Sample submission format.

### Methodology
- Data preprocessing: Handled missing values, encoded categorical variables, and normalized numerical features.
- Model training: Used Random Forest, XGBoost, and LightGBM classifiers.
- Evaluation metric: AUC-ROC to measure model performance.

### Results
- Final AUC-ROC: 0.92731 (private score).

### Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing: `python scripts/preprocess_data.py`
3. Train models: `python scripts/train_model.py`
4. Evaluate models: `python scripts/evaluate_model.py`

### References
- Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996.
