import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load datasets
train_data = pd.read_csv("train_final.csv")
test_data = pd.read_csv("test_final.csv")

# Split train data into features and target
X = train_data.drop(columns=["income>50K"])
y = train_data["income>50K"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing for numerical and categorical data
numerical_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Define pipeline with XGBoost
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
])

# Define hyperparameter search space
param_dist = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__subsample": [0.8, 1.0]
}

# Set up RandomizedSearchCV
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Train model using RandomizedSearchCV
xgb_random_search.fit(X, y)

# Best model
best_xgb_model = xgb_random_search.best_estimator_
print("Best parameters:", xgb_random_search.best_params_)

# Evaluate the model on training data
train_preds = best_xgb_model.predict_proba(X)[:, 1]
train_auc = roc_auc_score(y, train_preds)
print("Training AUC:", train_auc)

# Predict probabilities for the test dataset
test_ids = test_data["ID"]
test_predictions = best_xgb_model.predict_proba(test_data)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "ID": test_ids,
    "Prediction": test_predictions
})

submission.to_csv("xgb_submission.csv", index=False)
print("Submission file saved as xgb_submission.csv")
