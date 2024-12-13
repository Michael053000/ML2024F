import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV

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

# Define LightGBM pipeline
lgbm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(random_state=42))
])

# Define Bayesian Optimization parameter space
param_space = {
    "classifier__n_estimators": (100, 500),
    "classifier__learning_rate": (0.01, 0.2, "log-uniform"),
    "classifier__num_leaves": (20, 100),
    "classifier__max_depth": (5, 15),
    "classifier__min_child_samples": (10, 100),
    "classifier__subsample": (0.6, 1.0, "uniform")
}

# Bayesian Optimization with LightGBM
bayes_search = BayesSearchCV(
    estimator=lgbm_pipeline,
    search_spaces=param_space,
    scoring="roc_auc",
    n_iter=30,  # Number of parameter sets to try
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Train model using Bayesian Optimization
bayes_search.fit(X, y)

# Best model
best_lgbm_model = bayes_search.best_estimator_
print("Best parameters:", bayes_search.best_params_)

# Evaluate the model on training data
train_preds = best_lgbm_model.predict_proba(X)[:, 1]
train_auc = roc_auc_score(y, train_preds)
print("Training AUC:", train_auc)

# Predict probabilities for the test dataset
test_ids = test_data["ID"]
test_predictions = best_lgbm_model.predict_proba(test_data)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "ID": test_ids,
    "Prediction": test_predictions
})

submission.to_csv("lgbm_bayes_submission.csv", index=False)
print("Submission file saved as lgbm_bayes_submission.csv")
