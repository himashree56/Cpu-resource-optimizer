import dagshub
import mlflow
import pandas as pd
import numpy as np
import joblib
import yaml
import subprocess
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Connect MLflow to DagsHub
# -----------------------------
dagshub.init(repo_owner="himashree56", repo_name="Cpu-resource-optimizer", mlflow=True)
mlflow.set_experiment("CPU Optimization Experiments")

# -----------------------------
# 2. Load dataset
# -----------------------------
data = pd.read_csv("Dataset/data-1756126612050.csv")

target_col = "cpu_usage"
feature_cols = ["cpu_request", "mem_request", "cpu_limit", "mem_limit", "runtime_minutes", "controller_kind"]

X = data[feature_cols]
y = data[target_col]

# -----------------------------
# 3. Preprocessing pipeline
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------------
# 4. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Load hyperparameters from params.yaml
# -----------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

rf_params = params["model"]["random_forest"]

# -----------------------------
# 6. Utility: log Git + DVC metadata + params.yaml
# -----------------------------
def log_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        mlflow.log_param("git_commit", commit)
    except Exception:
        print("⚠️ Could not log Git commit")

    try:
        with open("Dataset/data-1756126612050.csv.dvc") as f:
            for line in f:
                if line.strip().startswith("md5:"):
                    dvc_hash = line.split(":")[1].strip()
                    mlflow.log_param("dvc_data_hash", dvc_hash)
                    break
    except Exception:
        print("⚠️ Could not log DVC data hash")

    try:
        if os.path.exists("params.yaml"):
            mlflow.log_artifact("params.yaml")
    except Exception:
        print("⚠️ Could not log params.yaml")

# -----------------------------
# 7. Train & Log models with MLflow
# -----------------------------
def train_and_log(model, model_name):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        log_metadata()

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Save model locally and log as artifact
        model_filename = f"{model_name}_model.pkl"
        joblib.dump(pipeline, model_filename)
        mlflow.log_artifact(model_filename)

        print(f"✅ {model_name} training complete. Metrics, params.yaml & model logged to DagsHub 🚀")

# -----------------------------
# 8. Run both models
# -----------------------------
train_and_log(LinearRegression(), "LinearRegression")
train_and_log(
    RandomForestRegressor(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        n_jobs=rf_params["n_jobs"],
        random_state=rf_params["random_state"],
    ),
    "RandomForestRegressor",
)

print("🏁 All done!")
