# ⚡ CPU Resource Optimizer – ML Pipeline with DVC, MLflow & SHAP

This repository implements a machine learning pipeline for **CPU usage prediction** using **Linear Regression** and **Random Forest Regressor**.  
The pipeline is built with **DVC** for data/model versioning, **MLflow + DagsHub** for experiment tracking, and **SHAP** for model explainability.

---

## 📂 Repository Structure

```
├── Dataset/
│   ├── data-1754297123597.csv       # Versioned dataset (DVC)
│   └── data-1756126612050.csv       # Another dataset version (DVC)
├── train.py                         # Training pipeline script
├── params.yaml                      # Hyperparameters (Random Forest)
├── dvc.yaml                         # DVC pipeline definition
├── dvc.lock                         # Snapshot of pipeline state
├── LinearRegression_model.pkl       # Linear Regression trained model (DVC)
├── RandomForestRegressor_model.pkl  # Random Forest trained model (DVC)
├── README.md                        # Project documentation
```

---

## 🚀 Features

- **Two models**: Linear Regression & Random Forest Regressor
- **Version control with DVC**: datasets and trained models are tracked
- **Experiment tracking with MLflow + DagsHub**: metrics, parameters, artifacts
- **Reproducibility**: full pipeline re-runs with `dvc repro`
- **Explainability with SHAP**: summary plots & per-sample force plots
- **Evaluation metrics**: MSE, RMSE, MAE, R²

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Cpu-resource-optimizer
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Pull dataset & models (if stored remotely with DVC)**
   ```bash
   dvc pull
   ```

---

## 📊 Usage

### 1. Run the pipeline with DVC
```bash
dvc repro
```
This command:
- Pulls dataset versions
- Executes `train.py`
- Saves models (`.pkl`)
- Logs metrics & SHAP plots to MLflow/DagsHub

### 2. Run training manually
```bash
python train.py
```

### 3. View MLflow experiments
```bash
mlflow ui
```
Then visit: [https://dagshub.com/himashree56/Cpu-resource-optimizer.mlflow/#/experiments](https://dagshub.com/himashree56/Cpu-resource-optimizer.mlflow/#/experiments)

---

## 🔧 Configuration

Hyperparameters are stored in `params.yaml`:

```yaml
model:
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    n_jobs: -1
    random_state: 42
```

Modify these values and rerun with `dvc repro` to automatically retrain the models.

---

## 📈 Reproducibility

Reproducibility means ensuring experiments can be re-created exactly, even months later.

In this project:
- **Git** → Version controls code (`train.py`, `params.yaml`, `dvc.yaml`)
- **DVC** → Versions datasets & model outputs
- **dvc repro** → Reruns only necessary pipeline stages when inputs change
- **git push + dvc push** → Upload code + data to remote storage
- **MLflow** → Tracks metrics, artifacts, SHAP plots, and trained models

This guarantees:
- Every experiment is fully tracked
- Code, data, and models are always in sync
- Results can be reproduced by anyone with the repo + DVC storage

---

## 📊 Experiment Tracking

For each model run, the following are logged to **MLflow/DagsHub**:
- Parameters (from `params.yaml`)
- Metrics: **MSE, RMSE, MAE, R²**
- Artifacts: trained models (`.pkl`), SHAP plots, params file
- Git commit & DVC data hash for reproducibility

---

