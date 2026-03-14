# MLOps Lab 1 — Wine Quality Classification with W&B

> **Original Lab:** Dermatology dataset + XGBoost only  
> **This Version:** UCI Red Wine Quality dataset + XGBoost & Random Forest + Hyperparameter Sweep

---

## 📌 Modifications from Original Lab

| | Original | This Version |
|---|---|---|
| **Dataset** | UCI Dermatology (366 samples) | UCI Red Wine Quality (1599 samples) |
| **Task** | 6-class classification | Binary classification (good wine ≥ 7) |
| **Models** | XGBoost only | XGBoost + Random Forest |
| **W&B Logging** | Confusion matrix only | + ROC curve, PR curve, Feature Importance, Correlation Heatmap |
| **Extra** | None | Hyperparameter sweep (3×3 grid over `eta` × `max_depth`) |

---

## 📊 W&B Dashboard

🔗 [View Live Experiment Results](https://wandb.ai/patel-heet-northeastern-university/Lab1-wine-quality-mlops/reports/heet-s-wandb-lab--VmlldzoxNjIwMTgyMw?accessToken=5ezowkir6al0l9izvs0omah04zij40yhrtjrnpml70jml7sn54qtr86i2vlxb86g)

<img width="1320" height="960" alt="Image" src="https://github.com/user-attachments/assets/ca621956-03ba-496d-a428-96bfd4e8f923" />

---

## 🖥️ How to Recreate Results from Scratch

### Prerequisites
- Google Account (for Colab)
- W&B Account — sign up free at [wandb.ai](https://wandb.ai)

---

### Step 1 — Clone this Repo

```bash
git clone https://github.com/HP161103/wandb.git
cd wandb
```

**Or download directly:**  
Go to [github.com/HP161103/wandb](https://github.com/HP161103/wandb) → click **Code → Download ZIP**

---

### Step 2 — Open in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `wandb.ipynb` from the cloned/downloaded repo

---

### Step 3 — Install Dependencies

Run this in the **first cell** of the notebook:

```python
!pip install wandb xgboost scikit-learn pandas matplotlib seaborn -q
```

Expected output:
```
Successfully installed wandb xgboost scikit-learn ...
```

---

### Step 4 — Login to W&B

Run the login cell:

```python
import wandb
wandb.login()
```

1. It will print a link — click it
2. You'll be taken to [wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy your API key
4. Paste it back in the Colab cell and press Enter

Expected output:
```
Successfully logged in to Weights & Biases!
```

---

### Step 5 — Run All Cells

Go to **Runtime → Run all** (or press `Ctrl + F9`).

The notebook will automatically:
- ✅ Download the UCI Red Wine Quality dataset
- ✅ Preprocess and split the data (75/25 stratified split)
- ✅ Train XGBoost model and log to W&B
- ✅ Train Random Forest model and log to W&B
- ✅ Run a 9-run hyperparameter sweep (3×3 grid)
- ✅ Log confusion matrix, ROC curve, PR curve, feature importance

Total runtime: ~3–5 minutes on Colab free tier.

---

### Step 6 — View Results on W&B

After `wandb.init()` runs, you'll see a link in the cell output:

```
🚀 View run at: https://wandb.ai/YOUR-USERNAME/Lab1-wine-quality-mlops/runs/xxxxx
```

Click it to see your live dashboard.

---

## 📁 Project Structure

```
├── wandb.ipynb    # Main notebook
└── README.md      # This file
```

---

## 🧠 Model Details

### XGBoost
```python
params = {
    'objective':        'binary:logistic',
    'eta':              0.1,
    'max_depth':        6,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'eval_metric':      'logloss',
    'seed':             42
}
```

### Random Forest
```python
params = {
    'n_estimators':      300,
    'max_depth':         10,
    'min_samples_split': 5,
    'class_weight':      'balanced',
    'random_state':      42
}
```

### Hyperparameter Sweep (XGBoost)
```python
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'roc_auc', 'goal': 'maximize'},
    'parameters': {
        'eta':       {'values': [0.05, 0.1, 0.2]},
        'max_depth': {'values': [4, 6, 8]},
    }
}
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| wandb | ≥ 0.25.1 |
| xgboost | ≥ 1.7 |
| scikit-learn | ≥ 1.2 |
| pandas | ≥ 1.5 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |
| Python | 3.10 / 3.11 (not 3.14) |

> ⚠️ **Note:** wandb does not support Python 3.14 yet. Use Google Colab (Python 3.10) or a local Python 3.11 environment.

---

## ⚠️ Common Issues

| Error | Fix |
|---|---|
| `NameError: wandb is not defined` | Run the imports cell first |
| `ServicePollForTokenError` | You're on Python 3.14 — use Colab instead |
| `!pip: event not found` | You're in terminal, not a notebook cell — remove the `!` |
| W&B project stays private | Use a personal W&B account or share via Reports magic link |

---

## 👤 Author

**Heet Patel**  
Northeastern University — MLOps Course  
🔗 [github.com/HP161103/wandb](https://github.com/HP161103/wandb)
