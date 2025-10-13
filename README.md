-----

# Polypharmacy Side Effect Prediction

## 📝 Overview

This project aims to predict the likelihood of a specific adverse side effect arising from a combination of multiple drugs (polypharmacy). Using a real-world dataset of reported drug-side effect associations, we build a machine learning model to classify a given drug combination and side effect pair as either high-risk or low-risk.

The core of the project is an **XGBoost Classifier** trained on a rich feature set engineered from the drug lists and side effect data.

## 📂 Dataset

The data is sourced from the [Drug Datasets]([https://www.google.com/search?q=https://www.kaggle.com/datasets/warcoder/drug-datasets](https://github.com/TIML-Group/HODDI/tree/main/dataset/HODDI_v1)) and includes the following key files:

  * `pos.csv`: Positive samples where a drug combination was reported with a specific side effect.
  * `neg.csv`: Negative samples where a drug combination was *not* associated with a specific side effect.
  * `Side_effects_unique.csv`: Contains pre-computed 768-dimensional embeddings for thousands of unique side effects, identified by their UMLS CUI.
  * `Drugbank_ID_SMILE_all_structure links.csv`: A mapping file to convert `DrugBankID`s to human-readable drug names.

## 🛠️ Project Pipeline

The project follows a standard machine learning workflow from data preprocessing to model evaluation and inference.

### 1\. Data Preprocessing & Sampling

  * **Data Loading:** The `pos.csv` and `neg.csv` files are loaded and concatenated into a single DataFrame.
  * **Labeling:** Positive samples (`pos.csv`) are assigned `label = 1`, and negative samples (`neg.csv`) are assigned `label = 0`.
  * **Data Cleaning:** The `DrugBankID` column, which contains string representations of lists, is parsed into actual Python lists of drug IDs. Duplicate and empty entries are removed.
  * **Balanced Sampling:** To ensure efficient training and prevent class imbalance, a balanced sample of 50,000 positive and 50,000 negative reports is created for model training and validation.

### 2\. Feature Engineering

A diverse set of features was engineered to capture information about the drug combinations and the specific side effect:

  * **Drug Combination Features (Bag-of-Words + SVD):**

    1.  The top 300 most frequently reported drugs are identified.
    2.  Each report is converted into a multi-hot encoded vector (Bag-of-Words) indicating which of these top drugs are present.
    3.  **Truncated SVD** is used to reduce the dimensionality of this sparse matrix from 300 to 64, creating dense, information-rich features.

  * **Drug Fingerprint Features (Hashing):**

    1.  A simple 256-dimension hashed fingerprint is generated for each unique drug.
    2.  For each report, the fingerprints of all drugs in the list are averaged to create a single vector representing the chemical-structural properties of the drug combination.

  * **Side Effect Embeddings:**

    1.  The pre-computed 768-dimension vector for the report's side effect (`SE_above_0.9`) is looked up from `Side_effects_unique.csv`.

  * **Numerical Features:**

      * `n_drugs`: The number of drugs in the combination.
      * `possible_pairs`: The number of unique drug-drug pairs possible ($n(n-1)/2$).
      * `time_num`: A numerical representation of the report's date (e.g., '2015Q4' -\> 2015.75).

The final feature matrix for the model is a horizontal stack of these four feature sets, resulting in a shape of `(n_samples, 1091)`.

### 3\. Model Training

An **XGBoost Classifier** (`XGBClassifier`) was chosen for its performance and efficiency.

  * **Training:** The model was trained on the engineered features from the 100k sampled dataset.
  * **Hyperparameters:** Key parameters include `n_estimators=200`, `max_depth=6`, and `learning_rate=0.05`.
  * **Optimization:** Early stopping was used with a validation set to prevent overfitting and find the optimal number of boosting rounds. The final model was then retrained on the combined training and validation data.

## 📈 Results

The model's performance was evaluated on a held-out test set.

| Metric        | Score        |
| ------------- | :----------- |
| **ROC-AUC** | **0.9059** |
| **F1-Score** | **0.8295** |
| **Accuracy** | **83%** |

**Classification Report (Test Set):**

```
              precision    recall  f1-score   support

           0       0.81      0.85      0.83      9165
           1       0.85      0.81      0.83      9705

    accuracy                           0.83     18870
   macro avg       0.83      0.83      0.83     18870
weighted avg       0.83      0.83      0.83     18870
```

The model demonstrates strong predictive power, effectively distinguishing between high-risk and low-risk drug-side effect combinations.

## 🚀 How to Use

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### Setup

1.  Download the dataset from Kaggle and place the CSV files in a directory.
2.  Update the `DATA_DIR` path in the notebook to point to your dataset directory.
3.  The training process will generate the following artifacts:
      * `xgb_fast.joblib`: The trained XGBoost model.
      * `mlb_fast.joblib`: The fitted `MultiLabelBinarizer`.
      * `svd_fast.joblib`: The fitted `TruncatedSVD` transformer.

### Making Predictions

A prediction function is available in the notebook to easily predict the risk for a new combination of drugs and a side effect.

**Example Usage:**

```python
import joblib
import numpy as np

# Load the trained model and transformers
model = joblib.load("xgb_fast.joblib")
mlb = joblib.load("mlb_fast.joblib")
svd = joblib.load("svd_fast.joblib")
# ... (plus other feature preparation steps as in the notebook)

# Example: Predict risk for Ibuprofen, Lamotrigine, and Fluoxetine
# for the side effect 'Electrocardiogram QT prolonged' (C0151878)
drug_ids = ["DB01050", "DB00555", "DB00472"]
side_effect_code = "C0151878"

# The full prediction pipeline requires generating all features
# A simplified example call would look like this:
# prediction_result = predict_with_names(drug_ids, side_effect_code)
# print(prediction_result)

# Expected output from the notebook's inference cell:
# {
#     'drugbank_ids': ['DB01050', 'DB00555', 'DB00472'],
#     'drug_names': ['Ibuprofen', 'Lamotrigine', 'Fluoxetine'],
#     'n_drugs': 3,
#     'probability': 0.46,
#     'label': 'LOW RISK',
#     'side_effect_code': 'C0151878',
#     'side_effect_name': 'Electrocardiogram QT prolonged'
# }
```

## 챌린지 & 향후 작업 (Challenges & Future Work)

  * **Inference Pipeline Bug:** The final verification cell in the notebook shows predictions being `0.0`. This suggests a potential bug or mismatch when applying the feature engineering pipeline during inference on the full dataset. This needs to be debugged for reliable deployment.
  * **Advanced Fingerprints:** Replace the simple hashed fingerprints with more chemically-aware fingerprints like Morgan Fingerprints (ECFP) using `RDKit`.
  * **Experiment with Models:** Explore deep learning architectures (e.g., LSTMs, Transformers) that could better capture the interactions within a set of drugs.
  * **Feature Enrichment:** Incorporate additional data, such as drug dosage, drug targets, or patient demographics, to improve model accuracy.
