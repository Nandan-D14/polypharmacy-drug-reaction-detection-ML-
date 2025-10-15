# Drug Interaction Prediction System

## 🎯 Overview
This system predicts drug-drug interactions and side effects using a Graph Neural Network (GNN) trained on real polypharmacy data.

## 📊 Data Sources
The system uses the following datasets from `drug-datasets/`:

1. **pos.csv** - Positive drug interactions with side effects
2. **neg.csv** - Negative drug interactions (no side effects)
3. **Side_effects_unique.csv** - Side effect names and UMLS codes
4. **DrugBankID2SMILES.csv** - Drug molecular structures

## 🏗️ System Architecture

### Model Components
- **ImprovedGraphSAGE**: Graph encoder using SAGEConv layers
- **ImprovedEdgeClassifier**: MLP for predicting interaction risk
- **Node Features**: 262-dimensional drug features
- **Hidden Dimensions**: 256 for embeddings, 128 for classifier

### Data Processing
1. Loads positive/negative interaction data
2. Extracts unique drugs and builds drug-to-index mapping
3. Creates graph edges from known interactions
4. Stores side effect information for each drug pair

## 🚀 Features

### 1. Single Drug Pair Prediction
- Input: Two DrugBank IDs (e.g., DB01050, DB00555)
- Output: Risk score (0-1) and interpretation
- Shows known side effects from historical data

### 2. Multiple Drug Combination
- Input: 3+ DrugBank IDs
- Output: All pairwise interactions
- Example: 3 drugs → 3 predictions, 4 drugs → 6 predictions

### 3. Risk Interpretation
- 🟢 **LOW RISK** (0.0 - 0.3): Likely safe combination
- 🟡 **MEDIUM RISK** (0.3 - 0.6): Monitor for interactions
- 🟠 **HIGH RISK** (0.6 - 0.8): Significant risk
- 🔴 **VERY HIGH RISK** (0.8 - 1.0): Avoid combination

### 4. Side Effect Information
- Displays known side effects from dataset
- Shows UMLS codes and human-readable names
- Includes reporting time period

## 📁 File Structure
```
drug_interaction_app/
├── app.py                  # Streamlit UI
├── utils.py                # Model and prediction logic
├── gnn_optimized.pth       # Trained model weights
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔧 Usage

### Installation
```bash
cd drug_interaction_app
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

### Using the Interface
1. Click "🔄 Load Model" to initialize
2. Enter drug IDs in the input fields
3. Click "🔬 Predict Interaction" or "🧪 Predict All Combinations"
4. View results with risk scores and side effects

## 📝 Sample Drug IDs
- DB01050, DB00555, DB00472
- DB00321, DB00612, DB00333
- DB00631, DB01017

## ⚠️ Important Notes

### Current Limitations
1. **Mock Node Features**: Currently using random features. For production:
   - Load actual molecular features from SMILES
   - Use pre-computed drug embeddings
   - Extract features from drug properties

2. **Dataset Path**: Assumes `../drug-datasets/` relative path
   - Adjust path if datasets are in different location

3. **Side Effects**: Only shows effects from training data
   - New drug combinations may not have historical data

### Future Improvements
1. Load real molecular features from SMILES strings
2. Add drug name search (not just IDs)
3. Include confidence intervals
4. Add more detailed side effect probabilities
5. Support for drug name autocomplete
6. Export results to PDF/CSV

## 🔬 Technical Details

### Model Architecture
```python
ImprovedGraphSAGE(
    in_channels=262,
    hidden_channels=256,
    n_layers=2,
    dropout=0.3
)

ImprovedEdgeClassifier(
    node_emb_dim=256,
    hidden=128,
    dropout=0.3
)
```

### Prediction Pipeline
1. Load drug pair (A, B)
2. Get node indices from drug_to_idx mapping
3. Compute node embeddings using GNN encoder
4. Concatenate embeddings [emb_A, emb_B]
5. Pass through edge classifier
6. Apply sigmoid for probability
7. Lookup known side effects from dataset

## 📊 Dataset Statistics
- **Drugs**: Extracted from pos.csv and neg.csv
- **Interactions**: Built from positive interaction records
- **Side Effects**: 768 unique side effects tracked
- **Time Range**: 2015Q4 - 2024Q3

## 🤝 Contributing
To improve the system:
1. Add real molecular features
2. Implement drug name search
3. Add more visualization options
4. Improve side effect probability estimation

## 📄 License
Educational/Research Use
