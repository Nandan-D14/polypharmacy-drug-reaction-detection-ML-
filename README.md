# 💊 Polypharmacy Drug Reaction Detection using Machine Learning

A Graph Neural Network (GNN) based system for predicting adverse drug-drug interactions and side effects in polypharmacy scenarios.

## 🎯 Overview

This project uses a Graph Neural Network to predict potential adverse reactions when multiple drugs are taken together. The system analyzes drug combinations and provides risk scores along with known side effects from historical data.

## ✨ Features

- **Multi-Drug Prediction**: Analyze interactions for 2+ drugs simultaneously
- **Pairwise Analysis**: Individual drug pair interaction predictions
- **Risk Scoring**: 0-1 risk scores with interpretations (Low/Medium/High/Very High)
- **Side Effect Information**: Shows known side effects from historical data
- **Interactive Visualizations**: Bar charts and pie charts for risk analysis
- **Real-time Predictions**: Instant results using trained GNN model

## 🏗️ Architecture

### Model Components
- **ImprovedGraphSAGE**: Graph encoder using SAGEConv layers
  - Input: 262-dimensional drug features
  - Hidden: 256 dimensions
  - Layers: 2 with dropout (0.3)
  
- **ImprovedEdgeClassifier**: MLP for interaction prediction
  - Input: Concatenated node embeddings (512-dim)
  - Hidden: 128 dimensions
  - Output: Binary interaction probability

### Technology Stack
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data processing

## 📁 Project Structure

```
ML_mini_project/
├── drug_interaction_app/       # Streamlit web application
│   ├── app.py                  # Main UI
│   ├── utils.py                # Model & prediction logic
│   ├── gnn_optimized.pth       # Trained model weights
│   ├── requirements.txt        # Dependencies
│   └── README.md              # App documentation
├── models/                     # Model training
│   ├── gnn-polypharmacy.ipynb # Training notebook
│   └── gnn_optimized.pth      # Model checkpoint
├── drug-datasets/             # Dataset files (not in repo)
│   ├── pos.csv               # Positive interactions
│   ├── neg.csv               # Negative interactions
│   └── Side_effects_unique.csv
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nandan-D14/polypharmacy-drug-reaction-detection-ML-.git
cd polypharmacy-drug-reaction-detection-ML-
```

2. Install dependencies:
```bash
cd drug_interaction_app
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## 💻 Usage

### Web Application

1. **Load Model**: Click "🔄 Load Model" to initialize the GNN
2. **Enter Drugs**: Input 2+ DrugBank IDs (one per line)
   ```
   DB01050
   DB00555
   DB00472
   ```
3. **Predict**: Click "🧪 Predict All Combinations"
4. **View Results**:
   - Multi-drug combination risk (for 3+ drugs)
   - Pairwise interaction risks
   - Interactive charts
   - Known side effects

### Sample Drug IDs
- DB01050, DB00555, DB00472
- DB00321, DB00612, DB00333
- DB00631, DB01017

## 📊 Model Training

The model was trained on polypharmacy data including:
- **Positive interactions**: Drug combinations with reported side effects
- **Negative interactions**: Safe drug combinations
- **Side effects**: 768 unique side effects tracked
- **Time range**: 2015Q4 - 2024Q3

### Training Process
1. Load drug interaction data
2. Build graph structure (drugs as nodes, interactions as edges)
3. Extract molecular features (262-dim)
4. Train GraphSAGE encoder + Edge classifier
5. Optimize using binary cross-entropy loss

See `models/gnn-polypharmacy.ipynb` for training details.

## 🎨 Visualizations

### Risk Score Bar Chart
- Horizontal bars showing risk for each combination
- Color gradient: Green (safe) → Red (dangerous)
- Hover for details

### Risk Distribution Pie Chart
- Breakdown by risk category
- Color-coded segments
- Percentage distribution

### Data Table
- Drug combinations
- Type (Multi-drug or Pairwise)
- Risk scores
- Interpretations

## 📈 Risk Interpretation

| Risk Score | Category | Interpretation |
|------------|----------|----------------|
| 0.0 - 0.3  | 🟢 Low Risk | Likely safe combination |
| 0.3 - 0.6  | 🟡 Medium Risk | Monitor for interactions |
| 0.6 - 0.8  | 🟠 High Risk | Significant interaction risk |
| 0.8 - 1.0  | 🔴 Very High Risk | Avoid this combination |

## 🔬 Multi-Drug Prediction Algorithm

For 3+ drugs, the system:
1. Computes all pairwise interactions
2. Aggregates risk scores:
   - Max risk (70% weight)
   - Average risk (30% weight)
3. Combines side effects from all pairs
4. Returns overall risk assessment

**Formula**: `Final Risk = (Max × 0.7) + (Average × 0.3)`

## 📝 Dataset

The project uses the following datasets (not included in repo):
- `pos.csv`: Positive drug interactions with side effects
- `neg.csv`: Negative drug interactions
- `Side_effects_unique.csv`: Side effect names and UMLS codes
- `DrugBankID2SMILES.csv`: Drug molecular structures

**Note**: Datasets are loaded in chunks (10,000 rows) for performance optimization.

## ⚠️ Important Notes

- This is a research/educational tool
- **Not a substitute for medical advice**
- Consult healthcare professionals for actual prescriptions
- Predictions based on historical data
- New drug combinations may not have historical records

## 🛠️ Performance Optimizations

- **Chunked data loading**: Processes large datasets incrementally
- **Memory efficient**: Only loads necessary columns
- **Fast predictions**: Pre-computed embeddings
- **Responsive UI**: Streamlit caching

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add drug name search (not just IDs)
- Implement molecular feature extraction from SMILES
- Add confidence intervals
- Export results to PDF/CSV
- Network graph visualization
- Temporal analysis of side effects

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

Nandan D
- GitHub: [@Nandan-D14](https://github.com/Nandan-D14)

## 🙏 Acknowledgments

- DrugBank for drug data
- PyTorch Geometric team
- Streamlit community

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⚡ Built with PyTorch, PyTorch Geometric, and Streamlit**
