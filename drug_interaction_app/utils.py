import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import pandas as pd
import joblib
import ast
import os

class ImprovedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(n_layers-1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        return x

class ImprovedEdgeClassifier(nn.Module):
    def __init__(self, node_emb_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, ha, hb):
        h = torch.cat([ha, hb], dim=1)
        return self.mlp(h).squeeze(1)

class DrugInteractionPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.encoder = None
        self.edge_clf = None
        self.drug_to_idx = None
        self.node_features = None
        self.edge_index = None

    def load_datasets(self):
        """Load the actual drug datasets in chunks to prevent lagging"""
        try:
            print("📥 Loading datasets in chunks...")
            
            # Load positive and negative interaction data in chunks
            chunk_size = 10000
            all_drugs = set()
            edges = []
            self.interaction_data = {}
            
            # Process positive interactions in chunks
            for chunk in pd.read_csv('../drug-datasets/pos.csv', chunksize=chunk_size):
                for drug_list_str in chunk['DrugBankID']:
                    try:
                        drug_list = ast.literal_eval(drug_list_str)
                        all_drugs.update(drug_list)
                    except:
                        continue
            
            # Process negative interactions for drug list
            for chunk in pd.read_csv('../drug-datasets/neg.csv', chunksize=chunk_size):
                for drug_list_str in chunk['DrugBankID']:
                    try:
                        drug_list = ast.literal_eval(drug_list_str)
                        all_drugs.update(drug_list)
                    except:
                        continue
            
            all_drugs = sorted(list(all_drugs))
            self.drug_to_idx = {drug: idx for idx, drug in enumerate(all_drugs)}
            self.idx_to_drug = {idx: drug for drug, idx in self.drug_to_idx.items()}
            
            print(f"✅ Loaded {len(all_drugs)} unique drugs")
            
            # Load side effects data (only first 2 columns for mapping)
            self.side_effects_df = pd.read_csv(
                '../drug-datasets/Side_effects_unique.csv',
                usecols=['umls_cui_from_meddra', 'side_effect_name']
            )
            
            print("✅ Loaded side effects mapping")
            
            # Build edge index from positive interactions (in chunks)
            print("🔗 Building interaction graph...")
            for chunk in pd.read_csv('../drug-datasets/pos.csv', chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    try:
                        drug_list = ast.literal_eval(row['DrugBankID'])
                        side_effect = row['SE_above_0.9']
                        
                        # Create edges for all drug pairs in this interaction
                        for i in range(len(drug_list)):
                            for j in range(i + 1, len(drug_list)):
                                if drug_list[i] in self.drug_to_idx and drug_list[j] in self.drug_to_idx:
                                    idx_i = self.drug_to_idx[drug_list[i]]
                                    idx_j = self.drug_to_idx[drug_list[j]]
                                    edges.append([idx_i, idx_j])
                                    edges.append([idx_j, idx_i])
                                    
                                    # Store interaction info
                                    pair_key = tuple(sorted([drug_list[i], drug_list[j]]))
                                    if pair_key not in self.interaction_data:
                                        self.interaction_data[pair_key] = []
                                    self.interaction_data[pair_key].append({
                                        'side_effect': side_effect,
                                        'label': 1,
                                        'time': row['time']
                                    })
                    except:
                        continue
            
            print(f"✅ Built graph with {len(edges)} edges")
            return all_drugs, edges
        except Exception as e:
            print(f"⚠️ Could not load datasets: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_model(self):
        """Load the trained model and necessary data"""
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize models
            in_dim = 262  # Based on your node feature dimension
            self.encoder = ImprovedGraphSAGE(in_dim).to(self.device)
            self.edge_clf = ImprovedEdgeClassifier(256).to(self.device)
            
            # Load weights
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.edge_clf.load_state_dict(checkpoint['edge_clf'])
            
            self.encoder.eval()
            self.edge_clf.eval()
            
            # Try to load real datasets
            all_drugs, edges = self.load_datasets()
            
            if all_drugs is None:
                # Fallback to sample data
                print("⚠️ Using sample data for demo")
                all_drugs = [
                    "DB01050", "DB00555", "DB00472", "DB00321", 
                    "DB00612", "DB00333", "DB00631", "DB01017",
                    "DB00001", "DB00002", "DB00003", "DB00004"
                ]
                self.drug_to_idx = {drug: idx for idx, drug in enumerate(all_drugs)}
                
                # Create mock edges
                edges = []
                for i in range(len(all_drugs)):
                    for j in range(i + 1, len(all_drugs)):
                        edges.append([i, j])
                        edges.append([j, i])
            
            num_nodes = len(all_drugs)
            
            # Create node features (mock for now - in production, use molecular features)
            self.node_features = torch.randn(num_nodes, in_dim).to(self.device)
            
            # Create edge index
            if edges:
                self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
            else:
                # Empty graph fallback
                self.edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            
            print("✅ Model loaded successfully!")
            print(f"📊 Loaded {num_nodes} drugs with {self.edge_index.shape[1]} edges")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_side_effect_name(self, umls_code):
        """Get side effect name from UMLS code"""
        if hasattr(self, 'side_effects_df'):
            match = self.side_effects_df[self.side_effects_df['umls_cui_from_meddra'] == umls_code]
            if not match.empty:
                return match.iloc[0]['side_effect_name']
        return umls_code
    
    def get_known_side_effects(self, drug_a, drug_b):
        """Get known side effects for a drug pair from the dataset"""
        if not hasattr(self, 'interaction_data'):
            return []
        
        pair_key = tuple(sorted([drug_a, drug_b]))
        if pair_key in self.interaction_data:
            # Add side effect names
            effects = []
            for effect in self.interaction_data[pair_key]:
                effect_copy = effect.copy()
                effect_copy['side_effect_name'] = self.get_side_effect_name(effect['side_effect'])
                effects.append(effect_copy)
            return effects
        return []
    
    def predict_interaction(self, drug_a, drug_b):
        """Predict interaction between two drugs"""
        if self.encoder is None or self.edge_clf is None:
            return "Model not loaded", 0.0, []
        
        try:
            with torch.no_grad():
                # Check if drugs are in database
                if drug_a not in self.drug_to_idx:
                    return f"Drug {drug_a} not found in database", 0.0, []
                if drug_b not in self.drug_to_idx:
                    return f"Drug {drug_b} not found in database", 0.0, []
                
                # Get node embeddings
                node_emb = self.encoder(self.node_features, self.edge_index)
                
                # Get drug indices
                u_idx = self.drug_to_idx[drug_a]
                v_idx = self.drug_to_idx[drug_b]
                
                # Get embeddings for the two drugs
                ha = node_emb[u_idx]
                hb = node_emb[v_idx]
                
                # Predict interaction
                logit = self.edge_clf(ha.unsqueeze(0), hb.unsqueeze(0))
                risk_score = torch.sigmoid(logit).item()
                
                # Get known side effects
                side_effects = self.get_known_side_effects(drug_a, drug_b)
                
                return self.interpret_risk(risk_score), risk_score, side_effects
                
        except Exception as e:
            return f"Prediction error: {e}", 0.0, []

    def interpret_risk(self, risk_score):
        """Interpret the risk score"""
        if risk_score < 0.3:
            return "🟢 LOW RISK - Likely safe combination"
        elif risk_score < 0.6:
            return "🟡 MEDIUM RISK - Monitor for potential interactions"
        elif risk_score < 0.8:
            return "🟠 HIGH RISK - Significant interaction risk"
        else:
            return "🔴 VERY HIGH RISK - Avoid this combination"

    def predict_multi_drug_interaction(self, drug_list):
        """Predict interaction for multiple drugs as a single combination"""
        if self.encoder is None or self.edge_clf is None:
            return "Model not loaded", 0.0, []
        
        try:
            with torch.no_grad():
                # Check if all drugs are in database
                missing_drugs = [d for d in drug_list if d not in self.drug_to_idx]
                if missing_drugs:
                    return f"Drugs not found: {', '.join(missing_drugs)}", 0.0, []
                
                # Get node embeddings
                node_emb = self.encoder(self.node_features, self.edge_index)
                
                # Get embeddings for all drugs
                drug_indices = [self.drug_to_idx[drug] for drug in drug_list]
                drug_embeddings = [node_emb[idx] for idx in drug_indices]
                
                # Aggregate embeddings (mean pooling)
                combined_emb = torch.stack(drug_embeddings).mean(dim=0)
                
                # Predict pairwise interactions and aggregate
                all_scores = []
                all_side_effects = []
                
                for i in range(len(drug_list)):
                    for j in range(i + 1, len(drug_list)):
                        ha = drug_embeddings[i]
                        hb = drug_embeddings[j]
                        
                        logit = self.edge_clf(ha.unsqueeze(0), hb.unsqueeze(0))
                        score = torch.sigmoid(logit).item()
                        all_scores.append(score)
                        
                        # Get side effects for this pair
                        side_effects = self.get_known_side_effects(drug_list[i], drug_list[j])
                        all_side_effects.extend(side_effects)
                
                # Aggregate risk score (max or mean)
                if all_scores:
                    # Use max risk as overall risk
                    max_risk = max(all_scores)
                    avg_risk = sum(all_scores) / len(all_scores)
                    
                    # Combine both for final score
                    final_risk = (max_risk * 0.7 + avg_risk * 0.3)
                else:
                    final_risk = 0.0
                
                # Remove duplicate side effects
                unique_side_effects = []
                seen = set()
                for se in all_side_effects:
                    key = se['side_effect']
                    if key not in seen:
                        seen.add(key)
                        unique_side_effects.append(se)
                
                return self.interpret_risk(final_risk), final_risk, unique_side_effects
                
        except Exception as e:
            return f"Prediction error: {e}", 0.0, []
    
    def predict_combination(self, drug_list):
        """Predict interactions for multiple drugs (both pairwise and combined)"""
        results = {}
        
        # Pairwise predictions
        for i in range(len(drug_list)):
            for j in range(i + 1, len(drug_list)):
                drug_a, drug_b = drug_list[i], drug_list[j]
                interpretation, score, side_effects = self.predict_interaction(drug_a, drug_b)
                results[f"{drug_a} + {drug_b}"] = {
                    'interpretation': interpretation,
                    'score': score,
                    'side_effects': side_effects,
                    'type': 'pairwise'
                }
        
        # Multi-drug combination prediction
        if len(drug_list) > 2:
            interpretation, score, side_effects = self.predict_multi_drug_interaction(drug_list)
            drug_combo = " + ".join(drug_list)
            results[drug_combo] = {
                'interpretation': interpretation,
                'score': score,
                'side_effects': side_effects,
                'type': 'multi-drug'
            }
        
        return results