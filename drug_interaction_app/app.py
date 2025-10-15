import streamlit as st
import pandas as pd
import numpy as np
from utils import DrugInteractionPredictor
import torch
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Drug Interaction Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.risk-low {
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #28a745;
}
.risk-medium {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #ffc107;
}
.risk-high {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #dc3545;
}
.drug-input {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

class DrugInteractionApp:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
        if 'predictor' not in st.session_state:
            st.session_state.predictor = None

    def load_model(self):
        """Load the prediction model"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st.info(f"Using device: {device}")
            
            predictor = DrugInteractionPredictor('gnn_optimized.pth', device)
            success = predictor.load_model()
            
            if success:
                st.session_state.predictor = predictor
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check the model file.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

    def render_sidebar(self):
        """Render the sidebar"""
        st.sidebar.title("💊 Drug Side effect Predictor")
        st.sidebar.markdown("---")
        
        st.sidebar.subheader("About")
        st.sidebar.info(
            "This app predicts potential adverse drug interactions "
            "using a Graph Neural Network trained on polypharmacy data."
        )
        
        st.sidebar.subheader("Sample Drug IDs")
        sample_drugs = [
            "DB01050", "DB00555", "DB00472", "DB00321", 
            "DB00612", "DB00333", "DB00631", "DB01017"
        ]
        for drug in sample_drugs:
            st.sidebar.code(drug)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Risk Interpretation")
        st.sidebar.markdown("""
- 🟢 **LOW RISK** (0.0 - 0.3): Likely safe combination
- 🟡 **MEDIUM RISK** (0.3 - 0.6): Monitor for interactions
- 🟠 **HIGH RISK** (0.6 - 0.8): Significant risk
- 🔴 **VERY HIGH RISK** (0.8 - 1.0): Avoid combination
        """)

    def render_main_interface(self):
        """Render the main interface"""
        # Header
        st.markdown('<h1 class="main-header">💊 Drug Interaction Predictor</h1>', unsafe_allow_html=True)
        
        # Model loading section
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Model Status")
            if not st.session_state.model_loaded:
                st.warning("⚠️ Model not loaded. Click the button below to load the model.")
            else:
                st.success("✅ Model is loaded and ready for predictions!")
        
        with col2:
            if st.button("🔄 Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    self.load_model()
        
        st.markdown("---")
        
        # Input section
        st.subheader("🔍 Enter Drug Combinations")
        
        st.markdown('<div class="drug-input">', unsafe_allow_html=True)
        drugs_input = st.text_area(
            "Enter Drug IDs or Names (one per line, minimum 2 drugs)",
            value="DB01050\nDB00555\nDB00472",
            placeholder="DB01050\nDB00555\nDB00472",
            height=150,
            help="Enter at least 2 drugs. You can use DrugBank IDs (e.g., DB01050) or drug names."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🧪 Predict All Combinations", type="primary"):
                if st.session_state.model_loaded and st.session_state.predictor and drugs_input:
                    drug_list = [drug.strip() for drug in drugs_input.split('\n') if drug.strip()]
                    if len(drug_list) >= 2:
                        with st.spinner("Predicting all combinations..."):
                            results = st.session_state.predictor.predict_combination(drug_list)
                            st.session_state.predictions.update(results)
                    else:
                        st.error("Please enter at least 2 drug IDs")
                else:
                    st.error("Please load the model and enter drug IDs")
        
        with col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.predictions = {}
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results
        self.render_results()

    def render_results(self):
        """Render prediction results"""
        if st.session_state.predictions:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Convert to DataFrame for better display
            results_data = []
            for pair, data in st.session_state.predictions.items():
                pred_type = data.get('type', 'pairwise')
                if pred_type == 'multi-drug':
                    # Multi-drug combination
                    results_data.append({
                        'Drug Combination': pair,
                        'Type': '🔷 Multi-Drug',
                        'Risk Score': data['score'],
                        'Risk Score Str': f"{data['score']:.3f}",
                        'Interpretation': data['interpretation']
                    })
                else:
                    # Pairwise
                    results_data.append({
                        'Drug Combination': pair.replace('+', ' + '),
                        'Type': '🔗 Pairwise',
                        'Risk Score': data['score'],
                        'Risk Score Str': f"{data['score']:.3f}",
                        'Interpretation': data['interpretation']
                    })
            
            if results_data:
                df = pd.DataFrame(results_data)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart of risk scores
                    fig_bar = px.bar(
                        df,
                        x='Risk Score',
                        y='Drug Combination',
                        orientation='h',
                        title='Risk Scores by Drug Combination',
                        color='Risk Score',
                        color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                        range_color=[0, 1],
                        hover_data=['Type']
                    )
                    fig_bar.update_layout(yaxis_title="Drug Combination", xaxis_title="Risk Score")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart of risk categories
                    risk_categories = []
                    for score in df['Risk Score']:
                        if score < 0.3:
                            risk_categories.append('Low Risk')
                        elif score < 0.6:
                            risk_categories.append('Medium Risk')
                        elif score < 0.8:
                            risk_categories.append('High Risk')
                        else:
                            risk_categories.append('Very High Risk')
                    
                    df['Risk Category'] = risk_categories
                    category_counts = df['Risk Category'].value_counts()
                    
                    fig_pie = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title='Risk Distribution',
                        color=category_counts.index,
                        color_discrete_map={
                            'Low Risk': 'green',
                            'Medium Risk': 'yellow',
                            'High Risk': 'orange',
                            'Very High Risk': 'red'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Data table
                display_df = df[['Drug Combination', 'Type', 'Risk Score Str', 'Interpretation']].copy()
                display_df.columns = ['Drug Combination', 'Type', 'Risk Score', 'Interpretation']
                
                def color_risk(val):
                    score = float(val)
                    if score < 0.3:
                        return 'background-color: #d4edda; color: #155724;'
                    elif score < 0.6:
                        return 'background-color: #fff3cd; color: #856404;'
                    elif score < 0.8:
                        return 'background-color: #f8d7da; color: #721c24;'
                    else:
                        return 'background-color: #dc3545; color: white;'
                
                styled_df = display_df.style.map(color_risk, subset=['Risk Score'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Detailed view
                st.subheader("🔍 Detailed Analysis")
                
                # Show multi-drug combination first if exists
                multi_drug_results = {k: v for k, v in st.session_state.predictions.items() 
                                     if v.get('type') == 'multi-drug'}
                pairwise_results = {k: v for k, v in st.session_state.predictions.items() 
                                   if v.get('type') == 'pairwise'}
                
                if multi_drug_results:
                    st.markdown("### 💊 Multi-Drug Combination Analysis")
                    for combo, data in multi_drug_results.items():
                        score = data['score']
                        interpretation = data['interpretation']
                        side_effects = data.get('side_effects', [])
                        
                        # Determine risk class for styling
                        if score < 0.3:
                            risk_class = "risk-low"
                            emoji = "🟢"
                        elif score < 0.6:
                            risk_class = "risk-medium"
                            emoji = "🟡"
                        elif score < 0.8:
                            risk_class = "risk-high"
                            emoji = "🟠"
                        else:
                            risk_class = "risk-high"
                            emoji = "🔴"
                        
                        side_effects_html = ""
                        if side_effects:
                            side_effects_html = "<p><strong>Known Side Effects (from all pairs):</strong></p><ul>"
                            for se in side_effects[:10]:  # Show top 10 for multi-drug
                                se_name = se.get('side_effect_name', se['side_effect'])
                                side_effects_html += f"<li>{se_name} (reported {se['time']})</li>"
                            if len(side_effects) > 10:
                                side_effects_html += f"<li><em>...and {len(side_effects) - 10} more</em></li>"
                            side_effects_html += "</ul>"
                        
                        st.markdown(f"""
                        <div class="{risk_class}" style="border: 3px solid #333; margin-bottom: 20px;">
                            <h3>{emoji} {combo}</h3>
                            <p><strong>Overall Risk Score:</strong> {score:.3f}</p>
                            <p><strong>Interpretation:</strong> {interpretation}</p>
                            <p><em>This score represents the combined risk of taking all these drugs together.</em></p>
                            {side_effects_html}
                        </div>
                        """, unsafe_allow_html=True)
                
                if pairwise_results:
                    st.markdown("### 🔗 Pairwise Interaction Analysis")
                    for pair, data in pairwise_results.items():
                        score = data['score']
                        interpretation = data['interpretation']
                        side_effects = data.get('side_effects', [])
                        
                        # Determine risk class for styling
                        if score < 0.3:
                            risk_class = "risk-low"
                            emoji = "🟢"
                        elif score < 0.6:
                            risk_class = "risk-medium"
                            emoji = "🟡"
                        elif score < 0.8:
                            risk_class = "risk-high"
                            emoji = "🟠"
                        else:
                            risk_class = "risk-high"
                            emoji = "🔴"
                        
                        side_effects_html = ""
                        if side_effects:
                            side_effects_html = "<p><strong>Known Side Effects:</strong></p><ul>"
                            for se in side_effects[:5]:  # Show top 5
                                se_name = se.get('side_effect_name', se['side_effect'])
                                side_effects_html += f"<li>{se_name} (reported {se['time']})</li>"
                            side_effects_html += "</ul>"
                        
                        st.markdown(f"""
                        <div class="{risk_class}">
                            <h4>{emoji} {pair}</h4>
                            <p><strong>Risk Score:</strong> {score:.3f}</p>
                            <p><strong>Interpretation:</strong> {interpretation}</p>
                            {side_effects_html}
                        </div>
                        """, unsafe_allow_html=True)

def main():
    app = DrugInteractionApp()
    app.render_sidebar()
    app.render_main_interface()

if __name__ == "__main__":
    main()