import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_inference import ExoplanetModelInference, get_model_instance
from feature_processor import ExoplanetFeatureProcessor
from shap_explainer import ExoplanetExplainer

# Page configuration
st.set_page_config(
    page_title="Exoplanet Classifier", 
    page_icon="🔭", 
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
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .candidate { background-color: #fff3cd; border: 2px solid #ffc107; }
    .confirmed { background-color: #d4edda; border: 2px solid #28a745; }
    .false-positive { background-color: #f8d7da; border: 2px solid #dc3545; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model once and cache it."""
    try:
        model = ExoplanetModelInference()
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_schema():
    """Load feature schema."""
    try:
        schema_path = Path(__file__).parent.parent / "artifacts" / "schema.json"
        with open(schema_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load schema: {e}")
        return {}

def create_input_form(schema):
    """Create input form based on schema configuration."""
    st.header("🌍 Exoplanet Parameters")
    st.markdown("Enter the observed characteristics of your celestial object:")
    
    inputs = {}
    
    # Get numeric features from schema
    numeric_features = schema.get('numeric_features', [])
    
    # Organize features into categories for better UX
    orbital_features = []
    stellar_features = []
    signal_features = []
    
    for feature in numeric_features:
        name = feature['name']
        if any(keyword in name for keyword in ['orbital', 'period', 'duration', 'transit']):
            orbital_features.append(feature)
        elif any(keyword in name for keyword in ['stellar', 'star']):
            stellar_features.append(feature)
        else:
            signal_features.append(feature)
    
    # Create tabs for different feature categories
    tab1, tab2, tab3 = st.tabs(["🪐 Orbital & Transit", "⭐ Stellar Properties", "📊 Signal Characteristics"])
    
    with tab1:
        st.subheader("Orbital and Transit Properties")
        col1, col2 = st.columns(2)
        
        with col1:
            for i, feature in enumerate(orbital_features):
                if i % 2 == 0:  # Even indices go in first column
                    # Ensure all values have consistent types
                    min_val = float(feature['min'])
                    max_val = float(feature['max'])
                    default_val = float(feature['default'])
                    step_val = feature.get('step', (max_val - min_val) / 100)
                    
                    inputs[feature['name']] = st.number_input(
                        feature['label'],
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=float(step_val),
                        help=f"Range: {min_val} - {max_val}"
                    )
        
        with col2:
            for i, feature in enumerate(orbital_features):
                if i % 2 == 1:  # Odd indices go in second column
                    # Ensure all values have consistent types
                    min_val = float(feature['min'])
                    max_val = float(feature['max'])
                    default_val = float(feature['default'])
                    step_val = feature.get('step', (max_val - min_val) / 100)
                    
                    inputs[feature['name']] = st.number_input(
                        feature['label'],
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=float(step_val),
                        help=f"Range: {min_val} - {max_val}"
                    )
    
    with tab2:
        st.subheader("Stellar Properties")
        col1, col2 = st.columns(2)
        
        with col1:
            for i, feature in enumerate(stellar_features):
                if i % 2 == 0:
                    # Ensure all values have consistent types
                    min_val = float(feature['min'])
                    max_val = float(feature['max'])
                    default_val = float(feature['default'])
                    step_val = feature.get('step', (max_val - min_val) / 100)
                    
                    inputs[feature['name']] = st.number_input(
                        feature['label'],
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=float(step_val),
                        help=f"Range: {min_val} - {max_val}"
                    )
        
        with col2:
            for i, feature in enumerate(stellar_features):
                if i % 2 == 1:
                    # Ensure all values have consistent types
                    min_val = float(feature['min'])
                    max_val = float(feature['max'])
                    default_val = float(feature['default'])
                    step_val = feature.get('step', (max_val - min_val) / 100)
                    
                    inputs[feature['name']] = st.number_input(
                        feature['label'],
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=float(step_val),
                        help=f"Range: {min_val} - {max_val}"
                    )
    
    with tab3:
        st.subheader("Signal and Detection Characteristics")
        for feature in signal_features:
            # Ensure all values have consistent types
            min_val = float(feature['min'])
            max_val = float(feature['max'])
            default_val = float(feature['default'])
            step_val = feature.get('step', (max_val - min_val) / 100)
            
            inputs[feature['name']] = st.number_input(
                feature['label'],
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=float(step_val),
                help=f"Range: {min_val} - {max_val}"
            )
    
    return inputs

def display_prediction_result(result):
    """Display prediction results with styling."""
    st.header("🔮 Classification Results")
    
    prediction = result['prediction_label']
    confidence = result['confidence']
    probabilities = result['probabilities']
    
    # Determine styling based on prediction
    if 'Candidate' in prediction:
        class_style = "candidate"
        icon = "🟡"
        interpretation = "This object shows characteristics consistent with a planet candidate requiring further verification."
    elif 'Confirmed' in prediction:
        class_style = "confirmed" 
        icon = "🟢"
        interpretation = "This object has characteristics strongly indicating a confirmed exoplanet."
    else:
        class_style = "false-positive"
        icon = "🔴"
        interpretation = "This signal appears to be a false positive, likely caused by stellar activity or systematic noise."
    
    # Main prediction display
    st.markdown(f"""
    <div class="prediction-box {class_style}">
        <h2>{icon} {prediction}</h2>
        <h3>Confidence: {confidence:.1%}</h3>
        <p>{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed probability breakdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Probability Breakdown")
        
        # Create probability chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        colors = ['#dc3545', '#28a745', '#ffc107']  # Red, Green, Yellow
        
        bars = ax.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Probability')
        ax.set_title('Classification Probabilities')
        ax.set_ylim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Model Metrics")
        
        # Display individual model results if available
        individual_preds = result.get('individual_predictions', {})
        if len(individual_preds) > 1:
            st.write("**Model Agreement:**")
            for model_name, pred in individual_preds.items():
                st.write(f"- {model_name}: Class {pred}")
        
        st.write(f"**Features Used:** {result['features_used']}")
        st.write(f"**Models Used:** {', '.join(result['models_used'])}")
        
        # Confidence interpretation
        if confidence > 0.8:
            conf_text = "Very High 🎯"
            conf_color = "green"
        elif confidence > 0.6:
            conf_text = "High ✅"
            conf_color = "blue"
        elif confidence > 0.4:
            conf_text = "Moderate ⚠️"
            conf_color = "orange"
        else:
            conf_text = "Low ❓"
            conf_color = "red"
        
        st.markdown(f"**Confidence Level:** <span style='color: {conf_color}'>{conf_text}</span>", 
                   unsafe_allow_html=True)

def display_shap_explanations(result, inputs):
    """Display SHAP explanations for the prediction."""
    explanations = result.get('explanations', {})
    
    if not explanations:
        st.info("ℹ️ SHAP explanations not available for this prediction.")
        return
    
    st.header("🧠 Model Explanations")
    st.markdown("""
    **Why did the model make this prediction?** 
    The charts below show which features most influenced the decision using SHAP (SHapley Additive exPlanations) values.
    """)
    
    # Create tabs for different models if multiple explanations available
    if len(explanations) > 1:
        tabs = st.tabs([f"📊 {model.upper()} Explanation" for model in explanations.keys()])
        
        for i, (model_name, explanation) in enumerate(explanations.items()):
            with tabs[i]:
                display_single_shap_explanation(explanation, model_name, inputs)
    
    elif len(explanations) == 1:
        model_name, explanation = list(explanations.items())[0]
        display_single_shap_explanation(explanation, model_name, inputs)

def display_single_shap_explanation(explanation, model_name, inputs):
    """Display SHAP explanation for a single model."""
    if 'error' in explanation:
        st.error(f"❌ Could not generate explanation: {explanation['error']}")
        return
    
    # Create two columns for different visualizations
    col1, col2 = st.columns([1, 1])
    
    # Create a temporary explainer for visualization (we just need the plotting methods)
    try:
        from shap_explainer import ExoplanetExplainer
        temp_explainer = ExoplanetExplainer(None, [], {})
        temp_explainer.explainer_available = False  # We only use plotting methods
    except Exception as e:
        st.error(f"Could not initialize explainer for visualization: {e}")
        return
    
    with col1:
        st.subheader("🎯 Feature Contributions")
        
        # Create waterfall plot
        try:
            fig = temp_explainer.create_waterfall_plot(explanation, "")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not create waterfall plot: {e}")
    
    with col2:
        st.subheader("📈 Feature Importance")
        
        # Create importance plot
        try:
            fig = temp_explainer.create_feature_importance_plot(explanation)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not create importance plot: {e}")
    
    # Text explanation
    st.subheader("📝 Explanation Summary")
    try:
        summary = temp_explainer.get_explanation_summary(explanation, inputs)
        st.markdown(summary)
    except Exception as e:
        st.error(f"Could not generate text summary: {e}")
    
    # Feature details table
    with st.expander("📋 Detailed Feature Analysis", expanded=False):
        top_features = explanation.get('top_features', [])[:10]
        
        if top_features:
            # Create DataFrame for display
            df_data = []
            for feature in top_features:
                df_data.append({
                    'Feature': feature['feature'].replace('_', ' ').title(),
                    'Value': f"{feature['value']:.3f}",
                    'SHAP Value': f"{feature['shap_value']:.3f}",
                    'Impact': '🔴 Decreases' if feature['shap_value'] < 0 else '🟢 Increases',
                    'Magnitude': f"{feature['magnitude']:.3f}"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No feature details available.")

def main():
    """Main application function."""
    
    # Title and description
    st.markdown('<h1 class="main-header">🔭 Exoplanet Classification System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Discover new worlds!** This system uses machine learning models trained on NASA's Kepler, K2, and TESS mission data 
    to classify celestial objects as **confirmed exoplanets**, **planet candidates**, or **false positives**.
    
    Simply enter the observed characteristics below and let our AI analyze the data! 🚀
    """)
    
    # Load model and schema
    model, model_error = load_model()
    schema = load_schema()
    
    # Sidebar with status and information
    with st.sidebar:
        st.header("🔧 System Status")
        
        if model:
            st.success("✅ Models Loaded Successfully")
            model_info = model.get_model_info()
            st.write(f"**Available Models:** {', '.join(model_info['models_available'])}")
            st.write(f"**Features:** {model_info['feature_count']}")
            
            if model_info.get('model_metadata'):
                metadata = model_info['model_metadata']
                if metadata.get('target_f1'):
                    st.write(f"**Target F1 Score:** {metadata['target_f1']}")
        else:
            st.error(f"❌ Model Error: {model_error}")
            st.stop()
        
        st.header("🎯 Classification Classes")
        st.write("**🟢 Confirmed Planet:** Verified exoplanet")
        st.write("**🟡 Planet Candidate:** Awaiting confirmation") 
        st.write("**🔴 False Positive:** Non-planetary signal")
        
        st.header("📖 About the Models")
        st.markdown("""
        This system uses **ensemble machine learning** with:
        - **XGBoost**: Gradient boosting for tabular data
        - **LightGBM**: Fast gradient boosting 
        - **69 engineered features** from transit photometry
        - Trained on **21,000+ objects** from NASA missions
        """)
    
    # Main content area
    if schema and model:
        # Create input form
        inputs = create_input_form(schema)
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Classify Object", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing object characteristics..."):
                    try:
                        # Make prediction with explanations
                        result = model.predict_with_explanation(inputs, use_ensemble=True)
                        
                        # Display results
                        display_prediction_result(result)
                        
                        # Display SHAP explanations if available
                        display_shap_explanations(result, inputs)
                        
                        # Show feature summary
                        with st.expander("🔍 Feature Processing Details", expanded=False):
                            summary = model.feature_processor.get_feature_summary(inputs)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Input Summary:**")
                                st.write(f"- Features provided: {summary['input_features']}")
                                st.write(f"- Features mapped: {summary['mapped_features']}")
                                st.write(f"- Using defaults: {len(summary['missing_features'])}")
                            
                            with col2:
                                st.write("**Provided Features:**")
                                for feature in summary['provided_features']:
                                    st.write(f"- {feature}")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
        
        # Example presets
        st.header("🎲 Try Example Objects")
        st.markdown("Click on an example to load typical values:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌍 Earth-like Planet", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🪐 Hot Jupiter", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("⭐ False Positive", use_container_width=True):
                st.rerun()
    
    else:
        st.error("Could not load required components. Please check the configuration.")

if __name__ == "__main__":
    main()
