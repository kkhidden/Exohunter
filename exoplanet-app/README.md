# 🔭 Exoplanet Classification Application

A comprehensive machine learning application for classifying celestial objects as **confirmed exoplanets**, **planet candidates**, or **false positives** using NASA mission data.

## 🌟 Features

- **Machine Learning Models**: XGBoost and LightGBM ensemble trained on NASA Kepler/K2/TESS data
- **Interactive Web Interface**: Streamlit app with intuitive parameter input forms
- **Model Explainability**: SHAP explanations showing feature contributions
- **Data Visualizations**: Confidence plots, feature importance, parameter distributions
- **REST API**: FastAPI backend for programmatic access
- **69 Engineered Features**: Advanced preprocessing pipeline with missing value handling
- **Real-time Predictions**: Instant classification with confidence scores

## 🚀 Quick Start

### Prerequisites

- Python 3.9 (configured to use conda environment)
- Conda package manager
- Pre-trained models (included in `model/models/` directory)

### Installation

1. **Activate the Python 3.9 environment:**
   ```bash
   conda activate exoplanet
   ```

2. **Navigate to project directory:**
   ```bash
   cd /Users/kkgogada/Code/exoplanet-app
   ```

3. **Verify installation:**
   ```bash
   python -c "import sys; print(f'Python version: {sys.version}')"
   ```

### Running the Application

#### 🖥️ Streamlit Web Interface
```bash
conda activate exoplanet
python -m streamlit run app/streamlit_app.py --server.port 8503
```
Then open http://localhost:8503 in your browser.

#### 🔌 API Server
```bash
conda activate exoplanet  
python backend/api.py
```
API available at http://localhost:8000 with interactive docs at http://localhost:8000/docs

#### 🧪 Test Client
```bash
conda activate exoplanet
python backend/test_client.py
```

## 📁 Project Structure

```
exoplanet-app/
├── app/                          # Main application code
│   ├── streamlit_app.py         # Streamlit web interface
│   ├── model_inference.py       # Model loading and prediction
│   ├── feature_processor.py     # Feature engineering pipeline
│   ├── shap_explainer.py       # SHAP explanations
│   └── visualizer.py           # Data visualizations
├── backend/                     # API backend
│   ├── api.py                   # FastAPI server
│   └── test_client.py          # API client example
├── model/                       # ML models and training code
│   ├── models/                  # Trained model files
│   ├── dataset/                 # Data processing utilities
│   ├── gradient_boosted_tree.ipynb
│   └── cnn_lightcurve.ipynb
├── artifacts/                   # Configuration files
│   └── schema.json             # Feature schema and defaults
├── tests/                       # Test suite
│   └── test_application.py     # Comprehensive tests
└── requirements.txt            # Python dependencies
```

## 🔧 Configuration

The application is configured to use:
- **Python 3.9.23** via conda environment "exoplanet"
- **XGBoost** and **LightGBM** models trained on NASA data
- **69 features** including orbital, stellar, and signal characteristics
- **SHAP explanations** for model interpretability

## 🎯 Usage Examples

### Web Interface
1. Open http://localhost:8503
2. Enter exoplanet parameters in the form tabs:
   - **Orbital & Transit**: period, duration, radius, etc.
   - **Stellar Properties**: temperature, surface gravity, mass
   - **Signal Characteristics**: SNR, impact parameter
3. Click "🚀 Classify Object" to get predictions
4. View results with confidence scores and SHAP explanations

### API Usage
```python
import requests

# Basic prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "orbital_period": 15.2,
        "planet_radius": 1.3,
        "stellar_teff": 5800,
        "snr": 12.5
    }
)

result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Python Code
```python
from app.model_inference import ExoplanetModelInference

# Load models
model = ExoplanetModelInference()

# Make prediction
sample_data = {
    "orbital_period": 10.5,
    "planet_radius": 1.1,
    "stellar_teff": 5500,
    "snr": 15.0
}

result = model.predict_with_explanation(sample_data)
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.3f}")

# Access SHAP explanations
explanations = result.get('explanations', {})
for model_name, explanation in explanations.items():
    print(f"{model_name} top features:")
    for feature in explanation['top_features'][:3]:
        print(f"  - {feature['feature']}: {feature['shap_value']:.3f}")
```

## 📊 Model Information

- **Training Data**: 21,271 objects from NASA Kepler, K2, and TESS missions
- **Features**: 69 engineered features including:
  - Orbital characteristics (period, duration, depth)
  - Stellar properties (temperature, gravity, mass)
  - Signal quality metrics (SNR, transit count)
  - Derived ratios and transformations
- **Models**: Ensemble of XGBoost and LightGBM with hyperparameter optimization
- **Performance**: Optimized for F1-score with class balancing
- **Classes**:
  - **Confirmed Planet**: Verified exoplanet
  - **Planet Candidate**: Awaiting confirmation
  - **False Positive**: Non-planetary signal

## 🧪 Testing

Run the comprehensive test suite:
```bash
conda activate exoplanet
python tests/test_application.py
```

Tests cover:
- ✅ File structure validation
- ✅ Model loading and inference  
- ✅ Feature preprocessing
- ✅ Multiple prediction scenarios
- ✅ Edge case handling
- ✅ API endpoint functionality

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Basic prediction |
| `/predict/detailed` | POST | Prediction with SHAP explanations |
| `/model/info` | GET | Model metadata |
| `/model/features` | GET | Feature list |
| `/examples` | GET | Example input data |

## 🎨 Visualizations

The application includes various visualizations:
- **Prediction Confidence**: Bar and pie charts of class probabilities
- **SHAP Explanations**: Waterfall plots showing feature contributions
- **Feature Importance**: Comparison across models
- **Parameter Context**: Where inputs fall in typical ranges
- **Exoplanet Gallery**: Educational plots of different planet types

## 🔍 SHAP Explanations

Every prediction includes SHAP (SHapley Additive exPlanations) values that show:
- Which features most influenced the prediction
- Positive vs negative contributions
- Feature importance rankings
- Text summaries of key factors

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure conda environment is activated:
   ```bash
   conda activate exoplanet
   ```

2. **Model loading fails**: Check that model files exist in `model/models/`:
   ```bash
   ls model/models/*.pkl
   ```

3. **Port conflicts**: Change port numbers in startup commands if needed

4. **Python version**: Verify using Python 3.9:
   ```bash
   python --version  # Should show 3.9.x
   ```

### Environment Setup

If the conda environment needs to be recreated:
```bash
conda create -n exoplanet python=3.9
conda activate exoplanet
pip install pandas numpy scikit-learn streamlit shap matplotlib xgboost lightgbm fastapi uvicorn pydantic requests
```

## 🤝 Contributing

The application is fully functional with:
- ✅ Model inference system
- ✅ Feature preprocessing pipeline  
- ✅ Interactive web interface
- ✅ SHAP explainability
- ✅ Data visualizations
- ✅ REST API backend
- ✅ Comprehensive testing

## 📄 License

This project uses machine learning models trained on public NASA mission data for educational and research purposes.

## 🔗 Resources

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **Kepler Mission**: https://www.nasa.gov/kepler
- **TESS Mission**: https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite
- **SHAP Documentation**: https://shap.readthedocs.io/

---

**🎉 Ready for exoplanet discovery!** The application is fully configured with Python 3.9 and ready to classify celestial objects using state-of-the-art machine learning models.