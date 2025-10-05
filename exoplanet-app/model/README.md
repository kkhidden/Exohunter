# NASA Exoplanet Classification

## 🎯 Goal
Develop machine learning models to classify exoplanet candidates into:
- **CONFIRMED** - Verified exoplanets
- **CANDIDATE** - Awaiting confirmation  
- **FALSE POSITIVE** - Non-planetary signals

**Target**: Achieve 80%+ F1 score

## 📊 Data
- **KOI.csv** - Kepler Objects of Interest (9,564 candidates)
- **K2OI.csv** - K2 Mission Objects (4,004 candidates)  
- **TESSOI.csv** - TESS Objects of Interest (7,703 candidates)
- **Total**: 21,271 candidates across 17,182 stars

## 🤖 Models

### 1. Gradient Boosted Trees (`gradient_boosting_enhanced.py`)
- XGBoost & LightGBM with hyperparameter optimization
- Advanced feature engineering for tabular data
- Handles class imbalance and prevents overfitting

### 2. CNN Light Curves (`cnn_lightcurve_enhanced.py`)  
- Two-view 1D CNN architecture
- Synthetic light curve generation from transit parameters
- Deep learning on time-series patterns

## 🚀 Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run models
python gradient_boosting_enhanced.py
python cnn_lightcurve_enhanced.py
```

## 📁 Structure
```
├── gradient_boosting_enhanced.py  # GBT model pipeline
├── cnn_lightcurve_enhanced.py     # CNN model pipeline  
├── dataset/loader.py              # Data loading utilities
├── *.csv                          # NASA datasets
└── requirements.txt               # Dependencies
```