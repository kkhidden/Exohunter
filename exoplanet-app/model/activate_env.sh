#!/bin/bash
# Activation script for NASASAC2025 environment
echo "Activating NASASAC2025 Environment..."
conda activate exoplanet
echo "Environment activated!"
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "XGBoost version: $(python -c 'import xgboost; print(xgboost.__version__)')"
echo ""
echo "Ready to run models! Use:"
echo "  python gradient_boosting_enhanced.py"
echo "  python cnn_lightcurve_enhanced.py"
echo ""