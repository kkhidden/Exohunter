#!/usr/bin/env python3
"""
FastAPI Backend for Exoplanet Classification System

This provides REST API endpoints for the exoplanet classification service,
allowing integration with other applications and services.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add app directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "app"))

try:
    from model_inference import ExoplanetModelInference
    from feature_processor import ExoplanetFeatureProcessor
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    ExoplanetModelInference = None
    ExoplanetFeatureProcessor = None

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Classification API",
    description="ML-powered API for classifying celestial objects as exoplanets, candidates, or false positives",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_instance = None

# Pydantic models for request/response
class ExoplanetInput(BaseModel):
    """Input parameters for exoplanet classification."""
    orbital_period: Optional[float] = Field(None, ge=0.01, le=10000, description="Orbital period in days")
    transit_duration: Optional[float] = Field(None, ge=0.01, le=100, description="Transit duration in hours")
    planet_radius: Optional[float] = Field(None, ge=0.1, le=50, description="Planet radius in Earth radii")
    snr: Optional[float] = Field(None, ge=0.0, le=10000, description="Signal-to-noise ratio")
    impact_parameter: Optional[float] = Field(None, ge=0.0, le=2.0, description="Impact parameter")
    stellar_teff: Optional[float] = Field(None, ge=2000, le=15000, description="Stellar effective temperature in K")
    stellar_logg: Optional[float] = Field(None, ge=0.0, le=6.0, description="Stellar surface gravity (log g)")
    stellar_radius: Optional[float] = Field(None, ge=0.1, le=200, description="Stellar radius in solar radii")
    stellar_mass: Optional[float] = Field(None, ge=0.1, le=200, description="Stellar mass in solar masses")
    transit_depth: Optional[float] = Field(None, ge=0.0, le=100000, description="Transit depth in ppm")
    
    class Config:
        schema_extra = {
            "example": {
                "orbital_period": 15.2,
                "transit_duration": 4.1,
                "planet_radius": 1.3,
                "snr": 12.5,
                "impact_parameter": 0.3,
                "stellar_teff": 5800,
                "stellar_logg": 4.2,
                "stellar_radius": 1.1,
                "stellar_mass": 1.05,
                "transit_depth": 150
            }
        }

class PredictionResponse(BaseModel):
    """Response from exoplanet classification."""
    prediction: int = Field(description="Predicted class index")
    prediction_label: str = Field(description="Human-readable prediction label")
    confidence: float = Field(description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(description="Probability for each class")
    features_used: int = Field(description="Number of features used in prediction")
    models_used: List[str] = Field(description="List of models used for prediction")

class ModelInfo(BaseModel):
    """Information about loaded models."""
    models_available: List[str] = Field(description="Available model types")
    feature_count: int = Field(description="Number of features used by models")
    class_names: Dict = Field(description="Mapping of class indices to names")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    models_loaded: bool = Field(description="Whether models are loaded")
    version: str = Field(description="API version")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model_instance
    
    if ExoplanetModelInference:
        try:
            model_instance = ExoplanetModelInference()
            print("✅ Models loaded successfully on startup")
        except Exception as e:
            print(f"❌ Failed to load models on startup: {e}")
            model_instance = None
    else:
        print("⚠️ Model classes not available - running in mock mode")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "service": "Exoplanet Classification API",
        "version": "1.0.0",
        "description": "ML-powered exoplanet detection and classification",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_instance else "degraded",
        models_loaded=model_instance is not None,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models."""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        info = model_instance.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(input_data: ExoplanetInput):
    """
    Classify a celestial object as an exoplanet, candidate, or false positive.
    
    Args:
        input_data: Orbital and stellar parameters for the object
        
    Returns:
        Classification prediction with confidence scores
    """
    if not model_instance:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert Pydantic model to dictionary, excluding None values
        input_dict = {k: v for k, v in input_data.dict().items() if v is not None}
        
        if not input_dict:
            raise HTTPException(status_code=400, detail="At least one parameter must be provided")
        
        # Make prediction
        result = model_instance.predict(input_dict, use_ensemble=True)
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            features_used=result['features_used'],
            models_used=result['models_used']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/detailed", response_model=Dict)
async def predict_detailed(input_data: ExoplanetInput):
    """
    Get detailed prediction with explanations and feature analysis.
    
    Args:
        input_data: Orbital and stellar parameters for the object
        
    Returns:
        Detailed prediction results including SHAP explanations
    """
    if not model_instance:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert Pydantic model to dictionary, excluding None values
        input_dict = {k: v for k, v in input_data.dict().items() if v is not None}
        
        if not input_dict:
            raise HTTPException(status_code=400, detail="At least one parameter must be provided")
        
        # Make prediction with explanations
        result = model_instance.predict_with_explanation(input_dict, use_ensemble=True)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed prediction failed: {str(e)}")

@app.get("/model/features", response_model=List[str])
async def get_model_features():
    """Get list of features used by the models."""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        info = model_instance.get_model_info()
        return info.get('feature_names', [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")

# Additional utility endpoints

@app.get("/examples", response_model=Dict[str, ExoplanetInput])
async def get_example_inputs():
    """Get example input data for different types of objects."""
    examples = {
        "hot_jupiter": ExoplanetInput(
            orbital_period=3.2,
            transit_duration=2.8,
            planet_radius=11.2,
            snr=45.0,
            impact_parameter=0.2,
            stellar_teff=5200,
            stellar_logg=4.3,
            stellar_radius=1.1,
            stellar_mass=1.0,
            transit_depth=8500
        ),
        "super_earth": ExoplanetInput(
            orbital_period=28.5,
            transit_duration=3.1,
            planet_radius=1.8,
            snr=15.2,
            impact_parameter=0.4,
            stellar_teff=4800,
            stellar_logg=4.6,
            stellar_radius=0.8,
            stellar_mass=0.75,
            transit_depth=420
        ),
        "earth_like": ExoplanetInput(
            orbital_period=365.0,
            transit_duration=13.2,
            planet_radius=1.0,
            snr=8.5,
            impact_parameter=0.1,
            stellar_teff=5778,
            stellar_logg=4.4,
            stellar_radius=1.0,
            stellar_mass=1.0,
            transit_depth=84
        )
    }
    
    return examples

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Exoplanet Classification API...")
    print("Available endpoints:")
    print("  - GET  /              : API information")
    print("  - GET  /health        : Health check") 
    print("  - GET  /model/info    : Model information")
    print("  - POST /predict       : Basic prediction")
    print("  - POST /predict/detailed : Detailed prediction with explanations")
    print("  - GET  /model/features: List model features")
    print("  - GET  /examples      : Example inputs")
    print("\nStarting server on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)