#!/usr/bin/env python3
"""
Simple client example for testing the Exoplanet Classification API.
"""

import requests
import json
from typing import Dict, Any

class ExoplanetAPIClient:
    """Client for interacting with the Exoplanet Classification API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a basic prediction.
        
        Args:
            input_data: Dictionary with orbital and stellar parameters
            
        Returns:
            Prediction results
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_detailed(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a detailed prediction with explanations.
        
        Args:
            input_data: Dictionary with orbital and stellar parameters
            
        Returns:
            Detailed prediction results with SHAP explanations
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict/detailed",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_examples(self) -> Dict[str, Any]:
        """Get example input data."""
        try:
            response = requests.get(f"{self.base_url}/examples")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def main():
    """Test the API client with sample data."""
    print("🔭 Testing Exoplanet Classification API Client")
    print("=" * 50)
    
    # Initialize client
    client = ExoplanetAPIClient()
    
    # Health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Models loaded: {health.get('models_loaded', False)}")
    
    # Model info
    print("\n2. Model Information:")
    model_info = client.get_model_info()
    if 'error' not in model_info:
        print(f"   Available models: {model_info.get('models_available', [])}")
        print(f"   Feature count: {model_info.get('feature_count', 0)}")
    else:
        print(f"   Error: {model_info['error']}")
    
    # Test prediction with sample data
    print("\n3. Sample Prediction:")
    sample_data = {
        "orbital_period": 10.5,
        "transit_duration": 3.2,
        "planet_radius": 1.1,
        "snr": 15.0,
        "stellar_teff": 5500,
        "stellar_logg": 4.4,
        "stellar_radius": 1.0
    }
    
    print(f"   Input: {sample_data}")
    result = client.predict(sample_data)
    
    if 'error' not in result:
        print(f"   Prediction: {result.get('prediction_label')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Probabilities: {result.get('probabilities', {})}")
    else:
        print(f"   Error: {result['error']}")
    
    # Test with example data
    print("\n4. Example Data Test:")
    examples = client.get_examples()
    
    if 'error' not in examples and examples:
        example_name = list(examples.keys())[0]
        example_data = examples[example_name]
        
        print(f"   Testing with {example_name}: {example_data}")
        
        result = client.predict(example_data)
        if 'error' not in result:
            print(f"   Result: {result.get('prediction_label')} ({result.get('confidence', 0):.3f})")
        else:
            print(f"   Error: {result['error']}")
    else:
        print("   Could not get example data")
    
    print("\n" + "=" * 50)
    print("✅ API Client test completed!")

if __name__ == "__main__":
    main()