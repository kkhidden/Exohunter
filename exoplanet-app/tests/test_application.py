#!/usr/bin/env python3
"""
Comprehensive Test Suite for Exoplanet Classification Application
Tests all components: model inference, feature processing, API, and Streamlit app.
"""

import sys
import os
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

class ExoplanetAppTester:
    """Comprehensive tester for the exoplanet classification application."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.sample_data = {
            "hot_jupiter": {
                "orbital_period": 3.2,
                "transit_duration": 2.8,
                "planet_radius": 11.2,
                "snr": 45.0,
                "stellar_teff": 5200,
                "stellar_logg": 4.3,
                "stellar_radius": 1.1
            },
            "super_earth": {
                "orbital_period": 28.5,
                "transit_duration": 3.1,
                "planet_radius": 1.8,
                "snr": 15.2,
                "stellar_teff": 4800,
                "stellar_logg": 4.6,
                "stellar_radius": 0.8
            },
            "earth_like": {
                "orbital_period": 365.0,
                "transit_duration": 13.2,
                "planet_radius": 1.0,
                "snr": 8.5,
                "stellar_teff": 5778,
                "stellar_logg": 4.4,
                "stellar_radius": 1.0
            }
        }
    
    def print_header(self, title: str):
        """Print a formatted test header."""
        print("\n" + "=" * 60)
        print(f"🔬 {title}")
        print("=" * 60)
    
    def print_test(self, test_name: str, status: str = "RUNNING"):
        """Print test status."""
        status_icons = {
            "RUNNING": "🔄",
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
            "WARN": "⚠️"
        }
        icon = status_icons.get(status, "❓")
        print(f"{icon} {test_name}")
    
    def test_file_structure(self) -> bool:
        """Test that all required files and directories exist."""
        self.print_header("Testing File Structure")
        
        required_paths = [
            "app/streamlit_app.py",
            "app/model_inference.py", 
            "app/feature_processor.py",
            "app/shap_explainer.py",
            "app/visualizer.py",
            "backend/api.py",
            "backend/test_client.py",
            "artifacts/schema.json",
            "model/models/",
            "requirements.txt"
        ]
        
        all_exist = True
        
        for path in required_paths:
            full_path = Path(__file__).parent.parent / path
            if full_path.exists():
                self.print_test(f"Found {path}", "PASS")
            else:
                self.print_test(f"Missing {path}", "FAIL")
                all_exist = False
        
        # Check for model files
        models_dir = Path(__file__).parent.parent / "model" / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            if model_files:
                self.print_test(f"Found {len(model_files)} model files", "PASS")
            else:
                self.print_test("No model files found", "FAIL")
                all_exist = False
        
        self.test_results["file_structure"] = all_exist
        return all_exist
    
    def test_model_loading(self) -> bool:
        """Test model loading and inference."""
        self.print_header("Testing Model Loading & Inference")
        
        try:
            from model_inference import ExoplanetModelInference
            
            # Test model initialization
            self.print_test("Initializing model inference", "RUNNING")
            model = ExoplanetModelInference()
            self.print_test("Model inference initialized", "PASS")
            
            # Test model info
            self.print_test("Getting model info", "RUNNING")
            info = model.get_model_info()
            
            if info['models_available']:
                self.print_test(f"Models available: {info['models_available']}", "PASS")
            else:
                self.print_test("No models available", "FAIL")
                return False
            
            # Test prediction
            self.print_test("Testing prediction", "RUNNING")
            sample = self.sample_data["earth_like"]
            result = model.predict(sample)
            
            required_keys = ['prediction', 'prediction_label', 'confidence', 'probabilities']
            if all(key in result for key in required_keys):
                self.print_test(f"Prediction successful: {result['prediction_label']} ({result['confidence']:.3f})", "PASS")
            else:
                self.print_test("Prediction missing required fields", "FAIL")
                return False
            
            # Test SHAP explanations
            self.print_test("Testing SHAP explanations", "RUNNING")
            explained_result = model.predict_with_explanation(sample)
            
            if 'explanations' in explained_result:
                explanations = explained_result['explanations']
                if explanations:
                    self.print_test(f"SHAP explanations available for: {list(explanations.keys())}", "PASS")
                else:
                    self.print_test("SHAP explanations empty", "WARN")
            else:
                self.print_test("No SHAP explanations in result", "WARN")
            
            self.test_results["model_loading"] = True
            return True
            
        except Exception as e:
            self.print_test(f"Model loading failed: {str(e)}", "FAIL")
            self.test_results["model_loading"] = False
            return False
    
    def test_feature_processing(self) -> bool:
        """Test feature preprocessing."""
        self.print_header("Testing Feature Processing")
        
        try:
            from feature_processor import ExoplanetFeatureProcessor
            
            # Initialize processor
            self.print_test("Initializing feature processor", "RUNNING")
            processor = ExoplanetFeatureProcessor()
            self.print_test("Feature processor initialized", "PASS")
            
            # Test feature vector creation
            self.print_test("Testing feature vector creation", "RUNNING")
            sample = self.sample_data["super_earth"]
            
            feature_vector = processor.create_feature_vector(sample)
            
            if feature_vector.shape == (69,):
                self.print_test(f"Feature vector created: shape {feature_vector.shape}", "PASS")
            else:
                self.print_test(f"Wrong feature vector shape: {feature_vector.shape}", "FAIL")
                return False
            
            # Test all values are finite
            import numpy as np
            if np.isfinite(feature_vector).all():
                self.print_test("All feature values are finite", "PASS")
            else:
                self.print_test("Feature vector contains invalid values", "FAIL")
                return False
            
            # Test feature summary
            self.print_test("Testing feature summary", "RUNNING")
            summary = processor.get_feature_summary(sample)
            
            if all(key in summary for key in ['input_features', 'mapped_features', 'missing_features']):
                self.print_test(f"Feature summary: {summary['mapped_features']}/{summary['input_features']} mapped", "PASS")
            else:
                self.print_test("Feature summary missing fields", "FAIL")
                return False
            
            self.test_results["feature_processing"] = True
            return True
            
        except Exception as e:
            self.print_test(f"Feature processing failed: {str(e)}", "FAIL")
            self.test_results["feature_processing"] = False
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test FastAPI backend endpoints (requires running server)."""
        self.print_header("Testing API Endpoints")
        
        api_url = "http://localhost:8000"
        
        try:
            # Test health endpoint
            self.print_test("Testing health endpoint", "RUNNING")
            response = requests.get(f"{api_url}/health", timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('models_loaded'):
                    self.print_test("Health check passed - models loaded", "PASS")
                else:
                    self.print_test("Health check passed - models not loaded", "WARN")
            else:
                self.print_test(f"Health check failed: {response.status_code}", "FAIL")
                return False
            
            # Test prediction endpoint
            self.print_test("Testing prediction endpoint", "RUNNING")
            sample = self.sample_data["hot_jupiter"]
            
            response = requests.post(
                f"{api_url}/predict",
                json=sample,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                pred_data = response.json()
                self.print_test(f"Prediction API: {pred_data.get('prediction_label')} ({pred_data.get('confidence', 0):.3f})", "PASS")
            else:
                self.print_test(f"Prediction API failed: {response.status_code}", "FAIL")
                return False
            
            # Test model info endpoint
            self.print_test("Testing model info endpoint", "RUNNING")
            response = requests.get(f"{api_url}/model/info", timeout=5)
            
            if response.status_code == 200:
                model_data = response.json()
                self.print_test(f"Model info: {model_data.get('models_available', [])}", "PASS")
            else:
                self.print_test(f"Model info failed: {response.status_code}", "FAIL")
                return False
            
            self.test_results["api_endpoints"] = True
            return True
            
        except requests.exceptions.RequestException as e:
            self.print_test(f"API connection failed: {str(e)}", "SKIP")
            self.print_test("(Start API server with: python backend/api.py)", "SKIP")
            self.test_results["api_endpoints"] = None
            return None
    
    def test_multiple_predictions(self) -> bool:
        """Test predictions on multiple sample objects."""
        self.print_header("Testing Multiple Predictions")
        
        try:
            from model_inference import ExoplanetModelInference
            model = ExoplanetModelInference()
            
            all_passed = True
            
            for obj_type, sample_data in self.sample_data.items():
                self.print_test(f"Testing {obj_type} classification", "RUNNING")
                
                result = model.predict(sample_data)
                
                prediction = result['prediction_label']
                confidence = result['confidence']
                
                # Basic validation
                if prediction in ['False Positive', 'Confirmed Planet', 'Planet Candidate']:
                    self.print_test(f"{obj_type}: {prediction} (conf: {confidence:.3f})", "PASS")
                else:
                    self.print_test(f"{obj_type}: Invalid prediction {prediction}", "FAIL")
                    all_passed = False
                
                if 0 <= confidence <= 1:
                    pass  # Confidence is valid
                else:
                    self.print_test(f"{obj_type}: Invalid confidence {confidence}", "FAIL")
                    all_passed = False
            
            self.test_results["multiple_predictions"] = all_passed
            return all_passed
            
        except Exception as e:
            self.print_test(f"Multiple predictions test failed: {str(e)}", "FAIL")
            self.test_results["multiple_predictions"] = False
            return False
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        self.print_header("Testing Edge Cases")
        
        try:
            from model_inference import ExoplanetModelInference
            model = ExoplanetModelInference()
            
            # Test empty input
            self.print_test("Testing empty input", "RUNNING")
            try:
                result = model.predict({})
                self.print_test("Empty input handled gracefully", "PASS")
            except Exception as e:
                self.print_test(f"Empty input error: {str(e)}", "FAIL")
                return False
            
            # Test single parameter
            self.print_test("Testing single parameter", "RUNNING")
            try:
                result = model.predict({"orbital_period": 10.0})
                self.print_test("Single parameter handled", "PASS")
            except Exception as e:
                self.print_test(f"Single parameter error: {str(e)}", "FAIL")
                return False
            
            # Test extreme values
            self.print_test("Testing extreme values", "RUNNING")
            try:
                extreme_data = {
                    "orbital_period": 0.01,  # Very short period
                    "planet_radius": 50.0,   # Very large planet
                    "stellar_teff": 10000,   # Very hot star
                    "snr": 1000.0           # Very high SNR
                }
                result = model.predict(extreme_data)
                self.print_test("Extreme values handled", "PASS")
            except Exception as e:
                self.print_test(f"Extreme values error: {str(e)}", "FAIL")
                return False
            
            self.test_results["edge_cases"] = True
            return True
            
        except Exception as e:
            self.print_test(f"Edge cases test failed: {str(e)}", "FAIL")
            self.test_results["edge_cases"] = False
            return False
    
    def generate_report(self):
        """Generate final test report."""
        self.print_header("Test Report Summary")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        failed_tests = sum(1 for result in self.test_results.values() if result is False)
        skipped_tests = sum(1 for result in self.test_results.values() if result is None)
        
        print(f"\n📊 Test Results:")
        print(f"   Total tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   ⏭️ Skipped: {skipped_tests}")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_map = {True: "✅ PASS", False: "❌ FAIL", None: "⏭️ SKIP"}
            status = status_map.get(result, "❓ UNKNOWN")
            print(f"   {test_name}: {status}")
        
        success_rate = (passed_tests / (total_tests - skipped_tests)) * 100 if (total_tests - skipped_tests) > 0 else 0
        
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Application is ready for deployment!")
        elif success_rate >= 60:
            print("⚠️ Application has some issues but is functional.")
        else:
            print("❌ Application needs significant fixes before deployment.")
        
        return success_rate
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting Comprehensive Exoplanet Application Tests")
        
        # Run all test categories
        self.test_file_structure()
        self.test_feature_processing()
        self.test_model_loading()
        self.test_multiple_predictions()
        self.test_edge_cases()
        self.test_api_endpoints()  # May skip if server not running
        
        # Generate final report
        success_rate = self.generate_report()
        
        return success_rate

def main():
    """Main test runner."""
    tester = ExoplanetAppTester()
    success_rate = tester.run_all_tests()
    
    # Exit with appropriate code
    if success_rate >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()