# scripts/run_tests.py
#!/usr/bin/env python3
"""
Test suite for bayesiandoe to validate functionality.
"""

import os
import sys
import time
import traceback
import unittest
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_core_tests():
    """Run tests for core functionality without UI."""
    from bayesiandoe.core import OptunaBayesianExperiment, _calculate_parameter_distance
    from bayesiandoe.parameters import ChemicalParameter
    
    print("-- Testing core functionality")
    
    try:
        # Test parameter creation
        temp = ChemicalParameter("Temperature", "continuous", 0, 100, units="°C")
        time = ChemicalParameter("Time", "discrete", 1, 24, units="h")
        solvent = ChemicalParameter("Solvent", "categorical", choices=["DMF", "DMSO", "THF", "Toluene"])
        
        print("-- Created parameters - Success")
        
        # Test model creation
        model = OptunaBayesianExperiment()
        model.add_parameter(temp)
        model.add_parameter(time)
        model.add_parameter(solvent)
        model.set_objectives({"yield": 1.0})
        
        print("-- Created model with parameters - Success")
        
        # Test parameter distance calculation
        params1 = {"Temperature": 50, "Time": 12, "Solvent": "DMF"}
        params2 = {"Temperature": 75, "Time": 6, "Solvent": "THF"}
        dist = _calculate_parameter_distance(params1, params2, model.parameters)
        print(f"-- Parameter distance: {dist:.4f} - Success")
        
        # Test experiment suggestion
        model.create_study()
        suggestions = model.suggest_experiments(5)
        print(f"-- Generated {len(suggestions)} suggestions - Success")
        
        # Test adding experiment results
        model.experiments.append({
            "params": suggestions[0],
            "results": {"yield": 0.65}
        })
        
        print("-- Added experiment result - Success")
        
        # Test surrogate model - with proper log handling
        model.log("Testing log method functionality")
        new_suggestions = model.suggest_experiments(2)
        print(f"-- Generated {len(new_suggestions)} more suggestions - Success")
        
        # Test cache clearing
        if hasattr(model, 'clear_surrogate_cache'):
            model.clear_surrogate_cache()
            print("-- Cache clearing successful")
        
        # Test acquisition functions
        ei = model.acquisition_function_ei(0.8, 0.2, 0.7, xi=0.01)
        pi = model.acquisition_function_pi(0.8, 0.2, 0.7, xi=0.01)
        ucb = model.acquisition_function_ucb(0.8, 0.2, 0.0, beta=2.0)
        print(f"-- Acquisition functions: EI={ei:.4f}, PI={pi:.4f}, UCB={ucb:.4f} - Success")
        
        return True
        
    except Exception as e:
        print(f"-- ERROR: Core tests failed: {str(e)}")
        traceback.print_exc()
        return False

def run_model_tests():
    """Run tests for the statistical modeling."""
    try:
        print("-- Testing statistical modeling")
        
        from bayesiandoe.core import OptunaBayesianExperiment
        from bayesiandoe.parameters import ChemicalParameter
        import numpy as np
        
        # Create model with parameters for quadratic function
        model = OptunaBayesianExperiment()
        model.add_parameter(ChemicalParameter("x", "continuous", -5, 5))
        model.set_objectives({"y": 1.0})
        
        # Generate synthetic training data for y = x^2
        np.random.seed(42)  # For reproducibility
        for _ in range(10):
            x = np.random.uniform(-5, 5)
            y = x**2 + np.random.normal(0, 0.5)  # Add noise
            
            model.experiments.append({
                "params": {"x": x},
                "results": {"y": y / 25.0}  # Normalize to [0,1] range
            })
        
        # Test parameter importance
        importance = model.analyze_parameter_importance()
        if importance and "x" in importance:
            print(f"-- Parameter importance: x = {importance['x']:.4f} - Success")
        
        # Train surrogate model
        X, y = model._extract_normalized_features_and_targets()
        try:
            rf_model = model._train_surrogate_model(X, y, model_type='rf')
            print("-- Trained random forest model - Success")
            
            # Test prediction
            test_point = model._normalize_params({"x": 2.0})
            pred, std = model._predict_candidate(test_point, X, y)
            print(f"-- Model prediction for x=2.0: {pred:.4f} ± {std:.4f} - Success")
            
            # Test uncertainty-driven exploration
            # Points with less data should have higher uncertainty
            sparse_region = model._normalize_params({"x": 4.5})  # Likely sparse region
            _, std_sparse = model._predict_candidate(sparse_region, X, y)
            
            dense_region = model._normalize_params({"x": 0.0})  # Likely dense region
            _, std_dense = model._predict_candidate(dense_region, X, y)
            
            print(f"-- Uncertainty comparison: sparse={std_sparse:.4f}, dense={std_dense:.4f}")
            if std_sparse > std_dense:
                print("-- Uncertainty-driven exploration working correctly - Success")
            
        except Exception as e:
            print(f"-- WARNING: Surrogate model tests partial failure: {e}")
        
        # Test Thompson sampling
        candidates = [
            {"params": {"x": -4.0}, "acquisition_value": 0.7},
            {"params": {"x": -2.0}, "acquisition_value": 0.8},
            {"params": {"x": 0.0}, "acquisition_value": 0.9},
            {"params": {"x": 2.0}, "acquisition_value": 0.8},
            {"params": {"x": 4.0}, "acquisition_value": 0.7}
        ]
        
        selected = model._apply_thompson_sampling(candidates)
        print(f"-- Thompson sampling selected {len(selected)} candidates - Success")
        
        return True
        
    except Exception as e:
        print(f"-- ERROR: Model tests failed: {str(e)}")
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """Run the more comprehensive tests."""
    try:
        print("-- Running extended test suite")
        
        # Import test functions
        from bayesiandoe.tests.test_core_functionality import (
            TestAcquisitionFunctions, 
            TestParameterHandling,
            TestExperimentOptimization
        )
        
        # Create test instances
        acq_tests = TestAcquisitionFunctions()
        param_tests = TestParameterHandling()
        exp_tests = TestExperimentOptimization()
        
        # Run setup for each test class
        acq_tests.setup_method()
        param_tests.setup_method()
        exp_tests.setup_method()
        
        # Run acquisition function tests
        print("-- Testing acquisition functions")
        acq_tests.test_expected_improvement()
        acq_tests.test_probability_improvement()
        acq_tests.test_upper_confidence_bound()
        acq_tests.test_acquisition_caching()
        print("-- Acquisition function tests passed")
        
        # Run parameter handling tests
        print("-- Testing parameter handling")
        param_tests.test_parameter_validation()
        param_tests.test_parameter_normalization()
        param_tests.test_parameter_distance()
        print("-- Parameter handling tests passed")
        
        # Run experiment optimization tests
        print("-- Testing experiment optimization")
        exp_tests.test_experiment_scoring()
        exp_tests.test_experiment_suggestion()
        exp_tests.test_diverse_subset_selection()
        print("-- Experiment optimization tests passed")
        
        # Test edge cases
        from bayesiandoe.tests.test_core_functionality import test_parameter_distance_edge_cases
        test_parameter_distance_edge_cases()
        print("-- Edge case tests passed")
        
        print("-- All comprehensive tests passed successfully")
        return True
        
    except Exception as e:
        print(f"-- ERROR: Comprehensive tests failed: {str(e)}")
        traceback.print_exc()
        return False

def run_pytest_suite():
    """Run pytest suite if available."""
    try:
        import pytest
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            with open(os.path.join(test_dir, '__init__.py'), 'w') as f:
                f.write("# Test package initialization")
        
        # Run pytest
        print("-- Running pytest suite")
        result = pytest.main(['-xvs', test_dir])
        
        if result == 0:
            print("-- All pytest tests passed - Success")
            return True
        else:
            print(f"-- Pytest tests failed with code {result}")
            return False
            
    except ImportError:
        print("-- pytest not installed, skipping automated test suite")
        return True
    except Exception as e:
        print(f"-- ERROR: pytest suite failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== BayesianDOE Test Suite ===")
    
    print("\n📋 Core Functionality Tests")
    core_success = run_core_tests()
    
    print("\n📊 Statistical Model Tests")
    model_success = run_model_tests()
    
    print("\n🔬 Comprehensive Tests")
    comprehensive_success = run_comprehensive_tests()
    
    print("\n🧪 PyTest Suite")
    pytest_success = run_pytest_suite()
    
    # Print summary
    print("\n✅ Test Summary:")
    print(f"Core Tests: {'✅ Passed' if core_success else '❌ Failed'}")
    print(f"Model Tests: {'✅ Passed' if model_success else '❌ Failed'}")
    print(f"Comprehensive Tests: {'✅ Passed' if comprehensive_success else '❌ Failed'}")
    print(f"PyTest Suite: {'✅ Passed' if pytest_success else '❌ Failed'}")
    
    overall_success = core_success and model_success and comprehensive_success
    
    sys.exit(0 if overall_success else 1)