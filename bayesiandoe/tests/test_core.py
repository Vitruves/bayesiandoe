import pytest
import numpy as np
from bayesiandoe.core import OptunaBayesianExperiment, _calculate_parameter_distance
from bayesiandoe.parameters import ChemicalParameter

def test_logger_fix():
    """Test that the logger is properly initialized."""
    model = OptunaBayesianExperiment()
    
    # Logger should be callable
    model.log("Test log message")
    
    # Should not raise an exception
    assert True

def test_parameter_distance_calculation():
    """Test parameter distance calculation function"""
    # Setup parameters
    parameters = {
        "temp": ChemicalParameter("temp", "continuous", 0, 100),
        "time": ChemicalParameter("time", "continuous", 0, 60),
        "solvent": ChemicalParameter("solvent", "categorical", choices=["A", "B", "C"])
    }
    
    # Test identical params
    params1 = {"temp": 50, "time": 30, "solvent": "A"}
    params2 = {"temp": 50, "time": 30, "solvent": "A"}
    assert _calculate_parameter_distance(params1, params2, parameters) == 0.0
    
    # Test fully different categorical
    params3 = {"temp": 50, "time": 30, "solvent": "B"}
    dist = _calculate_parameter_distance(params1, params3, parameters)
    assert dist > 0.0 and dist <= 1.0
    
    # Test fully different continuous
    params4 = {"temp": 100, "time": 60, "solvent": "A"}
    dist = _calculate_parameter_distance(params1, params4, parameters)
    assert dist > 0.0 and dist <= 1.0
    
    # Test missing params
    params5 = {"temp": 50}
    dist = _calculate_parameter_distance(params1, params5, parameters)
    assert dist > 0.0, "Missing parameters should result in non-zero distance"
    
    # Test empty params
    assert _calculate_parameter_distance({}, {}, parameters) == 1.0

def test_acquisition_functions():
    """Test acquisition function calculations"""
    model = OptunaBayesianExperiment()
    
    # EI test
    ei = model.acquisition_function_ei(0.8, 0.2, 0.7, xi=0.01)
    assert ei > 0.0
    
    # Edge cases
    assert model.acquisition_function_ei(0.5, 0.0, 0.7) == 0.0  # Zero std
    assert model.acquisition_function_ei(0.5, 0.1, 0.7) == 0.0  # Mean < best

    # PI test
    pi = model.acquisition_function_pi(0.8, 0.2, 0.7, xi=0.01)
    assert 0.0 <= pi <= 1.0
    
    # UCB test
    ucb = model.acquisition_function_ucb(0.7, 0.2, 0.0, beta=2.0)
    assert ucb == 0.7 + 2.0 * 0.2

def test_experiment_model_basic():
    """Test basic experiment model functionality"""
    model = OptunaBayesianExperiment()
    
    # Test parameter addition
    p1 = ChemicalParameter("temp", "continuous", 0, 100)
    p2 = ChemicalParameter("time", "discrete", 1, 10)
    p3 = ChemicalParameter("solvent", "categorical", choices=["A", "B", "C"])
    
    model.add_parameter(p1)
    model.add_parameter(p2)
    model.add_parameter(p3)
    
    assert len(model.parameters) == 3
    assert "temp" in model.parameters
    assert "time" in model.parameters
    assert "solvent" in model.parameters
    
    # Test objective setting
    model.set_objectives({"yield": 1.0})
    assert "yield" in model.objectives
    assert model.objective_weights["yield"] == 1.0
    
    # Test experiment addition
    exp1 = {
        "params": {"temp": 50, "time": 5, "solvent": "A"},
        "results": {"yield": 0.75}
    }
    
    model.experiments.append(exp1)
    
    # Test random suggestion generation
    suggestions = model._suggest_random(3)
    assert len(suggestions) == 3
    for suggestion in suggestions:
        assert "temp" in suggestion
        assert "time" in suggestion
        assert "solvent" in suggestion
        assert 0 <= suggestion["temp"] <= 100
        assert 1 <= suggestion["time"] <= 10
        assert suggestion["solvent"] in ["A", "B", "C"]
    
    # Test memoization in acquisition function
    if not hasattr(model, '_acq_cache'):
        model._acq_cache = {}
    
    # First call - should add to cache
    result1 = model.evaluate_acquisition(0.8, 0.2, 0.7)
    
    # Second call with same params - should hit cache
    cache_size = len(model._acq_cache)
    result2 = model.evaluate_acquisition(0.8, 0.2, 0.7)
    
    assert result1 == result2
    assert len(model._acq_cache) == cache_size  # No new entries added