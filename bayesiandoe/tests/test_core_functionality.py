# tests/test_core_functionality.py
import pytest
import numpy as np
from bayesiandoe.core import OptunaBayesianExperiment, _calculate_parameter_distance
from bayesiandoe.parameters import ChemicalParameter

class TestAcquisitionFunctions:
    def setup_method(self):
        self.model = OptunaBayesianExperiment()
        
    def test_expected_improvement(self):
        """Test expected improvement acquisition function."""
        # Simple positive case
        ei = self.model.acquisition_function_ei(0.8, 0.2, 0.7, xi=0.01)
        assert ei > 0
        
        # Mean below best value should give zero improvement
        ei_zero = self.model.acquisition_function_ei(0.5, 0.2, 0.7, xi=0.01)
        assert ei_zero == 0
        
        # Higher uncertainty should give higher EI
        ei_high_std = self.model.acquisition_function_ei(0.8, 0.4, 0.7, xi=0.01)
        assert ei_high_std > ei
        
    def test_probability_improvement(self):
        """Test probability of improvement acquisition function."""
        # Simple positive case
        pi = self.model.acquisition_function_pi(0.8, 0.2, 0.7, xi=0.01)
        assert 0 < pi < 1
        
        # Mean below best value should give zero probability
        pi_low = self.model.acquisition_function_pi(0.5, 0.2, 0.7, xi=0.01)
        assert pi_low == 0.0
        
        # Higher mean should give higher PI
        pi_high_mean = self.model.acquisition_function_pi(0.9, 0.2, 0.7, xi=0.01)
        assert pi_high_mean > pi
        
    def test_upper_confidence_bound(self):
        """Test UCB acquisition function."""
        # Basic calculation
        ucb = self.model.acquisition_function_ucb(0.7, 0.2, 0, beta=2.0)
        assert ucb == 0.7 + 2.0 * 0.2
        
        # Higher beta should give higher UCB
        ucb_high_beta = self.model.acquisition_function_ucb(0.7, 0.2, 0, beta=3.0)
        assert ucb_high_beta > ucb
        
    def test_acquisition_caching(self):
        """Test acquisition function caching."""
        # Set up cache
        if not hasattr(self.model, '_acq_cache'):
            self.model._acq_cache = {}
            
        # First call should compute and cache
        result1 = self.model.evaluate_acquisition(0.8, 0.2, 0.7)
        cache_size = len(self.model._acq_cache)
        assert cache_size > 0
        
        # Second identical call should use cache
        result2 = self.model.evaluate_acquisition(0.8, 0.2, 0.7)
        assert result1 == result2
        assert len(self.model._acq_cache) == cache_size  # No new entries


class TestParameterHandling:
    def setup_method(self):
        self.model = OptunaBayesianExperiment()
        
    def test_parameter_validation(self):
        """Test parameter validation during addition."""
        # Valid parameter should be accepted
        p1 = ChemicalParameter("temp", "continuous", 0, 100)
        self.model.add_parameter(p1)
        assert "temp" in self.model.parameters
        
        # Parameter with low >= high should raise ValueError
        p2 = ChemicalParameter("invalid", "continuous", 100, 0)
        with pytest.raises(ValueError):
            self.model.add_parameter(p2)
            
        # Categorical parameter without choices should raise ValueError
        p3 = ChemicalParameter("cat", "categorical")
        with pytest.raises(ValueError):
            self.model.add_parameter(p3)
            
    def test_parameter_normalization(self):
        """Test parameter normalization for model training."""
        self.model.add_parameter(ChemicalParameter("temp", "continuous", 0, 100))
        self.model.add_parameter(ChemicalParameter("time", "continuous", 10, 60))
        self.model.add_parameter(ChemicalParameter("solvent", "categorical", choices=["A", "B", "C"]))
        
        # Test continuous normalization
        params = {"temp": 50, "time": 35, "solvent": "B"}
        normalized = self.model._normalize_params(params)
        
        # Continuous params should be scaled to [0,1]
        assert 0.45 < normalized[0] < 0.55  # temp=50 should be ~0.5
        assert 0.45 < normalized[1] < 0.55  # time=35 should be ~0.5
        
        # One-hot encoding for categorical
        assert normalized[2] == 0  # First category (A) = 0
        assert normalized[3] == 1  # Second category (B) = 1
        assert normalized[4] == 0  # Third category (C) = 0
        
    def test_parameter_distance(self):
        """Test parameter distance calculation."""
        params = {
            "temp": ChemicalParameter("temp", "continuous", 0, 100),
            "cat": ChemicalParameter("cat", "categorical", choices=["X", "Y", "Z"])
        }
        
        # Same parameters should have zero distance
        p1 = {"temp": 50, "cat": "Y"}
        p2 = {"temp": 50, "cat": "Y"}
        assert _calculate_parameter_distance(p1, p2, params) == 0
        
        # Different continuous parameter
        p3 = {"temp": 75, "cat": "Y"}
        dist1 = _calculate_parameter_distance(p1, p3, params)
        assert 0.2 < dist1 < 0.3  # Should be around 0.25
        
        # Different categorical parameter
        p4 = {"temp": 50, "cat": "Z"}
        dist2 = _calculate_parameter_distance(p1, p4, params)
        assert 0.7 < dist2 < 1.0  # Should be relatively high
        
        # Both different
        p5 = {"temp": 100, "cat": "X"}
        dist3 = _calculate_parameter_distance(p1, p5, params)
        assert dist3 > max(dist1, dist2)  # Should be higher than either individual difference


class TestExperimentOptimization:
    def setup_method(self):
        self.model = OptunaBayesianExperiment()
        self.model.add_parameter(ChemicalParameter("temp", "continuous", 0, 100))
        self.model.add_parameter(ChemicalParameter("time", "continuous", 0, 60))
        self.model.set_objectives({"yield": 1.0})
        
    def test_experiment_scoring(self):
        """Test experiment scoring calculation."""
        # Single objective
        results = {"yield": 0.75}
        score = self.model._calculate_composite_score(results)
        assert score == 0.75
        
        # Multiple weighted objectives
        self.model.set_objectives({"yield": 0.7, "purity": 0.3})
        results2 = {"yield": 0.8, "purity": 0.6}
        score2 = self.model._calculate_composite_score(results2)
        assert score2 == 0.8*0.7 + 0.6*0.3
        
        # Missing objective should be handled
        results3 = {"yield": 0.9}
        score3 = self.model._calculate_composite_score(results3)
        assert score3 == 0.9*0.7  # Only yield is present
        
    def test_experiment_suggestion(self):
        """Test experiment suggestion methods."""
        # Random suggestions
        random_sugg = self.model._suggest_random(5)
        assert len(random_sugg) == 5
        for sugg in random_sugg:
            assert "temp" in sugg
            assert "time" in sugg
            assert 0 <= sugg["temp"] <= 100
            assert 0 <= sugg["time"] <= 60
            
        # Sobol suggestions if available
        try:
            sobol_sugg = self.model._suggest_with_sobol(3)
            assert len(sobol_sugg) == 3
            # Sobol sequence should be more evenly distributed
            temps = [s["temp"] for s in sobol_sugg]
            assert max(temps) - min(temps) > 20  # Should cover a good range
        except ImportError:
            pass  # Skip if scipy not available
            
    def test_diverse_subset_selection(self):
        """Test selection of diverse experiment subset."""
        candidates = [
            {"temp": 10, "time": 10},
            {"temp": 20, "time": 20},
            {"temp": 30, "time": 30},
            {"temp": 40, "time": 40},
            {"temp": 90, "time": 50},
            {"temp": 95, "time": 55},
        ]
        
        # With high diversity weight, should select more spread out points
        diverse = self.model.select_diverse_subset(candidates, 3, diversity_weight=0.9)
        assert len(diverse) == 3
        
        # Calculate distances between selected points
        distances = []
        for i in range(len(diverse)):
            for j in range(i+1, len(diverse)):
                dist = self.model.calculate_experiment_distance(diverse[i], diverse[j])
                distances.append(dist)
                
        # Average distance should be relatively high
        avg_dist = sum(distances) / len(distances)
        assert avg_dist > 0.3  # Reasonable threshold for diversity

def test_parameter_distance_edge_cases():
    """Test parameter distance calculation with edge cases."""
    from bayesiandoe.core import _calculate_parameter_distance
    from bayesiandoe.parameters import ChemicalParameter
    
    parameters = {
        "temp": ChemicalParameter("temp", "continuous", 0, 100),
        "time": ChemicalParameter("time", "continuous", 0, 60),
        "solvent": ChemicalParameter("solvent", "categorical", choices=["A", "B", "C"])
    }
    
    # Case 1: Completely different parameters
    params1 = {"temp": 0, "time": 0, "solvent": "A"}
    params2 = {"temp": 100, "time": 60, "solvent": "C"}
    dist = _calculate_parameter_distance(params1, params2, parameters)
    assert dist > 0.9, "Completely different parameters should have distance close to 1.0"
    
    # Case 2: One parameter missing from each
    params3 = {"temp": 50, "time": 30}
    params4 = {"temp": 50, "solvent": "B"}
    dist = _calculate_parameter_distance(params3, params4, parameters)
    assert dist > 0.0, "Different missing parameters should have non-zero distance"
    
    # Case 3: Parameters outside model parameters
    params5 = {"temp": 50, "time": 30, "solvent": "A", "extra": 100}
    params6 = {"temp": 50, "time": 30, "solvent": "A", "other": 200}
    dist = _calculate_parameter_distance(params5, params6, parameters)
    assert dist == 0.0, "Extra parameters not in the model should be ignored"