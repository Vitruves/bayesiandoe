# tests/test_data_processing.py
import pytest
import tempfile
import os
import json
from bayesiandoe.core import OptunaBayesianExperiment
from bayesiandoe.parameters import ChemicalParameter

class TestModelSerialization:
    def setup_method(self):
        self.model = OptunaBayesianExperiment()
        self.model.add_parameter(ChemicalParameter("temp", "continuous", 0, 100))
        self.model.add_parameter(ChemicalParameter("time", "discrete", 1, 10))
        self.model.add_parameter(ChemicalParameter("solvent", "categorical", choices=["A", "B", "C"]))
        self.model.set_objectives({"yield": 1.0})
        
        # Add some experiments
        self.model.experiments = [
            {
                "params": {"temp": 50, "time": 5, "solvent": "A"},
                "results": {"yield": 0.75},
                "score": 0.75
            },
            {
                "params": {"temp": 70, "time": 7, "solvent": "B"},
                "results": {"yield": 0.85},
                "score": 0.85
            }
        ]
        
    def test_save_load_model(self):
        """Test model serialization and deserialization."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            # Save the model
            self.model.save_model(filepath)
            
            # Check file exists and is not empty
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            
            # Create a new model and load into it
            new_model = OptunaBayesianExperiment()
            new_model.load_model(filepath)
            
            # Check parameters were loaded correctly
            assert set(new_model.parameters.keys()) == {"temp", "time", "solvent"}
            assert new_model.parameters["temp"].param_type == "continuous"
            assert new_model.parameters["temp"].low == 0
            assert new_model.parameters["temp"].high == 100
            
            assert new_model.parameters["solvent"].param_type == "categorical"
            assert set(new_model.parameters["solvent"].choices) == {"A", "B", "C"}
            
            # Check experiments were loaded
            assert len(new_model.experiments) == 2
            assert new_model.experiments[0]["results"]["yield"] == 0.75
            assert new_model.experiments[1]["results"]["yield"] == 0.85
            
            # Check objectives
            assert new_model.objectives == ["yield"]
            assert new_model.objective_weights["yield"] == 1.0
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
                
    def test_failed_load_handling(self):
        """Test handling of loading from invalid file."""
        # Create an invalid model file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filepath = tmp.name
            tmp.write(b'{"invalid": "json"')  # Malformed JSON
        
        try:
            # Loading should not crash but create empty model
            model = OptunaBayesianExperiment()
            model.load_model(filepath)
            
            # Model should be empty but usable
            assert len(model.parameters) == 0
            assert len(model.experiments) == 0
            
            # Should be able to add parameters
            model.add_parameter(ChemicalParameter("test", "continuous", 0, 1))
            assert "test" in model.parameters
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
                
class TestParameterOperations:
    def test_parameter_suggestion(self):
        """Test parameter value suggestion."""
        # Continuous parameter without prior
        p1 = ChemicalParameter("temp", "continuous", 0, 100)
        for _ in range(10):
            value = p1.suggest_value()
            assert 0 <= value <= 100
            
        # Discrete parameter
        p2 = ChemicalParameter("time", "discrete", 1, 5)
        for _ in range(10):
            value = p2.suggest_value()
            assert 1 <= value <= 5
            assert isinstance(value, int)
            
        # Categorical parameter
        p3 = ChemicalParameter("cat", "categorical", choices=["A", "B", "C"])
        for _ in range(10):
            value = p3.suggest_value()
            assert value in ["A", "B", "C"]
            
    def test_parameter_prior(self):
        """Test parameter prior influence."""
        p = ChemicalParameter("temp", "continuous", 0, 100)
        
        # Set a strong prior
        p.set_prior(mean=25, std=5)
        
        # Generate many samples and check distribution
        samples = [p.suggest_value() for _ in range(100)]
        
        # Most samples should be near the prior mean
        near_mean = sum(1 for v in samples if 15 <= v <= 35)
        assert near_mean >= 60  # At least 60% should be near mean with tight std
        
        # Few samples should be far from mean
        far_from_mean = sum(1 for v in samples if v > 50)
        assert far_from_mean <= 10  # Less than 10% should be far