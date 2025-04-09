import random
import numpy as np
from scipy import stats
from .core import settings

class ChemicalParameter:
    def __init__(self, name, param_type, low=None, high=None, choices=None, units=None):
        self.name = name
        self.param_type = param_type
        self.low = low
        self.high = high
        self.choices = choices
        self.units = units
        self.prior_mean = None
        self.prior_std = None
        
    def to_optuna_param(self, trial):
        if self.param_type == "continuous":
            return trial.suggest_float(self.name, self.low, self.high)
        elif self.param_type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        elif self.param_type == "discrete":
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        
    def set_prior(self, mean=None, std=None):
        # Properly set prior values
        if mean is not None and std is not None:
            self.prior_mean = float(mean)
            self.prior_std = float(std)
        else:
            # If any value is None, reset both to avoid partial prior states
            self.prior_mean = None
            self.prior_std = None
        
    def suggest_value(self):
        """Generate a parameter value based on parameter type and prior knowledge.
        
        Returns:
            Generated parameter value based on prior information if available,
            otherwise using parameter ranges or choices.
        """
        if self.param_type == "continuous" or self.param_type == "discrete":
            return self.suggest_numeric_value()
        elif self.param_type == "categorical":
            return self.suggest_categorical_value()
        return None
        
    def suggest_categorical_value(self):
        """Generate a categorical parameter value using preference weights if available.
        
        Returns:
            Selected category from available choices based on preferences.
        """
        if not self.choices:
            return None
        
        # If we have categorical preferences defined, use them
        if hasattr(self, 'categorical_preferences') and self.categorical_preferences:
            # Calculate sum of all weights
            total_weight = 0
            weights = []
            
            for choice in self.choices:
                weight = self.categorical_preferences.get(choice, 1.0)
                weights.append(weight)
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                
                try:
                    # Use weighted random choice
                    return random.choices(self.choices, weights=weights, k=1)[0]
                except (ValueError, ImportError):
                    pass
                
        # Fallback to simple random choice
        return random.choice(self.choices)

    def suggest_numeric_value(self):
        """Generate a numeric parameter value using prior information if available.
        
        Returns:
            Generated value from truncated normal distribution or uniform range.
        """
        value = None
        
        # If we have prior information, use it to inform the search
        if self.prior_mean is not None and self.prior_std is not None and self.prior_std > 0:
            try:
                from scipy import stats
                
                # Calculate truncated normal parameters
                a = (self.low - self.prior_mean) / self.prior_std
                b = (self.high - self.prior_mean) / self.prior_std
                
                # Sample from truncated normal distribution
                value = stats.truncnorm.rvs(a, b, loc=self.prior_mean, scale=self.prior_std, size=1)[0]
            except ImportError:
                # Fallback without scipy
                import random
                import math
                
                # Approximate truncated normal with rejection sampling
                while value is None or value < self.low or value > self.high:
                    # Box-Muller transform for normal distribution
                    u1 = random.random()
                    u2 = random.random()
                    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    
                    # Scale and shift
                    value = z0 * self.prior_std + self.prior_mean
                    
                    # Limit rejection attempts
                    if random.random() < 0.05:  # 5% chance to give up and use uniform
                        break
            except Exception as e:
                print(f"Error in suggest_numeric_value: {e}")
                value = None
        
        # Fallback to uniform sampling
        if value is None or value < self.low or value > self.high:
            import random
            if self.param_type == "continuous":
                value = random.uniform(self.low, self.high)
            elif self.param_type == "discrete":
                value = random.randint(int(self.low), int(self.high))
            else:
                return None
        
        # Apply rounding if needed
        if self.param_type == "continuous" and settings.auto_round:
            value = settings.apply_rounding(value)
        elif self.param_type == "discrete":
            value = int(round(value))
        
        return value