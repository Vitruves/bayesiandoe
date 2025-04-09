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
        if self.param_type == "continuous" or self.param_type == "discrete":
            return self.suggest_numeric_value()
        elif self.param_type == "categorical":
            return self.suggest_categorical_value()
        return None
        
    def suggest_categorical_value(self):
        if not self.choices:
            return None
        if hasattr(self, 'categorical_preferences') and self.categorical_preferences:
            total_weight = sum(self.categorical_preferences.values())
            if total_weight > 0:
                weights = [self.categorical_preferences.get(choice, 1) / total_weight for choice in self.choices]
                try:
                    return random.choices(self.choices, weights=weights, k=1)[0]
                except ValueError:
                    pass
        return random.choice(self.choices)

    def suggest_numeric_value(self):
        value = None
        
        if self.prior_mean is not None and self.prior_std is not None and self.prior_std > 0:
            try:
                a = (self.low - self.prior_mean) / self.prior_std
                b = (self.high - self.prior_mean) / self.prior_std
                value = stats.truncnorm.rvs(a, b, loc=self.prior_mean, scale=self.prior_std, size=1)[0]
            except ImportError:
                pass
            except Exception as e:
                print(f"Error in suggest_numeric_value: {e}")
                pass
        
        if value is None:
            if self.param_type == "continuous":
                value = random.uniform(self.low, self.high)
            elif self.param_type == "discrete":
                value = random.randint(int(self.low), int(self.high))
            else:
                return None
                
        # Apply rounding if it's a continuous parameter
        if self.param_type == "continuous" and settings.auto_round:
            value = settings.apply_rounding(value)
        elif self.param_type == "discrete":
            value = int(round(value))
            
        return value