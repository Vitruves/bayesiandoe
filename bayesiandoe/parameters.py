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
        self.linked_parameters = {}
        self.initialized_defaults = set()  # Track which default ranges have been applied (as name keys)
        
        # Apply default ranges based on parameter name if logical units are enabled
        if settings.use_logical_units and param_type in ["continuous", "discrete"]:
            self._apply_logical_unit_defaults()
        
    def _apply_logical_unit_defaults(self):
        """Apply logical unit defaults based on parameter name"""
        # Only apply if range is default (None or 0/1) AND not already initialized for this name
        should_apply = (
            (self.low is None or self.low == 0) and
            (self.high is None or self.high == 1) and
            (self.name.lower() not in self.initialized_defaults)
        )

        if not should_apply:
            # If range is already set, still add to initialized_defaults if not present
            if self.name.lower() not in self.initialized_defaults:
                 self.initialized_defaults.add(self.name.lower())
            return

        rounding_config = settings.get_parameter_rounding(self.name, self.param_type)
        if rounding_config and "min" in rounding_config and "max" in rounding_config:
            # Apply defaults from settings only if they are still default
            if self.low is None or self.low == 0:
                self.low = rounding_config["min"]
            if self.high is None or self.high == 1:
                self.high = rounding_config["max"]

            # Update units if they're not already set
            if self.units is None and "unit" in rounding_config:
                self.units = rounding_config["unit"]

            # Add parameter name to the initialized set
            self.initialized_defaults.add(self.name.lower())
        
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
        
    def add_linked_parameter(self, param_name, influence_strength=0.5, influence_type="positive"):
        self.linked_parameters[param_name] = {
            "strength": float(influence_strength), 
            "type": influence_type
        }
        
    def remove_linked_parameter(self, param_name):
        if param_name in self.linked_parameters:
            del self.linked_parameters[param_name]
            
    def get_linked_parameters(self):
        return self.linked_parameters
        
    def adjust_for_linked_parameters(self, param_values, all_parameters):
        if not self.linked_parameters:
            return None
            
        adjustments = {}
        
        if self.param_type == "continuous" or self.param_type == "discrete":
            adjusted_mean = self.prior_mean if self.prior_mean is not None else (self.low + self.high) / 2
            adjusted_std = self.prior_std if self.prior_std is not None else (self.high - self.low) / 4
            total_influence = 0
            
            for linked_name, link_info in self.linked_parameters.items():
                if linked_name in param_values and linked_name in all_parameters:
                    linked_param = all_parameters[linked_name]
                    linked_value = param_values[linked_name]
                    
                    if linked_param.param_type == "continuous" or linked_param.param_type == "discrete":
                        norm_value = (linked_value - linked_param.low) / (linked_param.high - linked_param.low)
                        influence = link_info["strength"]
                        
                        if link_info["type"] == "negative":
                            norm_value = 1.0 - norm_value
                            
                        shift = (norm_value - 0.5) * 2 * influence * (self.high - self.low) * 0.2
                        adjusted_mean += shift
                        total_influence += abs(influence)
                        
                    elif linked_param.param_type == "categorical" and linked_param.choices:
                        choices_count = len(linked_param.choices)
                        if choices_count > 1:
                            choice_index = linked_param.choices.index(linked_value) if linked_value in linked_param.choices else 0
                            norm_value = choice_index / (choices_count - 1)
                            
                            if link_info["type"] == "negative":
                                norm_value = 1.0 - norm_value
                                
                            shift = (norm_value - 0.5) * 2 * link_info["strength"] * (self.high - self.low) * 0.2
                            adjusted_mean += shift
                            total_influence += abs(link_info["strength"])
            
            if total_influence > 0:
                adjusted_mean = max(self.low, min(self.high, adjusted_mean))
                adjustments = {"mean": adjusted_mean, "std": adjusted_std}
                
        elif self.param_type == "categorical" and self.choices:
            preference_weights = {}
            base_weight = 1.0
            
            for choice in self.choices:
                preference_weights[choice] = base_weight
                
            for linked_name, link_info in self.linked_parameters.items():
                if linked_name in param_values and linked_name in all_parameters:
                    linked_param = all_parameters[linked_name]
                    linked_value = param_values[linked_name]
                    
                    if linked_param.param_type == "categorical" and linked_param.choices:
                        for choice in self.choices:
                            for linked_choice in linked_param.choices:
                                if linked_value == linked_choice:
                                    chemical_similarity = self._calculate_chemical_similarity(choice, linked_choice)
                                    influence = link_info["strength"] * chemical_similarity
                                    
                                    if link_info["type"] == "negative":
                                        influence = -influence
                                        
                                    preference_weights[choice] *= (1.0 + influence)
            
            adjustments = {"categorical_preferences": preference_weights}
                
        return adjustments
    
    def _calculate_chemical_similarity(self, compound1, compound2):
        similar_pairs = [
            ("DMSO", "Sulfolane"), ("DMSO", "NMP"), ("DMSO", "DMF"),
            ("THF", "Dioxane"), ("THF", "MTBE"), ("THF", "2-MeTHF"),
            ("MeOH", "EtOH"), ("MeOH", "i-PrOH"), ("EtOH", "i-PrOH"),
            ("Water", "MeOH"), ("Water", "EtOH"),
            ("DCM", "Chloroform"), ("DCM", "DCE"),
            ("Toluene", "Benzene"), ("Toluene", "Xylene"),
            ("Hexane", "Pentane"), ("Hexane", "Heptane"),
            ("Acetone", "Acetonitrile"), ("Acetone", "MEK")
        ]
        
        if compound1 == compound2:
            return 1.0
            
        for pair in similar_pairs:
            if (compound1 in pair and compound2 in pair):
                return 0.7
                
        return 0.1
        
    def suggest_value(self):
        """Generate a parameter value based on parameter type and prior knowledge.
        
        Returns:
            Generated parameter value based on prior information if available,
            otherwise using parameter ranges or choices.
        """
        if self.param_type == "continuous" or self.param_type == "discrete":
            value = self.suggest_numeric_value()
            
            # Apply logical unit rounding if enabled
            if settings.use_logical_units and value is not None:
                rounding_config = settings.get_parameter_rounding(self.name, self.param_type)
                if rounding_config and "interval" in rounding_config:
                    value = settings.round_to_interval(value, rounding_config["interval"])
                    # Ensure value stays within bounds after rounding
                    value = max(self.low, min(self.high, value))
            
            # Force integer conversion for discrete parameters
            if self.param_type == "discrete" and value is not None:
                value = int(value)
                
            return value
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
                
                # For discrete parameters, round immediately to ensure integer
                if self.param_type == "discrete":
                    value = int(round(value))
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
                    
                    # For discrete parameters, round immediately
                    if self.param_type == "discrete":
                        value = int(round(value))
                    
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