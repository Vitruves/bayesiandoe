import numpy as np
import datetime
import pickle
import optuna
from typing import Dict, List, Any, Optional
from optuna.samplers import BaseSampler, TPESampler
from scipy import stats

class AppSettings:
    """Global application settings"""
    def __init__(self):
        # Number of decimal places for automatic rounding
        self.rounding_precision = 4
        
        # Auto-round values when displaying
        self.auto_round = True
        
        # Smart rounding: use fewer decimals for larger values
        self.smart_rounding = True
        
        # Rounding behavior for different value ranges
        self.rounding_rules = {
            0.0001: 6,  # For values smaller than 0.0001, use 6 decimal places
            0.001: 5,   # For values between 0.0001 and 0.001, use 5 decimal places
            0.01: 4,    # For values between 0.001 and 0.01, use 4 decimal places
            0.1: 3,     # For values between 0.01 and 0.1, use 3 decimal places
            1.0: 2,     # For values between 0.1 and 1.0, use 2 decimal places
            10.0: 2,    # For values between 1.0 and 10.0, use 2 decimal places
            100.0: 1,   # For values between 10.0 and 100.0, use 1 decimal place
            1000.0: 1,  # For values between 100.0 and 1000.0, use 1 decimal place
            float('inf'): 0  # For values larger than 1000.0, use 0 decimal places
        }
    
    def get_precision_for_value(self, value: float) -> int:
        """Determine the appropriate decimal precision for a given value"""
        if not self.smart_rounding:
            return self.rounding_precision
            
        abs_value = abs(value)
        for threshold, precision in sorted(self.rounding_rules.items()):
            if abs_value < threshold:
                return precision
        return 0  # Default case
        
    def apply_rounding(self, value: float) -> float:
        """Apply appropriate rounding to a value"""
        if not self.auto_round:
            return value
            
        precision = self.get_precision_for_value(value)
        return round(value, precision)
        
    def format_value(self, value: Any) -> str:
        """Format a value for display with appropriate rounding"""
        if isinstance(value, float):
            if not self.auto_round:
                return f"{value}"
                
            precision = self.get_precision_for_value(value)
            format_str = f"{{:.{precision}f}}"
            return format_str.format(value).rstrip('0').rstrip('.') if '.' in format_str.format(value) else format_str.format(value)
        else:
            return str(value)

# Global application settings
settings = AppSettings()

def _calculate_parameter_distance(params1, params2, parameters):
    if not params1 or not params2:
        return float('inf')

    common_params = set(params1.keys()) & set(params2.keys())
    if not common_params:
        return float('inf')

    squared_diffs = []
    for param_name in common_params:
        if param_name in parameters:
            param = parameters[param_name]
            p_type = param.param_type
            val1 = params1[param_name]
            val2 = params2[param_name]

            if p_type == "categorical":
                squared_diff = 0 if val1 == val2 else 1
            else:
                p_range = param.high - param.low
                if p_range > 0:
                    try:
                        norm_val1 = (float(val1) - param.low) / p_range
                        norm_val2 = (float(val2) - param.low) / p_range
                        squared_diff = (norm_val1 - norm_val2) ** 2
                    except (ValueError, TypeError):
                        squared_diff = 1
                else:
                    squared_diff = 0

            squared_diffs.append(squared_diff)

    if not squared_diffs:
        return float('inf')

    return np.sqrt(sum(squared_diffs) / len(squared_diffs))

class OptunaBayesianExperiment:
    def __init__(self):
        self.parameters = {}
        self.categorical_mappings = {}
        self.objectives = ["yield", "purity", "selectivity"]
        self.objective_weights = {"yield": 1.0, "purity": 1.0, "selectivity": 1.0}
        self.experiments = []
        self.planned_experiments = []
        self.study: Optional[optuna.Study] = None
        self.sampler: BaseSampler = TPESampler(seed=42, consider_prior=True, prior_weight=1.0)
        
        # Advanced settings
        self.acquisition_function = "ei"  # Options: ei (expected improvement), pi (probability of improvement), ucb (upper confidence bound)
        self.exploitation_weight = 0.7  # Higher values favor exploitation (0-1)
        self.use_thompson_sampling = True  # Use Thompson sampling for uncertainty handling
        self.min_points_in_model = 3  # Minimum points required before using model-based suggestions
        self.exploration_noise = 0.05  # Noise level for exploration-exploitation balance
        
        self.create_study()
        
    def add_parameter(self, param):
        # Validate parameter before adding
        if not param.name or not param.param_type:
            raise ValueError("Parameter must have a name and type")
            
        if param.param_type == "categorical":
            if not param.choices or len(param.choices) == 0:
                raise ValueError(f"Categorical parameter '{param.name}' must have at least one choice")
            # Ensure choices are unique
            param.choices = list(dict.fromkeys(param.choices))
        elif param.param_type in ["continuous", "discrete"]:
            if param.low is None or param.high is None:
                raise ValueError(f"Parameter '{param.name}' must have low and high values")
            if param.low >= param.high:
                raise ValueError(f"Parameter '{param.name}' must have low < high, got: low={param.low}, high={param.high}")
        
        # Add parameter to the model
        self.parameters[param.name] = param
        
        # Update categorical mappings if needed
        if param.param_type == "categorical" and param.choices:
            self.categorical_mappings[param.name] = {val: i for i, val in enumerate(param.choices)}
        
        # Recreate study if it exists
        if self.study:
            self.create_study()
            
    def remove_parameter(self, name):
        if name in self.parameters:
            del self.parameters[name]
            if name in self.categorical_mappings:
                del self.categorical_mappings[name]
            if self.study:
                self.create_study()

    def create_study(self, sampler: Optional[BaseSampler] = None):
        if sampler:
            self.sampler = sampler

        def dummy_objective(trial: optuna.Trial) -> float:
            return 0.0

        self.study = optuna.create_study(
            study_name="chemical_optimization",
            direction="maximize",
            sampler=self.sampler,
        )

    def set_acquisition_function(self, acq_func: str):
        """
        Set the acquisition function for Bayesian optimization
        
        Args:
            acq_func: One of 'ei' (expected improvement), 'pi' (probability of improvement),
                    'ucb' (upper confidence bound)
        """
        if acq_func not in ["ei", "pi", "ucb"]:
            raise ValueError(f"Invalid acquisition function: {acq_func}")
        self.acquisition_function = acq_func
        
    def set_exploitation_weight(self, weight: float):
        """Set the exploitation weight (0-1) for balancing exploration-exploitation"""
        if not 0 <= weight <= 1:
            raise ValueError("Exploitation weight must be between 0 and 1")
        self.exploitation_weight = weight
        
    def _calculate_composite_score(self, results: Dict[str, float]) -> float:
        """Calculate composite score from multiple objectives"""
        score = 0.0
        total_weight = 0.0
        for obj in self.objectives:
            if obj in results and results[obj] is not None:
                weight = self.objective_weights.get(obj, 1.0)
                score += min(results[obj], 1.0) * weight
                total_weight += weight
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_sigmoid_score(self, score: float) -> float:
        """
        Apply sigmoid transformation to emphasize high scores
        This helps to accelerate convergence to 100% goals
        """
        # Scale to emphasize scores above 0.7
        k = 10  # Steepness parameter
        x0 = 0.7  # Midpoint of sigmoid
        
        # Modified sigmoid that maps [0,1] -> [0,1]
        base_sigmoid = 1 / (1 + np.exp(-k * (score - x0)))
        sigmoid_0 = 1 / (1 + np.exp(-k * (0 - x0)))
        sigmoid_1 = 1 / (1 + np.exp(-k * (1 - x0)))
        
        # Normalize to [0,1] range
        normalized = (base_sigmoid - sigmoid_0) / (sigmoid_1 - sigmoid_0)
        
        # Blend with original score for smoother transition
        blend_weight = 0.7  # Weight of sigmoid in final score
        blended = blend_weight * normalized + (1 - blend_weight) * score
        
        return blended
        
    def add_experiment_result(self, params: Dict[str, Any], results: Dict[str, float]):
        # Calculate the composite score from all objectives
        raw_score = self._calculate_composite_score(results)
        
        # Apply sigmoid transformation to accelerate convergence to 100%
        composite_score = self._calculate_sigmoid_score(raw_score)

        # Create a properly formatted experiment entry
        experiment = {
            'params': params.copy(),  # Copy to avoid reference issues
            'results': results.copy(),
            'timestamp': datetime.datetime.now().isoformat(),
            'raw_score': raw_score,
            'score': composite_score
        }
        
        # Add to our internal experiment list
        self.experiments.append(experiment)

        # Create a trial for Optuna
        try:
            trial = optuna.trial.create_trial(
                params=params,
                distributions=self._get_distributions(),
                value=composite_score
            )

            # Add the trial to the Optuna study
            self.study.add_trial(trial)
        except Exception as e:
            print(f"Warning: Could not add trial to Optuna study: {e}")

    def _get_distributions(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        distributions = {}
        for name, param in self.parameters.items():
            try:
                if param.param_type == "continuous":
                    if param.low is None or param.high is None or param.low >= param.high:
                        print(f"Warning: Invalid range for continuous parameter {name}: [{param.low}, {param.high}]")
                        continue
                    distributions[name] = optuna.distributions.FloatDistribution(param.low, param.high)
                elif param.param_type == "discrete":
                    if param.low is None or param.high is None or param.low >= param.high:
                        print(f"Warning: Invalid range for discrete parameter {name}: [{param.low}, {param.high}]")
                        continue
                    # Ensure values are integers
                    distributions[name] = optuna.distributions.IntDistribution(int(param.low), int(param.high))
                elif param.param_type == "categorical":
                    if not param.choices or len(param.choices) == 0:
                        print(f"Warning: No choices defined for categorical parameter {name}")
                        continue
                    distributions[name] = optuna.distributions.CategoricalDistribution(param.choices)
                else:
                    print(f"Warning: Unknown parameter type {param.param_type} for parameter {name}")
            except Exception as e:
                print(f"Error creating distribution for parameter {name}: {e}")
        return distributions

    def acquisition_function_ei(self, mean, std, best_value):
        """Expected Improvement acquisition function"""
        z = (mean - best_value) / (std + 1e-6)
        return (mean - best_value) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    
    def acquisition_function_pi(self, mean, std, best_value):
        """Probability of Improvement acquisition function"""
        z = (mean - best_value) / (std + 1e-6)
        return stats.norm.cdf(z)
        
    def acquisition_function_ucb(self, mean, std, best_value, beta=2.0):
        """Upper Confidence Bound acquisition function"""
        return mean + beta * std
        
    def evaluate_acquisition(self, mean, std, best_value):
        """Evaluate the selected acquisition function"""
        if self.acquisition_function == "ei":
            return self.acquisition_function_ei(mean, std, best_value)
        elif self.acquisition_function == "pi":
            return self.acquisition_function_pi(mean, std, best_value)
        elif self.acquisition_function == "ucb":
            # Dynamically adjust beta based on exploitation weight
            beta = 2.0 * (1 - self.exploitation_weight)
            return self.acquisition_function_ucb(mean, std, best_value, beta)
        else:
            # Default to expected improvement
            return self.acquisition_function_ei(mean, std, best_value)
            
    def _apply_thompson_sampling(self, candidates):
        """Apply Thompson sampling to candidate selection for increased exploration"""
        if not self.use_thompson_sampling or len(candidates) < 2:
            return candidates
            
        # Add random noise to scores for Thompson sampling
        for candidate in candidates:
            # Scale noise based on confidence (less noise for more confident predictions)
            if 'std' in candidate:
                noise_scale = candidate['std'] * self.exploration_noise
            else:
                noise_scale = self.exploration_noise
                
            noise = np.random.normal(0, noise_scale)
            if 'acquisition_value' in candidate:
                candidate['acquisition_value'] += noise
                
        # Resort candidates with Thompson sampling noise
        candidates.sort(key=lambda x: x.get('acquisition_value', 0), reverse=True)
        return candidates

    def suggest_experiments(self, n_suggestions=5) -> List[Dict[str, Any]]:
        if not self.parameters or not self.study:
            print("Warning: Cannot suggest experiments without parameters or study.")
            return []
            
        # Verify all parameters have valid distributions
        invalid_params = []
        for name, param in self.parameters.items():
            if param.param_type == "categorical" and (not param.choices or len(param.choices) == 0):
                invalid_params.append(name)
            elif param.param_type in ["continuous", "discrete"] and (param.low is None or param.high is None or param.low >= param.high):
                invalid_params.append(name)
                
        if invalid_params:
            error_msg = f"Invalid parameter state for: {', '.join(invalid_params)}"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)
            
        # Calculate number of points to acquire using Optuna and number to generate randomly
        n_optuna = n_suggestions
        n_random = 0
        
        # For very early stage optimization, use more random points to explore
        if len(self.experiments) < self.min_points_in_model:
            n_optuna = min(len(self.experiments), n_suggestions)
            n_random = n_suggestions - n_optuna
            
        suggestions = []
        distributions = self._get_distributions()
        
        # First, get Optuna suggestions
        if n_optuna > 0:
            try:
                for _ in range(n_optuna):
                    trial = self.study.ask(fixed_distributions=distributions)
                    suggestions.append(trial.params.copy())
            except Exception as e:
                print(f"Error during Optuna suggestion: {e}")
                n_random = n_suggestions  # Fall back to random sampling
                suggestions = []
        
        # Add random suggestions for exploration if needed
        for _ in range(n_random):
            params = {}
            try:
                for name, param in self.parameters.items():
                    params[name] = param.suggest_value()
                suggestions.append(params)
            except Exception as e:
                print(f"Error during random suggestion: {e}")
        
        # If we have a Gaussian Process model with enough data, use it for better suggestions
        if len(self.experiments) >= self.min_points_in_model and n_random == 0:
            try:
                # Try to generate more candidates and select the best subset using acquisition function
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern
                
                # Extract normalized features and targets
                X, y = self._extract_normalized_features_and_targets()
                if len(X) > 0:
                    # Define and fit Gaussian Process
                    kernel = Matern(nu=2.5)
                    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
                    gp.fit(X, y)
                    
                    # Generate additional candidates
                    candidate_suggestions = suggestions.copy()  # Keep Optuna suggestions
                    n_extra_candidates = min(n_suggestions * 10, 100)  # Generate 10x or up to 100 candidates
                    
                    # Generate random candidates for evaluation
                    for _ in range(n_extra_candidates):
                        params = {}
                        for name, param in self.parameters.items():
                            params[name] = param.suggest_value()
                        candidate_suggestions.append(params)
                    
                    # Evaluate all candidates
                    evaluated_candidates = []
                    best_value = max([exp.get('score', 0) for exp in self.experiments]) if self.experiments else 0
                    
                    for params in candidate_suggestions:
                        features = self._normalize_params(params)
                        X_test = np.array(features).reshape(1, -1)
                        mean, std = gp.predict(X_test, return_std=True)
                        mean, std = float(mean[0]), float(std[0])
                        
                        acq_value = self.evaluate_acquisition(mean, std, best_value)
                        
                        evaluated_candidates.append({
                            'params': params,
                            'mean': mean,
                            'std': std,
                            'acquisition_value': acq_value
                        })
                    
                    # Apply Thompson sampling for diversity
                    evaluated_candidates = self._apply_thompson_sampling(evaluated_candidates)
                    
                    # Select top candidates
                    top_candidates = evaluated_candidates[:n_suggestions]
                    
                    # Replace suggestions with top candidates
                    suggestions = [candidate['params'] for candidate in top_candidates]
                
            except Exception as e:
                print(f"Error during advanced suggestion generation: {e}")
                # Keep original suggestions if advanced method fails
                
        # Ensure we have exactly n_suggestions
        if len(suggestions) > n_suggestions:
            suggestions = suggestions[:n_suggestions]
        elif len(suggestions) < n_suggestions:
            # Fill with random suggestions if needed
            for _ in range(n_suggestions - len(suggestions)):
                params = {}
                for name, param in self.parameters.items():
                    params[name] = param.suggest_value()
                suggestions.append(params)
                
        return suggestions
    
    def _normalize_params(self, params):
        """Normalize parameters to [0,1] range for the Gaussian Process"""
        features = []
        
        for name, param in self.parameters.items():
            if name in params:
                value = params[name]
                
                if param.param_type == "categorical":
                    # One-hot encoding
                    for i, choice in enumerate(param.choices):
                        features.append(1.0 if value == choice else 0.0)
                elif param.param_type in ["continuous", "discrete"]:
                    # Normalize to [0,1]
                    norm_value = (float(value) - param.low) / (param.high - param.low)
                    features.append(norm_value)
                    
        return features
    
    def _extract_normalized_features_and_targets(self):
        """Extract normalized features and targets from experiments"""
        if not self.experiments:
            return np.array([]), np.array([])
            
        X = []
        y = []
        
        for exp in self.experiments:
            if 'params' in exp and 'score' in exp:
                features = self._normalize_params(exp['params'])
                X.append(features)
                y.append(exp['score'])
                
        return np.array(X), np.array(y)
        
    def get_best_experiments(self, n=5) -> List[Dict[str, Any]]:
        """Get the best experiments sorted by score"""
        if not self.experiments:
            return []
            
        scored_experiments = sorted(self.experiments, key=lambda x: x.get('score', 0.0), reverse=True)
        
        return scored_experiments[:n]
        
    def save_model(self, filepath):
        """Save model state to file"""
        trials_data = []
        
        if self.study and hasattr(self.study, 'trials'):
            trials_data = [
                {
                    "params": t.params,
                    "value": t.value,
                    "state": t.state.name,
                }
                for t in self.study.trials
            ]

        with open(filepath, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'categorical_mappings': self.categorical_mappings,
                'objectives': self.objectives,
                'objective_weights': self.objective_weights,
                'experiments': self.experiments,
                'planned_experiments': self.planned_experiments,
                'optuna_trials': trials_data,
                'acquisition_function': self.acquisition_function,
                'exploitation_weight': self.exploitation_weight,
                'use_thompson_sampling': self.use_thompson_sampling,
                'min_points_in_model': self.min_points_in_model,
                'exploration_noise': self.exploration_noise
            }, f)
            
    def load_model(self, filepath):
        """Load model state from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.parameters = data['parameters']
            self.categorical_mappings = data['categorical_mappings']
            self.objectives = data['objectives']
            self.objective_weights = data['objective_weights']
            self.experiments = data.get('experiments', [])
            self.planned_experiments = data.get('planned_experiments', [])
            optuna_trials_data = data.get('optuna_trials', [])
            
            # Load advanced settings if they exist
            self.acquisition_function = data.get('acquisition_function', self.acquisition_function)
            self.exploitation_weight = data.get('exploitation_weight', self.exploitation_weight)
            self.use_thompson_sampling = data.get('use_thompson_sampling', self.use_thompson_sampling)
            self.min_points_in_model = data.get('min_points_in_model', self.min_points_in_model)
            self.exploration_noise = data.get('exploration_noise', self.exploration_noise)

        self.create_study()

        try:
            distributions = self._get_distributions()
            for trial_data in optuna_trials_data:
                state = optuna.trial.TrialState.COMPLETE
                try:
                    state = optuna.trial.TrialState[trial_data.get('state', 'COMPLETE')]
                except KeyError:
                    pass

                if state != optuna.trial.TrialState.COMPLETE:
                     continue

                value = trial_data.get('value')
                if value is None:
                    found_exp = next((exp for exp in self.experiments if exp['params'] == trial_data['params']), None)
                    if found_exp and 'score' in found_exp:
                        value = found_exp['score']
                    else:
                        print(f"Warning: Skipping trial with missing value: {trial_data['params']}")
                        continue

                trial = optuna.trial.create_trial(
                    params=trial_data['params'],
                    distributions=distributions,
                    value=value,
                    state=state,
                )
                try:
                     self.study.add_trial(trial)
                except Exception as e:
                     print(f"Warning: Could not reload trial into study: {e}")
        except Exception as e:
            print(f"Warning: Error rebuilding Optuna study: {e}")
            # Continue with a fresh study
            self.create_study()

    def analyze_parameter_importance(self) -> Dict[str, float]:
        if not self.study or len(self.study.trials) < 2:
            print("Warning: Need at least 2 completed trials for importance analysis.")
            return {name: 0.0 for name in self.parameters}

        try:
            param_importance = optuna.importance.get_param_importances(self.study)

            max_importance = max(param_importance.values()) if param_importance else 1.0
            if max_importance > 0:
                 normalized_importance = {k: v / max_importance for k, v in param_importance.items()}
            else:
                 normalized_importance = param_importance

            final_importance = {name: normalized_importance.get(name, 0.0) for name in self.parameters}
            return final_importance

        except Exception as e:
            print(f"Error calculating parameter importance: {e}")
            return {name: 0.0 for name in self.parameters}

    def calculate_experiment_distance(self, params1, params2):
        """Calculate normalized distance between two sets of parameter values."""
        distance = 0.0
        count = 0
        
        for param_name, param in self.parameters.items():
            if param_name in params1 and param_name in params2:
                if param.param_type == "categorical":
                    # Binary distance for categorical parameters
                    distance += 0.0 if params1[param_name] == params2[param_name] else 1.0
                else:
                    # Normalized distance for numeric parameters
                    range_width = param.high - param.low
                    if range_width > 0:
                        norm_val1 = (float(params1[param_name]) - param.low) / range_width
                        norm_val2 = (float(params2[param_name]) - param.low) / range_width
                        distance += (norm_val1 - norm_val2) ** 2
                count += 1
        
        if count == 0:
            return float('inf')
        
        return np.sqrt(distance / count)

    def select_diverse_subset(self, candidates, n, diversity_weight=0.7):
        """Select a diverse subset of candidate experiments."""
        if len(candidates) <= n:
            return candidates
        
        selected = [candidates[0]]  # Start with first candidate
        candidates = candidates[1:]
        
        while len(selected) < n and candidates:
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in candidates:
                # Calculate diversity as minimum distance to already selected points
                min_distance = min(self.calculate_experiment_distance(candidate['params'], s['params']) for s in selected)
                
                # Calculate utility based on predicted score
                utility = candidate.get('predicted_score', 0.5)
                
                # Combined score with diversity weight
                score = (1 - diversity_weight) * utility + diversity_weight * min_distance
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
            
        return selected