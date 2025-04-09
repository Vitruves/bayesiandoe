import numpy as np
import datetime
import pickle
import optuna
from typing import Dict, List, Any, Optional
from optuna.samplers import BaseSampler, TPESampler
from scipy import stats
import random

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
        self.objectives = []
        self.objective_weights = {}
        self.objective_directions = {}
        self.experiments = []
        self.planned_experiments = []
        self.study = None
        self.acquisition_function = "ei"  # Default to Expected Improvement
        self.exploitation_weight = 0.5    # Balance between exploration/exploitation
        self.min_points_in_model = 3      # Minimum experiments before using model
        self.design_method = "TPE"        # Default design method
        
        # Model storage
        self.model_cache = {}
        self.best_candidate = None
        
        # Parameter importance analysis
        self.parameter_importance = {}
        
        # Logging
        import logging
        self.log = logging.getLogger("BayesianDOE")
        
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
        
        # Recreate study if it exists
        if self.study:
            self.create_study()
            
    def remove_parameter(self, name):
        if name in self.parameters:
            del self.parameters[name]
            if self.study:
                self.create_study()

    def create_study(self, sampler: Optional[BaseSampler] = None):
        if not sampler:
            # Use TPE sampler by default
            sampler = TPESampler(seed=42)
        
        self.sampler = sampler  # Store the sampler

        def dummy_objective(trial: optuna.Trial) -> float:
            return 0.0

        self.study = optuna.create_study(
            study_name="chemical_optimization",
            direction="maximize",
            sampler=sampler,
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
        """Add experimental result to the model"""
        # Calculate the composite score from all objectives
        score = self._calculate_composite_score(results)
        
        # Create a timestamp for the experiment
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Create a clean copy of parameters, removing any non-parameter keys
        clean_params = {}
        for k, v in params.items():
            if k in self.parameters:
                clean_params[k] = v
        
        # Store all data including results and score
        experiment_data = {
            'params': clean_params,
            'results': results,
            'score': score,
            'timestamp': timestamp
        }
        
        # Add to experiments list
        self.experiments.append(experiment_data)
        
        # We don't need to add to Optuna study for now, as it causes issues
        # This should fix the inconsistent parameters warning

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

    def acquisition_function_ei(self, mean, std, best_value, xi=0.01):
        """Expected Improvement acquisition function.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_value: Best observed value so far
            xi: Exploration parameter
            
        Returns:
            Expected improvement value
        """
        import scipy.stats as ss
        
        if std <= 0.0:
            return 0.0
        
        z = (mean - best_value - xi) / std
        ei = (mean - best_value - xi) * ss.norm.cdf(z) + std * ss.norm.pdf(z)
        return max(ei, 0.0)
    
    def acquisition_function_pi(self, mean, std, best_value, xi=0.01):
        """Probability of Improvement acquisition function.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_value: Best observed value so far
            xi: Exploration parameter
            
        Returns:
            Probability of improvement value
        """
        import scipy.stats as ss
        
        if std <= 0.0:
            return 0.0
        
        z = (mean - best_value - xi) / std
        return ss.norm.cdf(z)
        
    def acquisition_function_ucb(self, mean, std, best_value, beta=2.0):
        """Upper Confidence Bound acquisition function.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_value: Best observed value so far (not used)
            beta: Exploration parameter
            
        Returns:
            UCB value
        """
        return mean + beta * std
        
    def evaluate_acquisition(self, mean, std, best_value):
        """Evaluate the acquisition function based on current settings.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            best_value: Best observed value so far
            
        Returns:
            Acquisition function value
        """
        acq_func = self.acquisition_function
        
        # Scale the exploration weight based on slider
        exploration_factor = 1.0 - self.exploitation_weight
        
        if acq_func == "ei":
            return self.acquisition_function_ei(mean, std, best_value, xi=0.01*exploration_factor)
        elif acq_func == "pi":
            return self.acquisition_function_pi(mean, std, best_value, xi=0.01*exploration_factor)
        elif acq_func == "ucb":
            return self.acquisition_function_ucb(mean, std, best_value, beta=exploration_factor*3.0)
        else:
            # Default to EI
            return self.acquisition_function_ei(mean, std, best_value, xi=0.01*exploration_factor)
            
    def _apply_thompson_sampling(self, candidates):
        """Apply Thompson sampling to select candidates, with fixed key naming."""
        import numpy as np
        from scipy.special import softmax
        
        # Sort candidates by acquisition value (using the new key name)
        candidates = sorted(candidates, key=lambda x: x.get('acquisition_value', x.get('acq_value', 0.0)), reverse=True)
        
        # Apply softmax to acquisition values for probability distribution
        acq_values = np.array([c.get('acquisition_value', c.get('acq_value', 0.0)) for c in candidates])
        
        # Handle potential instability in softmax
        acq_values = acq_values - np.max(acq_values)  # For numerical stability
        
        # If all values are the same or very close
        if np.std(acq_values) < 1e-10:
            probs = np.ones_like(acq_values) / len(acq_values)
        else:
            # Apply temperature parameter (lower = more exploitation)
            temperature = max(0.1, self.exploitation_weight)
            probs = softmax(acq_values / temperature)
        
        # Get cumulative probabilities
        cumprobs = np.cumsum(probs)
        
        # Sample based on probabilities
        selected = []
        selected_indices = set()
        
        while len(selected) < min(len(candidates), len(candidates) // 3 + 2):
            r = np.random.random()
            for i, cp in enumerate(cumprobs):
                if r <= cp and i not in selected_indices:
                    selected.append(candidates[i])
                    selected_indices.add(i)
                    break
                
        # Always include top candidate for pure exploitation
        if 0 not in selected_indices and candidates:
            selected.insert(0, candidates[0])
        
        return selected

    def _normalize_params(self, params):
        """Convert parameter values to normalized feature vector for machine learning models"""
        features = []
        
        for name, param in self.parameters.items():
            if name in params:
                value = params[name]
                
                if param.param_type == "continuous" or param.param_type == "discrete":
                    # Normalize numeric parameters to [0, 1]
                    if param.high > param.low:
                        norm_value = (float(value) - param.low) / (param.high - param.low)
                        features.append(norm_value)
                    else:
                        features.append(0.5)  # Default if range is invalid
                elif param.param_type == "categorical":
                    # One-hot encode categorical parameters
                    choices = param.choices
                    for choice in choices:
                        features.append(1.0 if value == choice else 0.0)
                else:
                    features.append(0.0)  # Default for unknown types
            else:
                # Add default features if parameter is missing
                if param.param_type == "continuous" or param.param_type == "discrete":
                    features.append(0.5)  # Default to middle of range
                elif param.param_type == "categorical":
                    # One-hot encode with all zeros
                    features.extend([0.0] * len(param.choices))
        
        return features
    
    def _extract_normalized_features_and_targets(self):
        """Extract normalized features and target values for model training"""
        if not self.experiments:
            return [], []
            
        X = []
        y = []
        
        for exp in self.experiments:
            if 'params' in exp:
                # Normalize parameters
                features = self._normalize_params(exp['params'])
                X.append(features)
                
                # Calculate composite score as target
                score = 0.0
                weight_sum = 0.0
                
                for obj in self.objectives:
                    if obj in exp.get('results', {}):
                        weight = self.objective_weights.get(obj, 1.0)
                        score += exp['results'][obj] * weight
                        weight_sum += weight
                
                if weight_sum > 0:
                    score = score / weight_sum
                    
                y.append(score)
                
        return np.array(X), np.array(y)
    
    def _get_experiment_analysis(self):
        """Analyze current experiment set to guide future suggestions"""
        if not self.experiments or len(self.experiments) < 3:
            return {
                'best_score': 0.0,
                'parameter_importance': {},
                'exploration_recommended': True,
                'convergence_status': 'not_started',
                'estimated_rounds_to_convergence': float('inf')
            }
            
        # Calculate the best score so far
        best_score = 0.0
        for exp in self.experiments:
            score = 0.0
            weight_sum = 0.0
            
            for obj in self.objectives:
                if obj in exp.get('results', {}):
                    weight = self.objective_weights.get(obj, 1.0)
                    score += exp['results'][obj] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                score = score / weight_sum
                best_score = max(best_score, score)
        
        # Calculate parameter importance
        try:
            param_importance = self.analyze_parameter_importance()
        except:
            param_importance = {name: 1.0/len(self.parameters) for name in self.parameters}
            
        # Analyze convergence
        scores = []
        for exp in self.experiments:
            score = 0.0
            weight_sum = 0.0
            
            for obj in self.objectives:
                if obj in exp.get('results', {}):
                    weight = self.objective_weights.get(obj, 1.0)
                    score += exp['results'][obj] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                score = score / weight_sum
                
            scores.append(score)
            
        best_scores = np.maximum.accumulate(scores)
        
        # Check if scores are improving
        improvement_threshold = 0.01  # 1% improvement
        
        # Check last few experiments for improvement
        check_window = min(5, len(best_scores) // 3)
        if check_window > 1:
            recent_improvement = best_scores[-1] - best_scores[-check_window]
            relative_improvement = recent_improvement / (best_scores[-check_window] + 1e-10)
            
            if relative_improvement < improvement_threshold:
                convergence_status = 'converging' 
                exploration_recommended = True
            else:
                convergence_status = 'improving'
                exploration_recommended = False
        else:
            convergence_status = 'started'
            exploration_recommended = True
            
        # Estimate rounds to convergence using convergence model
        try:
            from scipy.optimize import curve_fit
            
            def convergence_model(x, a, b, c):
                return a * (1 - np.exp(-b * x)) + c
                
            x = np.array(range(1, len(best_scores) + 1))
            
            if len(x) >= 3:  # Need at least 3 points for fitting
                popt, _ = curve_fit(
                    convergence_model, x, best_scores, 
                    p0=[0.5, 0.1, best_scores[0]],
                    bounds=([0, 0.001, 0], [1, 1, 1]),
                    maxfev=10000
                )
                
                a, b, c = popt
                
                # Calculate target (95% of asymptotic value)
                asymptotic_value = a + c
                target = 0.95 * asymptotic_value
                
                # Calculate required iterations
                if a > 0 and target > best_scores[-1]:
                    from math import log
                    required_x = -log(1 - (target - c) / a) / b
                    remaining_rounds = max(0, int(np.ceil(required_x - len(best_scores))))
                else:
                    remaining_rounds = 0
                    
                estimated_rounds = remaining_rounds
            else:
                estimated_rounds = float('inf')
        except:
            estimated_rounds = float('inf')
            
        return {
            'best_score': best_score,
            'parameter_importance': param_importance,
            'exploration_recommended': exploration_recommended,
            'convergence_status': convergence_status,
            'estimated_rounds_to_convergence': estimated_rounds
        }
        
    def suggest_experiments(self, n_suggestions=5) -> List[Dict[str, Any]]:
        """Generate suggested experiments based on existing data"""
        print(f"Suggesting {n_suggestions} new experiments using {self.acquisition_function}")
        
        if len(self.experiments) < self.min_points_in_model:
            # Not enough data for modeling, use space-filling design
            return self._suggest_with_sobol(n_suggestions)
            
        # Default to TPE if we don't have a specific implementation
        method = "tpe"
        
        try:
            # If we have enough experiments, use the chosen method
            if method == "tpe":
                suggestions = self._suggest_with_tpe(n_suggestions)
            elif method == "gpei":
                suggestions = self._suggest_with_gp(n_suggestions)
            elif method == "random":
                suggestions = self._suggest_random(n_suggestions)
            elif method == "lhs":
                suggestions = self._suggest_with_lhs(n_suggestions)
            elif method == "sobol":
                suggestions = self._suggest_with_sobol(n_suggestions)
            else:
                # Default to TPE
                suggestions = self._suggest_with_tpe(n_suggestions)
            
            # Return suggestions
            return suggestions
        
        except Exception as e:
            print(f"Error in suggest_experiments: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to random if modeling fails
            return self._suggest_random(n_suggestions)
    
    def _suggest_with_prior_and_space_filling(self, n_suggestions):
        """Generate suggestions using prior knowledge and space-filling designs"""
        has_prior = any(p.prior_mean is not None for p in self.parameters.values()
                       if p.param_type in ["continuous", "discrete"])
        
        if has_prior:
            # Generate some suggestions based on prior knowledge
            prior_based = []
            for _ in range(n_suggestions // 2 + 1):
                params = {}
                for name, param in self.parameters.items():
                    if param.param_type in ["continuous", "discrete"] and param.prior_mean is not None:
                        # Sample from a distribution around the prior mean
                        if param.param_type == "continuous":
                            value = random.normalvariate(param.prior_mean, param.prior_std / 2)
                            value = max(param.low, min(param.high, value))
                        else:  # discrete
                            value = int(round(random.normalvariate(param.prior_mean, param.prior_std / 2)))
                            value = max(int(param.low), min(int(param.high), value))
                    else:
                        value = param.suggest_value()
                    params[name] = value
                prior_based.append(params)
            
            # Fill remaining with space-filling design
            remaining = n_suggestions - len(prior_based)
            space_filling = self._suggest_with_sobol(remaining)
            
            return prior_based + space_filling
        else:
            # No prior knowledge, use space-filling design
            return self._suggest_with_sobol(n_suggestions)

    def _train_surrogate_model(self, X, y, model_type='rf'):
        """Train a surrogate model for Bayesian optimization"""
        try:
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=100, 
                    min_samples_leaf=3,
                    random_state=42
                )
            elif model_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, ConstantKernel
                
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.05,
                    normalize_y=True,
                    random_state=42
                )
            else:
                # Default to random forest
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=100, 
                    min_samples_leaf=3,
                    random_state=42
                )
                
            model.fit(X, y)
            return model
        except Exception as e:
            self.log.warning(f"Error training surrogate model: {e}")
            return None

    def _predict_candidate(self, params, X, y, model_type='rf'):
        """Predict mean and std for a candidate using the surrogate model"""
        # Normalize the parameters
        x = self._normalize_params(params)
        
        # Make prediction with uncertainty
        try:
            if model_type == 'gp':
                # GP provides uncertainty directly
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, ConstantKernel
                
                if not hasattr(self, '_gp_model') or self._gp_model is None:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                    self._gp_model = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=0.05,
                        normalize_y=True,
                        random_state=42
                    )
                    self._gp_model.fit(X, y)
                
                mean, std = self._gp_model.predict([x], return_std=True)
                return float(mean[0]), float(std[0])
                
            else:
                # For RF, use variance of tree predictions as uncertainty
                from sklearn.ensemble import RandomForestRegressor
                
                if not hasattr(self, '_rf_model') or self._rf_model is None:
                    self._rf_model = RandomForestRegressor(
                        n_estimators=100, 
                        min_samples_leaf=3,
                        random_state=42
                    )
                    self._rf_model.fit(X, y)
                
                # Predict with each tree
                preds = []
                for tree in self._rf_model.estimators_:
                    preds.append(tree.predict([x])[0])
                
                mean = np.mean(preds)
                std = np.std(preds)
                
                return float(mean), float(std)
        except Exception as e:
            self.log.warning(f"Error in prediction: {e}")
            return 0.5, 0.5  # Default values
        
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
                'objectives': self.objectives,
                'objective_weights': self.objective_weights,
                'objective_directions': self.objective_directions,
                'experiments': self.experiments,
                'study': self.study,
                'acquisition_function': self.acquisition_function,
                'exploitation_weight': self.exploitation_weight,
                'min_points_in_model': self.min_points_in_model,
                'design_method': self.design_method,
                'model_cache': self.model_cache,
                'best_candidate': self.best_candidate,
                'parameter_importance': self.parameter_importance,
                'optuna_trials': trials_data
            }, f)
            
    def load_model(self, filepath):
        """Load model state from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.parameters = data['parameters']
            self.objectives = data['objectives']
            self.objective_weights = data['objective_weights']
            self.objective_directions = data['objective_directions']
            self.experiments = data.get('experiments', [])
            self.study = data.get('study')
            self.acquisition_function = data.get('acquisition_function', self.acquisition_function)
            self.exploitation_weight = data.get('exploitation_weight', self.exploitation_weight)
            self.min_points_in_model = data.get('min_points_in_model', self.min_points_in_model)
            self.design_method = data.get('design_method', self.design_method)
            self.model_cache = data.get('model_cache', {})
            self.best_candidate = data.get('best_candidate')
            self.parameter_importance = data.get('parameter_importance', {})
            optuna_trials_data = data.get('optuna_trials', [])

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
        """Select a diverse subset of candidates to ensure exploration.
        
        Args:
            candidates: List of parameter dictionaries
            n: Number of candidates to select
            diversity_weight: Weight of diversity vs. predicted performance
            
        Returns:
            List of selected diverse parameter sets
        """
        if len(candidates) <= n:
            return candidates
        
        # First, always include the candidate with highest predicted performance
        if hasattr(self, 'best_candidate') and self.best_candidate is not None:
            selected = [self.best_candidate]
            candidates = [c for c in candidates if c != self.best_candidate]
        else:
            selected = [candidates[0]]
        candidates = candidates[1:]
        
        # Then add remaining candidates based on diversity
        while len(selected) < n and candidates:
            max_min_dist = -1
            best_candidate = None
            best_idx = -1
            
            # For each remaining candidate
            for i, candidate in enumerate(candidates):
                # Calculate minimum distance to already selected candidates
                min_dist = float('inf')
                for sel in selected:
                    dist = self.calculate_experiment_distance(candidate, sel)
                    min_dist = min(min_dist, dist)
                
                # Apply diversity weighting
                score = min_dist * diversity_weight + (1 - diversity_weight) * (1.0 / (i + 1))
                
                if score > max_min_dist:
                    max_min_dist = score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.pop(best_idx)
            else:
                break
            
        return selected

    def _suggest_with_gp(self, n_suggestions):
        """Generate suggestions using Gaussian Process model with improved hyperparameters"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
            import numpy as np
            
            # Extract features and targets for model training
            X, y = self._extract_normalized_features_and_targets()
            
            if len(X) < 3 or len(np.unique(y)) < 2:
                # Not enough data for good GP fitting
                return self._suggest_with_tpe(n_suggestions)
            
            # Use wider bounds for length_scale to avoid convergence warnings
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=np.ones(X.shape[1]),
                length_scale_bounds=(1e-6, 1e6),
                nu=2.5
            )
            
            # Add noise term to handle potential instability
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.01,  # Increased noise level for stability
                normalize_y=True,
                n_restarts_optimizer=5
            )
            
            # Train the GP
            gp.fit(X, y)
            
            # Generate and evaluate candidates
            n_candidates = n_suggestions * 10
            candidates = []
            
            # Get best value so far for acquisition function
            best_value = max(y) if len(y) > 0 else 0.0
            
            for _ in range(n_candidates):
                # Generate random parameter values
                params = {}
                for name, param in self.parameters.items():
                    params[name] = param.suggest_value()
                
                # Normalize the parameters
                x = self._normalize_params(params)
                
                # Predict with GP
                try:
                    mean, std = gp.predict([x], return_std=True)
                    mean, std = float(mean[0]), float(std[0])
                    
                    # Use acquisition function for evaluation
                    acq_value = self.evaluate_acquisition(mean, std, best_value)
                    
                    candidates.append({
                        'params': params,
                        'mean': mean,
                        'std': std,
                        'acquisition_value': acq_value  # Renamed from acq_value to avoid key error
                    })
                except Exception as e:
                    self.log.warning(f"GP prediction error: {e}")
                    # If prediction fails, just add with default scores
                    candidates.append({
                        'params': params,
                        'mean': 0.5,
                        'std': 0.5,
                        'acquisition_value': 0.0
                    })
            
            # Sort by acquisition value and select top candidates
            candidates.sort(key=lambda x: x['acquisition_value'], reverse=True)
            return [c['params'] for c in candidates[:n_suggestions]]
        
        except Exception as e:
            self.log.warning(f"GP sampling failed: {e}")
            # Fall back to TPE if GP fails
            return self._suggest_with_tpe(n_suggestions)

    def _suggest_with_tpe(self, n_suggestions):
        """Generate suggestions using Tree-structured Parzen Estimator (TPE)"""
        try:
            import optuna
            from optuna.samplers import TPESampler
            
            # Create a new study with TPE sampler
            sampler = TPESampler(seed=42)
            study_name = "chemical_optimization"
            study = optuna.create_study(sampler=sampler, direction="maximize", study_name=study_name)
            
            # Define the objective function to sample the parameter space
            def dummy_objective(trial):
                params = {}
                for name, param in self.parameters.items():
                    if param.param_type == "continuous":
                        params[name] = trial.suggest_float(name, param.low, param.high)
                    elif param.param_type == "discrete":
                        params[name] = trial.suggest_int(name, int(param.low), int(param.high))
                    elif param.param_type == "categorical":
                        params[name] = trial.suggest_categorical(name, param.choices)
                return 0  # Dummy value
            
            # Generate n_suggestions parameter sets
            candidates = []
            for _ in range(n_suggestions):
                try:
                    study.optimize(dummy_objective, n_trials=1)
                    # Get the last trial and extract parameters safely
                    trial = study.trials[-1]
                    params = {}
                    for name in self.parameters.keys():
                        # Check if parameter is in trial params
                        if name in trial.params:
                            params[name] = trial.params[name]
                        else:
                            # If missing, generate a default value
                            params[name] = self.parameters[name].suggest_value()
                    candidates.append(params)
                except Exception as e:
                    print(f"Error during TPE optimization: {e}")
                    # Generate fallback parameters for this suggestion
                    params = {}
                    for name, param in self.parameters.items():
                        params[name] = param.suggest_value()
                    candidates.append(params)
            
            return candidates
        except Exception as e:
            print(f"TPE sampling failed: {e}")
            # Fallback to random sampling if TPE fails
            return self._suggest_random(n_suggestions)

    def _suggest_with_sobol(self, n_suggestions):
        """Generate suggestions using Sobol sequence (space-filling)"""
        try:
            from scipy.stats import qmc
            import numpy as np
            
            # Create Sobol sequence generator
            sampler = qmc.Sobol(d=len(self.parameters), scramble=True)
            
            # Generate samples in [0, 1]^d space
            sample = sampler.random(n=n_suggestions)
            
            # Transform to actual parameter ranges
            candidates = []
            for s in sample:
                params = {}
                for i, (name, param) in enumerate(self.parameters.items()):
                    if param.param_type == "continuous":
                        value = param.low + s[i] * (param.high - param.low)
                    elif param.param_type == "discrete":
                        value = int(round(param.low + s[i] * (param.high - param.low)))
                    else:  # categorical
                        idx = int(s[i] * len(param.choices))
                        if idx >= len(param.choices):
                            idx = len(param.choices) - 1
                        value = param.choices[idx]
                    params[name] = value
                candidates.append(params)
            
            return candidates
        
        except ImportError:
            # Fall back to random if scipy is not available
            self.log.warning("scipy.stats.qmc not available, falling back to random")
            return self._suggest_random(n_suggestions)

    def _suggest_with_lhs(self, n_suggestions):
        """Generate suggestions using Latin Hypercube Sampling"""
        try:
            from scipy.stats import qmc
            import numpy as np
            
            # Create Latin hypercube sampler
            sampler = qmc.LatinHypercube(d=len(self.parameters))
            
            # Generate samples in [0, 1]^d space
            sample = sampler.random(n=n_suggestions)
            
            # Transform to actual parameter ranges
            candidates = []
            for s in sample:
                params = {}
                for i, (name, param) in enumerate(self.parameters.items()):
                    if param.param_type == "continuous":
                        value = param.low + s[i] * (param.high - param.low)
                    elif param.param_type == "discrete":
                        value = int(round(param.low + s[i] * (param.high - param.low)))
                    else:  # categorical
                        idx = int(s[i] * len(param.choices))
                        if idx >= len(param.choices):
                            idx = len(param.choices) - 1
                        value = param.choices[idx]
                    params[name] = value
                candidates.append(params)
            
            return candidates
        
        except ImportError:
            # Fall back to random if scipy is not available
            self.log.warning("scipy.stats.qmc not available, falling back to random")
            return self._suggest_random(n_suggestions)

    def _suggest_random(self, n_suggestions):
        """Generate suggestions using pure random sampling"""
        candidates = []
        for _ in range(n_suggestions):
            params = {}
            for name, param in self.parameters.items():
                params[name] = param.suggest_value()
            candidates.append(params)
        return candidates

    def set_objectives(self, objectives):
        """Set optimization objectives with weights.
        
        Args:
            objectives: Dictionary with objective names as keys and weights as values
        """
        self.objectives = list(objectives.keys())
        self.objective_weights = objectives