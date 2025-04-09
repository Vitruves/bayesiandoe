import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PySide6.QtWidgets import QMessageBox
from ..visualizations import (
    plot_optimization_history, plot_parameter_importance, 
    plot_parameter_contour, plot_objective_correlation,
    plot_convergence, plot_response_surface
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import QApplication

def update_prior_plot(self):
    param_name = self.viz_param_combo.currentText()
    if not param_name or param_name not in self.model.parameters:
        self.prior_canvas.axes.clear()
        self.prior_canvas.axes.text(0.5, 0.5, "No parameter selected", 
            ha='center', va='center', transform=self.prior_canvas.axes.transAxes)
        self.prior_canvas.draw()
        return
        
    param = self.model.parameters[param_name]
    
    self.prior_canvas.axes.clear()
    
    if param.param_type in ["continuous", "discrete"]:
        if param.prior_mean is not None and param.prior_std is not None:
            if param.param_type == "continuous":
                x = np.linspace(max(param.low - 2*param.prior_std, param.low*0.8),
                              min(param.high + 2*param.prior_std, param.high*1.2), 1000)
            else:
                x = np.arange(param.low, param.high + 1)
                
            pdf = stats.norm.pdf(x, loc=param.prior_mean, scale=param.prior_std)
            
            self.prior_canvas.axes.plot(x, pdf, 'b-', linewidth=2)
            self.prior_canvas.axes.fill_between(x, pdf, color='blue', alpha=0.2)
            
            self.prior_canvas.axes.axvline(param.prior_mean, color='r', linestyle='-', alpha=0.7)
            self.prior_canvas.axes.axvline(param.prior_mean - param.prior_std, color='g', linestyle='--', alpha=0.7)
            self.prior_canvas.axes.axvline(param.prior_mean + param.prior_std, color='g', linestyle='--', alpha=0.7)
            
            self.prior_canvas.axes.axvline(param.low, color='k', linestyle=':', alpha=0.5)
            self.prior_canvas.axes.axvline(param.high, color='k', linestyle=':', alpha=0.5)
            
            self.prior_canvas.axes.set_xlabel(f"{param_name} {f'({param.units})' if param.units else ''}")
            self.prior_canvas.axes.set_ylabel('Probability Density')
            self.prior_canvas.axes.set_title(f"Prior Distribution for {param_name}")
        else:
            self.prior_canvas.axes.text(0.5, 0.5, "No prior defined for this parameter", 
                ha='center', va='center', transform=self.prior_canvas.axes.transAxes)
            
    else:
        if hasattr(param, 'categorical_preferences') and param.categorical_preferences:
            categories = []
            values = []
            
            for choice in param.choices:
                categories.append(choice)
                values.append(param.categorical_preferences.get(choice, 5))
                
            x = range(len(categories))
            self.prior_canvas.axes.bar(x, values, tick_label=categories)
            self.prior_canvas.axes.set_ylim(0, 11)
            self.prior_canvas.axes.set_ylabel('Preference Weight')
            self.prior_canvas.axes.set_title(f"Category Preferences for {param_name}")
            
            if len(categories) > 5:
                plt.setp(self.prior_canvas.axes.get_xticklabels(), rotation=45, ha="right")
        else:
            self.prior_canvas.axes.text(0.5, 0.5, "No preferences defined for this parameter", 
                ha='center', va='center', transform=self.prior_canvas.axes.transAxes)
    
    self.prior_canvas.draw()

def update_results_plot(self):
    plot_type = self.plot_type_combo.currentText()
    
    if not self.model.experiments:
        self.result_canvas.axes.clear()
        self.result_canvas.axes.text(0.5, 0.5, "No experiment results yet", 
            ha='center', va='center', transform=self.result_canvas.axes.transAxes)
        self.result_canvas.draw()
        return
        
    if plot_type == "Optimization History":
        self.result_canvas.axes = plot_optimization_history(self.model, self.result_canvas.axes)
        
    elif plot_type == "Parameter Importance":
        self.result_canvas.axes = plot_parameter_importance(self.model, self.result_canvas.axes)
        
    elif plot_type == "Parameter Contour":
        x_param = self.x_param_combo.currentText()
        y_param = self.y_param_combo.currentText()
        
        if not x_param or not y_param or x_param == y_param:
            self.result_canvas.axes.clear()
            self.result_canvas.axes.text(0.5, 0.5, "Select different X and Y parameters", 
                ha='center', va='center', transform=self.result_canvas.axes.transAxes)
        else:
            self.result_canvas.axes = plot_parameter_contour(
                self.model, x_param, y_param, self.result_canvas.axes
            )
            
    elif plot_type == "Objective Correlation":
        self.result_canvas.axes = plot_objective_correlation(self.model, self.result_canvas.axes)
    
    self.result_canvas.draw()

def update_model_plot(self):
    """Update model prediction plot with improved error handling."""
    objective = self.model_obj_combo.currentText()
    
    try:
        # Check basic prerequisites
        if not objective:
            self.model_canvas.axes.clear()
            self.model_canvas.axes.text(0.5, 0.5, "No objective selected", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            return
            
        # Ensure we have enough experiments
        valid_experiments = []
        for exp in self.model.experiments:
            if 'results' in exp and objective in exp['results'] and exp['results'][objective] is not None:
                valid_experiments.append(exp)
                
        if len(valid_experiments) < 3:
            self.model_canvas.axes.clear()
            self.model_canvas.axes.text(0.5, 0.5, f"Need at least 3 experiments with {objective} results", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            self.log(f"-- Model plot not updated: insufficient data for {objective}")
            return
            
        # Clear plot
        self.model_canvas.axes.clear()
        
        # Extract data safely
        X = []
        y = []
        
        for exp in valid_experiments:
            try:
                # Extract features for this experiment
                features = self.model._normalize_params(exp['params'])
                if features and len(features) > 0:
                    X.append(features)
                    y.append(exp['results'][objective] * 100.0)
            except Exception as e:
                print(f"Skipping experiment due to feature extraction error: {e}")
        
        # Check we have usable data
        if not X or len(X) < 3:
            self.model_canvas.axes.text(0.5, 0.5, "Failed to extract valid features from experiments", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            self.log("-- Model plot not updated: feature extraction failed")
            return
            
        # Ensure consistency
        X = np.array(X)
        y = np.array(y)
        
        # Check for data variation 
        if len(np.unique(y)) < 2:
            self.model_canvas.axes.text(0.5, 0.5, f"Insufficient variation in {objective} values", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            self.log("-- Model plot not updated: insufficient result variation")
            return
            
        # Fit GP model
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            
            # Try to use GP but fall back gracefully
            try:
                # More stable GP configuration
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, 
                                            alpha=0.1, normalize_y=True)
                gp.fit(X, y)
                
                # Make predictions
                y_pred, y_std = gp.predict(X, return_std=True)
                
                # Plot predictions vs actual with error bars
                self.model_canvas.axes.errorbar(
                    y, y_pred, yerr=2*y_std, fmt='o', alpha=0.7, 
                    ecolor='gray', capsize=5, markersize=8
                )
            except Exception as e:
                print(f"GP fitting failed, falling back to simple regression: {e}")
                # Fall back to simple regression
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(X, y)
                y_pred = lr.predict(X)
                self.model_canvas.axes.scatter(y, y_pred, marker='o', s=50, alpha=0.7)
                
            # Plot diagonal line representing perfect prediction
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            margin = (max_val - min_val) * 0.1
            line_start = min_val - margin
            line_end = max_val + margin
            
            self.model_canvas.axes.plot([line_start, line_end], [line_start, line_end], 
                                'k--', alpha=0.7, label='Perfect Prediction')
            
            # Calculate and show R² 
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            self.model_canvas.axes.annotate(f'R² = {r2:.3f}', 
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              ha='left', va='top', fontsize=12)
            
            # Set labels and title
            self.model_canvas.axes.set_xlabel(f'Actual {objective} (%)')
            self.model_canvas.axes.set_ylabel(f'Predicted {objective} (%)')
            self.model_canvas.axes.set_title(f'Prediction Model for {objective}')
            self.model_canvas.axes.grid(True, alpha=0.3)
            
            self.model_canvas.draw()
            self.log(f"-- Prediction model for {objective} updated - Success")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.model_canvas.axes.clear()
            self.model_canvas.axes.text(0.5, 0.5, f"Error fitting model: {str(e)}", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            self.log(f"-- Error updating model plot: {str(e)} - Error")
            print(f"Model plot error details: {error_details}")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        self.model_canvas.axes.clear()
        self.model_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
            ha='center', va='center', transform=self.model_canvas.axes.transAxes)
        self.model_canvas.draw()
        self.log(f"-- Error updating model plot: {str(e)} - Error")
        print(f"Model plot error details: {error_details}")

def update_surface_plot(self):
    # Quick check if plot generation is possible
    if not self.model.experiments or len(self.model.experiments) < 4:
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.text2D(0.5, 0.5, 
            "Need at least 4 experiments to generate surface plot",
            ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
        self.surface_canvas.draw()
        self.log("-- Surface plot not generated: insufficient data")
        return
    
    # Show loading indicator
    self.update_surface_btn.setEnabled(False)
    self.update_surface_btn.setText("Generating plot...")
    QApplication.processEvents()
    
    # Use worker thread for computation
    worker = SurfacePlotWorker(self.model, 
                            self.surface_obj_combo.currentText(),
                            self.surface_x_combo.currentText(), 
                            self.surface_y_combo.currentText())
    
    worker.resultReady.connect(lambda result: self._handle_surface_result(result))
    worker.finished.connect(lambda: self._reset_surface_button())
    worker.start()
    
def _handle_surface_result(self, result):
    if result['success']:
        # Apply result to plot
        from matplotlib import cm
        
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.plot_surface(
            result['x_mesh'], result['y_mesh'], result['z_values'],
            cmap=cm.viridis, alpha=0.8, antialiased=True)
            
        # Add data points
        self.surface_canvas.axes.scatter(
            result['x_data'], result['y_data'], result['z_data'],
            c='r', marker='o', s=50, label='Experiments')
            
        self.surface_canvas.axes.set_xlabel(result['x_label'])
        self.surface_canvas.axes.set_ylabel(result['y_label'])
        self.surface_canvas.axes.set_zlabel(result['z_label'])
        self.surface_canvas.axes.set_title(result['title'])
        
        self.log(f"-- Surface plot for {result['title']} generated successfully")
    else:
        # Show error message
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.text2D(0.5, 0.5, f"Error: {result['error']}",
            ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
        
        # Log the error
        self.log(f"-- Surface plot generation failed: {result['error']} - Error")
        
        # Print detailed traceback if available
        if 'details' in result:
            print(f"Surface plot error details:\n{result['details']}")
    
    self.surface_canvas.draw()
    
def _reset_surface_button(self):
    self.update_surface_btn.setEnabled(True)
    self.update_surface_btn.setText("Update Surface")

def update_convergence_plot(self):
    try:
        self.convergence_canvas.axes.clear()
        
        if not self.model.experiments or len(self.model.experiments) < 3:
            self.convergence_canvas.axes.text(0.5, 0.5, "Need at least 3 experiments for convergence analysis", 
                ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
            self.convergence_canvas.draw()
            self.log("-- Convergence plot not updated: insufficient data")
            return
            
        # Extract scores
        scores = []
        for exp in self.model.experiments:
            if 'score' in exp and exp['score'] is not None:
                scores.append(exp['score'] * 100.0)
            else:
                # Calculate score if not present
                if 'results' in exp:
                    score = self.model._calculate_composite_score(exp['results'])
                    scores.append(score * 100.0)
        
        if not scores:
            self.convergence_canvas.axes.text(0.5, 0.5, "No valid scores found in experiments", 
                ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
            self.convergence_canvas.draw()
            self.log("-- Convergence plot not updated: no valid scores")
            return
            
        # Plot scores
        x = range(1, len(scores) + 1)
        self.convergence_canvas.axes.plot(x, scores, 'o-', label='Experiment Scores')
        
        # Plot best so far
        best_scores = np.maximum.accumulate(scores)
        self.convergence_canvas.axes.plot(x, best_scores, 'r-', label='Best Score So Far')
        
        # Set labels
        self.convergence_canvas.axes.set_xlabel('Experiment Number')
        self.convergence_canvas.axes.set_ylabel('Score (%)')
        self.convergence_canvas.axes.set_title('Optimization Convergence')
        self.convergence_canvas.axes.legend()
        self.convergence_canvas.axes.grid(True, alpha=0.3)
        
        self.convergence_canvas.draw()
        self.log("-- Convergence plot updated - Success")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        self.convergence_canvas.axes.clear()
        self.convergence_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
            ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
        self.convergence_canvas.draw()
        self.log(f"-- Error updating convergence plot: {str(e)} - Error")
        print(f"Convergence plot error details: {error_details}")

def update_correlation_plot(self):
    if not self.model.experiments or len(self.model.experiments) < 3:
        self.corr_canvas.axes.clear()
        self.corr_canvas.axes.text(0.5, 0.5, "Need at least 3 experiments for correlation analysis", 
            ha='center', va='center', transform=self.corr_canvas.axes.transAxes)
        self.corr_canvas.draw()
        return
        
    try:
        import pandas as pd
        
        data = []
        for exp in self.model.experiments:
            row = {}
            for param_name in self.model.parameters:
                if param_name in exp['params']:
                    if self.model.parameters[param_name].param_type == "categorical":
                        choices = self.model.parameters[param_name].choices
                        value = exp['params'][param_name]
                        row[param_name] = choices.index(value) if value in choices else 0
                    else:
                        row[param_name] = float(exp['params'][param_name])
            
            for obj in self.model.objectives:
                if obj in exp['results']:
                    row[obj] = exp['results'][obj] * 100.0
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        corr = df.corr()
        
        self.corr_canvas.axes.clear()
        im = self.corr_canvas.axes.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=self.corr_canvas.axes)
        
        self.corr_canvas.axes.set_xticks(np.arange(len(corr.columns)))
        self.corr_canvas.axes.set_yticks(np.arange(len(corr.columns)))
        self.corr_canvas.axes.set_xticklabels(corr.columns, rotation=45, ha="right")
        self.corr_canvas.axes.set_yticklabels(corr.columns)
        
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = self.corr_canvas.axes.text(j, i, f"{corr.iloc[i, j]:.2f}",
                                            ha="center", va="center", 
                                            color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
                                 
        self.corr_canvas.axes.set_title("Parameter-Result Correlation Matrix")
        self.corr_canvas.draw()
        self.log("-- Correlation plot updated - Success")
        
    except Exception as e:
        self.corr_canvas.axes.clear()
        self.corr_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
            ha='center', va='center', transform=self.corr_canvas.axes.transAxes)
        self.corr_canvas.draw()
        self.log(f"-- Error updating correlation plot: {str(e)} - Error")

class SurfacePlotWorker(QThread):
    """Worker thread for generating surface plots without blocking UI."""
    resultReady = Signal(object)
    
    def __init__(self, model, objective, x_param, y_param):
        super().__init__()
        self.model = model
        self.objective = objective
        self.x_param = x_param
        self.y_param = y_param
        
    def run(self):
        """Generate surface plot data in background thread."""
        try:
            import numpy as np
            from scipy.interpolate import griddata
            
            # Extract data points
            x_data = []
            y_data = []
            z_data = []
            
            for exp in self.model.experiments:
                if 'params' in exp and 'results' in exp and self.objective in exp['results']:
                    if self.x_param in exp['params'] and self.y_param in exp['params']:
                        x_val = exp['params'][self.x_param]
                        y_val = exp['params'][self.y_param]
                        
                        if self.model.parameters[self.x_param].param_type == "categorical":
                            choices = self.model.parameters[self.x_param].choices
                            x_val = choices.index(x_val) if x_val in choices else 0
                            
                        if self.model.parameters[self.y_param].param_type == "categorical":
                            choices = self.model.parameters[self.y_param].choices
                            y_val = choices.index(y_val) if y_val in choices else 0
                            
                        x_data.append(float(x_val))
                        y_data.append(float(y_val))
                        z_data.append(float(exp['results'][self.objective]) * 100.0)
            
            if len(x_data) < 4:
                self.resultReady.emit({'success': False, 'error': 'Need at least 4 data points for surface plot'})
                return
                
            # Grid preparation
            x_min, x_max = min(x_data), max(x_data)
            y_min, y_max = min(y_data), max(y_data)
            
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            
            x_grid_min = x_min - x_range * 0.1
            x_grid_max = x_max + x_range * 0.1
            y_grid_min = y_min - y_range * 0.1
            y_grid_max = y_max + y_range * 0.1
            
            # Create grid
            n_grid = 50
            xi = np.linspace(x_grid_min, x_grid_max, n_grid)
            yi = np.linspace(y_grid_min, y_grid_max, n_grid)
            x_mesh, y_mesh = np.meshgrid(xi, yi)
            
            # Interpolate data
            try:
                z_values = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method='cubic')
                
                # Fall back to linear interpolation for NaN regions
                mask = np.isnan(z_values)
                if np.any(mask):
                    z_values[mask] = griddata((x_data, y_data), z_data, (x_mesh[mask], y_mesh[mask]), method='linear')
                
                # Fill remaining NaNs with nearest
                mask = np.isnan(z_values)
                if np.any(mask):
                    z_values[mask] = griddata((x_data, y_data), z_data, (x_mesh[mask], y_mesh[mask]), method='nearest')
            except Exception as e:
                # Last resort: nearest neighbor only
                z_values = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method='nearest')
            
            # Prepare labels
            x_label = self.x_param
            if self.model.parameters[self.x_param].units:
                x_label += f" ({self.model.parameters[self.x_param].units})"
                
            y_label = self.y_param
            if self.model.parameters[self.y_param].units:
                y_label += f" ({self.model.parameters[self.y_param].units})"
                
            z_label = f"{self.objective.capitalize()} (%)"
            title = f"Response Surface for {self.objective.capitalize()}"
            
            # Send results
            self.resultReady.emit({
                'success': True,
                'x_mesh': x_mesh,
                'y_mesh': y_mesh,
                'z_values': z_values,
                'x_data': x_data,
                'y_data': y_data,
                'z_data': z_data,
                'x_label': x_label,
                'y_label': y_label,
                'z_label': z_label,
                'title': title
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.resultReady.emit({'success': False, 'error': str(e), 'details': error_details})
