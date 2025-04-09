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
            self.prior_canvas.axes.axvline(param.prior_mean - param.prior_std, color='g', linestyle='', alpha=0.7)
            self.prior_canvas.axes.axvline(param.prior_mean + param.prior_std, color='g', linestyle='', alpha=0.7)
            
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
    """Update the results visualization plot."""
    try:
        if not hasattr(self, 'result_canvas') or not hasattr(self, 'plot_type_combo'):
            return
            
        if not self.model.experiments:
            self.log(" No experiments to display in results plot - Warning")
            self.result_canvas.axes.clear()
            self.result_canvas.axes.text(0.5, 0.5, "No experiment data available", 
                                      ha='center', va='center', fontsize=14, color='gray')
            self.result_canvas.draw()
            return
            
        plot_type = self.plot_type_combo.currentText()
        
        self.result_canvas.axes.clear()
        
        if plot_type == "Optimization History":
            _plot_optimization_history(self, self.result_canvas.axes)
        elif plot_type == "Parameter Importance":
            _plot_parameter_importance(self, self.result_canvas.axes)
        elif plot_type == "Parameter Contour":
            _plot_parameter_contour(self, self.result_canvas.axes)
        elif plot_type == "Objective Correlation":
            _plot_objective_correlation(self, self.result_canvas.axes)
        elif plot_type == "Parameter Links":
            _plot_parameter_links(self, self.result_canvas.axes)
        
        self.result_canvas.fig.tight_layout(pad=2.0)
        self.result_canvas.draw()
        
        self.log(f" {plot_type} plot updated - Success")
    except Exception as e:
        import traceback
        print(f"Error updating results plot: {e}")
        print(traceback.format_exc())
        self.log(f" Failed to update {plot_type if 'plot_type' in locals() else 'results'} plot: {str(e)} - Error")

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
            self.log(f" Model plot not updated: insufficient data for {objective}")
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
            self.log(" Model plot not updated: feature extraction failed")
            return
            
        # Ensure consistency
        X = np.array(X)
        y = np.array(y)
        
        # Check for data variation 
        if len(np.unique(y)) < 2:
            self.model_canvas.axes.text(0.5, 0.5, f"Insufficient variation in {objective} values", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            self.log(" Model plot not updated: insufficient result variation")
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
                                'k', alpha=0.7, label='Perfect Prediction')
            
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
            self.log(f" Prediction model for {objective} updated - Success")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.model_canvas.axes.clear()
            self.model_canvas.axes.text(0.5, 0.5, f"Error fitting model: {str(e)}", 
                ha='center', va='center', transform=self.model_canvas.axes.transAxes)
            self.model_canvas.draw()
            self.log(f" Error updating model plot: {str(e)} - Error")
            print(f"Model plot error details: {error_details}")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        self.model_canvas.axes.clear()
        self.model_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
            ha='center', va='center', transform=self.model_canvas.axes.transAxes)
        self.model_canvas.draw()
        self.log(f" Error updating model plot: {str(e)} - Error")
        print(f"Model plot error details: {error_details}")

def update_surface_plot(self):
    # Quick check if plot generation is possible
    if not self.model.experiments or len(self.model.experiments) < 4:
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.text2D(0.5, 0.5, 
            "Need at least 4 experiments to generate surface plot",
            ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
        self.surface_canvas.draw()
        self.log(" Surface plot not generated: insufficient data")
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
        
        self.log(f" Surface plot for {result['title']} generated successfully")
    else:
        # Show error message
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.text2D(0.5, 0.5, f"Error: {result['error']}",
            ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
        
        # Log the error
        self.log(f" Surface plot generation failed: {result['error']} - Error")
        
        # Print detailed traceback if available
        if 'details' in result:
            print(f"Surface plot error details:\n{result['details']}")
    
    self.surface_canvas.draw()
    
def _reset_surface_button(self):
    self.update_surface_btn.setEnabled(True)
    self.update_surface_btn.setText("Update Surface")

def update_convergence_plot(self):
    """Update the convergence analysis plot."""
    try:
        if not hasattr(self, 'convergence_canvas'):
            return
            
        if not self.model.experiments or len(self.model.experiments) < 2:
            self.log(" Not enough experiments for convergence plot - Warning")
            self.convergence_canvas.axes.clear()
            self.convergence_canvas.axes.text(0.5, 0.5, "Need at least 2 experiments for convergence plot", 
                                           ha='center', va='center', fontsize=14, color='gray')
            self.convergence_canvas.draw()
            return
            
        self.convergence_canvas.axes.clear()
        
        # Extract experiment scores in chronological order
        scores = []
        for exp in self.model.experiments:
            if 'score' in exp:
                scores.append(exp['score'] * 100.0)  # Convert to percentage
                
        # Calculate cumulative maximum (best score so far)
        cumulative_max = []
        current_max = 0
        for score in scores:
            current_max = max(current_max, score)
            cumulative_max.append(current_max)
            
        # Calculate moving average
        window_size = min(5, len(scores))
        moving_avg = []
        for i in range(len(scores)):
            if i < window_size - 1:
                # Not enough data points for full window
                window = scores[:i+1]
            else:
                window = scores[i-window_size+1:i+1]
            moving_avg.append(sum(window) / len(window))
            
        # Plot the data
        x = range(1, len(scores) + 1)
        self.convergence_canvas.axes.plot(x, scores, 'o-', color='#3498db', alpha=0.6, label='Individual Results')
        self.convergence_canvas.axes.plot(x, cumulative_max, 'r-', linewidth=2, label='Best So Far')
        self.convergence_canvas.axes.plot(x, moving_avg, 'g--', linewidth=2, label=f'{window_size}-Point Moving Avg')
        
        # Format the plot
        self.convergence_canvas.axes.set_xlabel('Experiment Number')
        self.convergence_canvas.axes.set_ylabel('Score (%)')
        self.convergence_canvas.axes.set_title('Optimization Convergence')
        self.convergence_canvas.axes.grid(True, linestyle='--', alpha=0.7)
        self.convergence_canvas.axes.legend()
        
        # Add regression trend line
        if len(scores) >= 3:
            try:
                import numpy as np
                from scipy import stats
                
                # Calculate trend line
                x_array = np.array(x)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, scores)
                line = slope * x_array + intercept
                
                # Plot trend line
                self.convergence_canvas.axes.plot(x, line, 'k:', linewidth=1.5, 
                                             label=f'Trend (r²={r_value**2:.2f})')
                self.convergence_canvas.axes.legend()
                
                # Extrapolate to estimate convergence
                if slope > 0.1:  # Only show extrapolation if still improving significantly
                    target = max(90, max(cumulative_max) + 5)  # Target either 90% or current best + 5%
                    est_experiments = int((target - intercept) / slope)
                    remaining = max(0, est_experiments - len(scores))
                    
                    if remaining > 0 and remaining < len(scores) * 2:  # Only show reasonable estimates
                        self.convergence_canvas.axes.annotate(
                            f"Est. {remaining} more experiments to reach {target:.0f}%",
                            xy=(len(scores), cumulative_max[-1]),
                            xytext=(len(scores) - 1, cumulative_max[-1] + 10),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='purple'),
                            color='purple'
                        )
            except:
                # Skip trend line if scipy not available or other issues
                pass
                
        self.convergence_canvas.fig.tight_layout(pad=2.0)
        self.convergence_canvas.draw()
        
        self.log(" Convergence plot updated - Success")
    except Exception as e:
        import traceback
        print(f"Error updating convergence plot: {e}")
        print(traceback.format_exc())
        self.log(f" Failed to update convergence plot: {str(e)} - Error")

def update_correlation_plot(self):
    """Update the parameter correlation plot."""
    try:
        if not hasattr(self, 'corr_canvas'):
            return
        
        if not self.model.experiments or len(self.model.experiments) < 3:
            self.log(" Not enough experiments for correlation plot - Warning")
            self.corr_canvas.axes.clear()
            self.corr_canvas.axes.text(0.5, 0.5, "Need at least 3 experiments for correlation analysis", 
                                     ha='center', va='center', fontsize=14, color='gray')
            self.corr_canvas.draw()
            return
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Extract parameters and results into a dataframe
        data = []
        for exp in self.model.experiments:
            if 'params' in exp and 'results' in exp:
                row = {}
                
                # Add parameters
                for param_name, param_value in exp['params'].items():
                    # Skip categorical parameters
                    param = self.model.parameters.get(param_name)
                    if param and param.param_type == 'categorical':
                        continue
                    row[param_name] = param_value
                
                # Add results
                for obj_name, obj_value in exp['results'].items():
                    if obj_value is not None:
                        row[obj_name] = obj_value * 100.0  # Convert to percentage
                
                data.append(row)
        
        if not data:
            self.log(" No valid data for correlation analysis - Warning")
            self.corr_canvas.axes.clear()
            self.corr_canvas.axes.text(0.5, 0.5, "No valid data for correlation analysis", 
                                     ha='center', va='center', fontsize=14, color='gray')
            self.corr_canvas.draw()
            return
        
        # Create dataframe and calculate correlation
        df = pd.DataFrame(data)
        corr = df.corr()
        
        # Plot correlation matrix
        self.corr_canvas.axes.clear()
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create colormap
        cmap = plt.cm.RdBu_r
        
        # Draw the heatmap with the mask and correct aspect ratio
        im = self.corr_canvas.axes.matshow(corr, cmap=cmap, vmin=-1, vmax=1)
        
        # Add colorbar to the current figure
        cbar = self.corr_canvas.fig.colorbar(im, ax=self.corr_canvas.axes)
        cbar.set_label('Correlation Coefficient')
        
        # Set ticks and labels
        self.corr_canvas.axes.set_xticks(range(len(corr.columns)))
        self.corr_canvas.axes.set_yticks(range(len(corr.columns)))
        self.corr_canvas.axes.set_xticklabels(corr.columns, rotation=45, ha='left')
        self.corr_canvas.axes.set_yticklabels(corr.columns)
        
        # Add correlation values to cells
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                # Only show the lower triangle
                if i >= j:
                    val = corr.iloc[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    self.corr_canvas.axes.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
        
        self.corr_canvas.axes.set_title('Parameter and Objective Correlations')
        self.corr_canvas.fig.tight_layout(pad=1.5)
        self.corr_canvas.draw()
        self.log(" Correlation plot updated - Success")
            
    except Exception as e:
        import traceback
        print(f"Error updating correlation plot: {e}")
        print(traceback.format_exc())
        self.log(f" Failed to update correlation plot: {str(e)} - Error")

def update_links_plot(self):
    """Create a network visualization of parameter links discovered during optimization."""
    try:
        ax = self.result_canvas.axes
        ax.clear()
        
        if not self.model.experiments or len(self.model.experiments) < 4:
            ax.text(0.5, 0.5, "Need more experiments to analyze parameter relationships", 
                  ha='center', va='center', transform=ax.transAxes)
            self.result_canvas.draw()
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import FancyArrowPatch
        
        # Get all parameters and count their links
        params = list(self.model.parameters.keys())
        n_params = len(params)
        
        # Create a matrix to represent links
        link_matrix = np.zeros((n_params, n_params))
        link_types = {}  # Store link types (positive/negative)
        
        # Check if parameters have linked_parameters attribute
        for i, p1 in enumerate(params):
            param = self.model.parameters[p1]
            if hasattr(param, 'linked_parameters'):
                for p2, link_info in param.linked_parameters.items():
                    if p2 in params:
                        j = params.index(p2)
                        link_matrix[i, j] = link_info['strength']
                        link_types[(i, j)] = link_info['type']
        
        # If no links found, try to infer them from correlations
        if np.sum(link_matrix) == 0:
            self.log(" No explicit parameter links found. Analyzing correlations...")
            
            # Extract parameter values and results from experiments
            param_values = []
            results = []
            
            for exp in self.model.experiments:
                if 'params' in exp and 'results' in exp and self.model.objectives[0] in exp['results']:
                    param_values.append(exp['params'])
                    results.append(exp['results'][self.model.objectives[0]])
                    
            if len(results) < 4:
                ax.text(0.5, 0.5, "Need more experimental results for correlation analysis", 
                      ha='center', va='center', transform=ax.transAxes)
                self.result_canvas.draw()
                return
                
            # Calculate correlations
            param_matrix = np.zeros((len(param_values), n_params))
            
            for i, param_dict in enumerate(param_values):
                for j, param_name in enumerate(params):
                    if param_name in param_dict:
                        param = self.model.parameters[param_name]
                        if param.param_type == "categorical" and param.choices:
                            # Convert categorical to numeric
                            try:
                                val = param.choices.index(param_dict[param_name]) / (len(param.choices) - 1)
                            except:
                                val = 0
                        else:
                            # Normalize continuous/discrete values
                            val = (param_dict[param_name] - param.low) / (param.high - param.low) if param.high > param.low else 0.5
                        param_matrix[i, j] = val
            
            # Calculate correlation matrix
            import pandas as pd
            corr_matrix = pd.DataFrame(param_matrix, columns=params).corr()
            
            # Use correlations to create links
            for i in range(n_params):
                for j in range(n_params):
                    if i != j:
                        # Only show strong correlations
                        if abs(corr_matrix.iloc[i, j]) > 0.4:
                            link_matrix[i, j] = abs(corr_matrix.iloc[i, j])
                            link_types[(i, j)] = 'positive' if corr_matrix.iloc[i, j] > 0 else 'negative'
        
        # If we still don't have any links, show a message
        if np.sum(link_matrix) == 0:
            ax.text(0.5, 0.5, "No significant parameter relationships detected yet", 
                  ha='center', va='center', transform=ax.transAxes)
            self.result_canvas.draw()
            return
            
        # Create network layout using spring layout
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes
            for i, param in enumerate(params):
                G.add_node(i, name=param)
                
            # Add edges
            for i in range(n_params):
                for j in range(n_params):
                    if link_matrix[i, j] > 0:
                        G.add_edge(i, j, weight=link_matrix[i, j], type=link_types.get((i, j), 'positive'))
            
            # Use spring layout with strength based on weights
            pos = nx.spring_layout(G, k=1/np.sqrt(n_params), iterations=50)
            
        except ImportError:
            # Fallback to simple circular layout
            theta = np.linspace(0, 2 * np.pi, n_params, endpoint=False)
            pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}
        
        # Draw parameter nodes
        for i, param in enumerate(params):
            x, y = pos[i]
            
            # Calculate node size based on number of connections
            node_size = 0.05 + 0.02 * np.sum(link_matrix[i, :] > 0) + 0.02 * np.sum(link_matrix[:, i] > 0)
            circle = plt.Circle((x, y), node_size, color='#4285f4')
            ax.add_patch(circle)
            
            # Add parameter name label
            ax.text(x, y, param, ha='center', va='center', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        
        # Draw links between parameters
        arrow_patches = []
        for i in range(n_params):
            for j in range(n_params):
                if link_matrix[i, j] > 0:
                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    
                    # Link color based on positive/negative relationship
                    color = 'green' if link_types.get((i, j), 'positive') == 'positive' else 'red'
                    
                    # Arrow width based on strength
                    width = 0.005 + 0.02 * link_matrix[i, j]
                    
                    # Create arrow
                    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                          arrowstyle='-|>', color=color,
                                          linewidth=width*50, alpha=0.7,
                                          shrinkA=15, shrinkB=15,
                                          mutation_scale=15,
                                          connectionstyle='arc3,rad=0.1')
                    arrow_patches.append(arrow)
                    ax.add_patch(arrow)
                    
                    # Add small label showing strength
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{link_matrix[i, j]:.2f}",
                          fontsize=7, ha='center', va='center',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Positive Relationship'),
            Line2D([0], [0], color='red', lw=2, label='Negative Relationship')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set equal aspect and remove axis
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set plot title
        ax.set_title('Parameter Relationship Network', fontsize=12)
        
        # Set limits with margin
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        self.result_canvas.draw()
        self.log(" Parameter relationship network plot updated - Success")
        
    except Exception as e:
        import traceback
        self.result_canvas.axes.clear()
        self.result_canvas.axes.text(0.5, 0.5, f"Error creating parameter links plot: {str(e)}", 
                                  ha='center', va='center', transform=self.result_canvas.axes.transAxes)
        self.result_canvas.draw()
        self.log(f" Error plotting parameter links: {str(e)} - Error")
        print(traceback.format_exc())

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

def _plot_optimization_history(self, ax):
    """Plot the optimization history showing how results have improved over time."""
    import numpy as np
    from ..core import settings
    
    if not self.model.experiments:
        ax.text(0.5, 0.5, "No experiment data available", 
                ha='center', va='center', fontsize=14, color='gray')
        return
        
    # Extract objective values
    objectives = self.model.objectives
    
    # Setup data structures to hold objective values
    experiment_indices = []
    objective_values = {obj: [] for obj in objectives}
    best_values = {obj: [] for obj in objectives}
    
    # Process experiments
    for i, exp in enumerate(self.model.experiments):
        experiment_indices.append(i + 1)  # 1-based experiment index
        
        if 'results' in exp:
            for obj in objectives:
                if obj in exp['results'] and exp['results'][obj] is not None:
                    value = float(exp['results'][obj]) * 100.0  # Scale to percentage
                    objective_values[obj].append(value)
                    
                    # Update best value so far
                    if best_values[obj]:
                        best_values[obj].append(max(best_values[obj][-1], value))
                    else:
                        best_values[obj].append(value)
                else:
                    # Fill with nan if missing
                    objective_values[obj].append(np.nan)
                    if best_values[obj]:
                        best_values[obj].append(best_values[obj][-1])
                    else:
                        best_values[obj].append(np.nan)
    
    # Plot 
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, obj in enumerate(objectives):
        color = colors[i % len(colors)]
        
        # Plot individual experiments with some transparency
        ax.scatter(experiment_indices, objective_values[obj], 
                label=f"{obj.capitalize()} Results", 
                color=color, alpha=0.7, s=40)
        
        # Plot best so far line
        ax.plot(experiment_indices, best_values[obj], 
              label=f"Best {obj.capitalize()}", 
              color=color, linestyle='-', linewidth=2)
    
    # Format plot
    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Result (%)")
    ax.set_title("Optimization Progress")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set reasonable y limits
    if len(experiment_indices) > 0:
        all_values = []
        for obj_values in objective_values.values():
            all_values.extend([v for v in obj_values if not np.isnan(v)])
            
        if all_values:
            min_val = max(0, min(all_values) - 5)
            max_val = min(100, max(all_values) + 5)
            ax.set_ylim(min_val, max_val)
    
    # Format x-axis to show integer experiment numbers
    ax.set_xticks(experiment_indices)
    
    # Add legend if we have multiple objectives
    if len(objectives) > 0:
        ax.legend()
    
    return ax

def _plot_parameter_importance(self, ax):
    """Plot parameter importance based on regression coefficients."""
    if not self.model.experiments or len(self.model.experiments) < 3:
        ax.text(0.5, 0.5, "Need at least 3 experiments for parameter importance", 
                ha='center', va='center', fontsize=14, color='gray')
        return
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression, Lasso, Ridge
        
        # Create dataframe from experiments
        data = []
        for exp in self.model.experiments:
            if 'params' not in exp or 'results' not in exp:
                continue
                
            row = {}
            
            # Extract parameter values
            for param_name, param in self.model.parameters.items():
                if param_name in exp['params']:
                    if param.param_type == 'categorical':
                        # We'll handle categorical parameters separately
                        row[f"{param_name}_{exp['params'][param_name]}"] = 1
                    else:
                        row[param_name] = float(exp['params'][param_name])
            
            # Extract result values
            for obj in self.model.objectives:
                if obj in exp['results'] and exp['results'][obj] is not None:
                    row[obj] = float(exp['results'][obj]) * 100.0
            
            data.append(row)
        
        # Convert to dataframe
        df = pd.DataFrame(data)
        
        if len(df) < 3:
            ax.text(0.5, 0.5, "Insufficient valid experiment data", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Select first objective for analysis
        target = self.model.objectives[0]
        if target not in df.columns:
            ax.text(0.5, 0.5, f"No data for {target}", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Separate features and target
        X = df.drop(columns=self.model.objectives, errors='ignore')
        y = df[target]
        
        if X.empty:
            ax.text(0.5, 0.5, "No parameter data available", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=0.5),
            "Lasso Regression": Lasso(alpha=0.1)
        }
        
        # Use the model that gives the best fit
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            model.fit(X_scaled, y)
            score = model.score(X_scaled, y)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Get feature importances (absolute coefficients)
        coefs = np.abs(best_model.coef_)
        
        # Create importance dataframe
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': coefs
        })
        
        # Sort by importance
        importance = importance.sort_values('Importance', ascending=False)
        
        # Plot bar chart
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.barh(
            y=importance['Feature'],
            width=importance['Importance'],
            color=[colors[i % len(colors)] for i in range(len(importance))]
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                ha='left',
                va='center',
                fontsize=9
            )
        
        # Format plot
        ax.set_xlabel("Relative Importance")
        ax.set_title(f"Parameter Importance for {target.capitalize()}")
        ax.set_xlim(0, max(coefs) * 1.1)
        
        # Add note about R² value
        r2_text = f"Model: {max(models.items(), key=lambda x: x[1].score(X_scaled, y))[0]}, R² = {best_score:.2f}"
        ax.text(0.5, -0.1, r2_text, ha='center', transform=ax.transAxes, fontsize=10)
        
    except Exception as e:
        import traceback
        print(f"Error in parameter importance plot: {e}")
        print(traceback.format_exc())
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', fontsize=12, color='red')
    
    return ax

def _plot_parameter_contour(self, ax):
    """Plot a contour map of the parameter space."""
    # Get parameters from UI
    x_param = self.x_param_combo.currentText()
    y_param = self.y_param_combo.currentText()
    
    # Verify parameter selection
    if not x_param or not y_param or x_param == y_param:
        ax.text(0.5, 0.5, "Select different X and Y parameters", 
                ha='center', va='center', fontsize=14, color='gray')
        return
    
    if not self.model.experiments or len(self.model.experiments) < 4:
        ax.text(0.5, 0.5, "Need at least 4 experiments for contour plot", 
                ha='center', va='center', fontsize=14, color='gray')
        return
    
    try:
        import numpy as np
        from scipy.interpolate import griddata
        import matplotlib.pyplot as plt
        
        # Check if parameters exist in the model
        if x_param not in self.model.parameters or y_param not in self.model.parameters:
            ax.text(0.5, 0.5, "Selected parameters not found in model", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Check parameter types
        x_param_obj = self.model.parameters[x_param]
        y_param_obj = self.model.parameters[y_param]
        
        if x_param_obj.param_type == 'categorical' or y_param_obj.param_type == 'categorical':
            ax.text(0.5, 0.5, "Contour plots require continuous parameters", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
        
        # Extract data
        x_values = []
        y_values = []
        z_values = []
        
        for exp in self.model.experiments:
            if 'params' not in exp or 'results' not in exp:
                continue
                
            if x_param in exp['params'] and y_param in exp['params'] and self.model.objectives[0] in exp['results']:
                x_val = float(exp['params'][x_param])
                y_val = float(exp['params'][y_param])
                z_val = float(exp['results'][self.model.objectives[0]]) * 100.0
                
                x_values.append(x_val)
                y_values.append(y_val)
                z_values.append(z_val)
        
        if len(x_values) < 4:
            ax.text(0.5, 0.5, "Insufficient data points for contour plot", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Create grid for contour
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        # Add margin
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 50)
        y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 50)
        
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate data
        try:
            Z = griddata((x_values, y_values), z_values, (X, Y), method='cubic')
            
            # Fill NaN values with linear interpolation
            mask = np.isnan(Z)
            if np.any(mask):
                Z[mask] = griddata((x_values, y_values), z_values, (X[mask], Y[mask]), method='linear')
                
            # Fill remaining NaNs with nearest
            mask = np.isnan(Z)
            if np.any(mask):
                Z[mask] = griddata((x_values, y_values), z_values, (X[mask], Y[mask]), method='nearest')
                
        except Exception as e:
            print(f"Interpolation error: {e}, falling back to nearest neighbor")
            Z = griddata((x_values, y_values), z_values, (X, Y), method='nearest')
        
        # Create contour plot
        levels = np.linspace(min(z_values), max(z_values), 15)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(f"{self.model.objectives[0].capitalize()} (%)")
        
        # Plot data points
        scatter = ax.scatter(x_values, y_values, c=z_values, cmap='viridis', 
                           edgecolor='white', s=50, zorder=10)
        
        # Formatting
        ax.set_xlabel(f"{x_param}{' (' + x_param_obj.units + ')' if x_param_obj.units else ''}")
        ax.set_ylabel(f"{y_param}{' (' + y_param_obj.units + ')' if y_param_obj.units else ''}")
        ax.set_title(f"Response Surface: {self.model.objectives[0].capitalize()}")
        
        # Find and mark best point
        best_idx = np.argmax(z_values)
        ax.scatter([x_values[best_idx]], [y_values[best_idx]], marker='*', s=200, 
                 edgecolor='black', facecolor='white', zorder=20, label='Best Result')
        
        # Add best point annotation
        ax.annotate(f"Best: {z_values[best_idx]:.1f}%", 
                  xy=(x_values[best_idx], y_values[best_idx]),
                  xytext=(10, 10), textcoords='offset points',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Grid and legend
        ax.grid(True, linestyle='--', alpha=0.3)
        
    except Exception as e:
        import traceback
        print(f"Error in contour plot: {e}")
        print(traceback.format_exc())
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', fontsize=12, color='red')
    
    return ax

def _plot_objective_correlation(self, ax):
    """Plot correlation between different objectives."""
    if len(self.model.objectives) < 2:
        ax.text(0.5, 0.5, "Need at least 2 objectives for correlation analysis", 
                ha='center', va='center', fontsize=14, color='gray')
        return
        
    if not self.model.experiments or len(self.model.experiments) < 3:
        ax.text(0.5, 0.5, "Need at least 3 experiments for correlation analysis", 
                ha='center', va='center', fontsize=14, color='gray')
        return
    
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Extract objective values
        data = {}
        for obj in self.model.objectives:
            data[obj] = []
            
        for exp in self.model.experiments:
            if 'results' not in exp:
                continue
                
            for obj in self.model.objectives:
                if obj in exp['results'] and exp['results'][obj] is not None:
                    data[obj].append(float(exp['results'][obj]) * 100.0)
                else:
                    data[obj].append(np.nan)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 3:
            ax.text(0.5, 0.5, "Insufficient data with all objectives measured", 
                    ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Get first two objectives for scatter plot
        obj1, obj2 = self.model.objectives[:2]
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(df[obj1], df[obj2])
        
        # Create scatter plot
        scatter = ax.scatter(df[obj1], df[obj2], 
                           c=range(len(df)), cmap='viridis', 
                           s=50, alpha=0.8)
        
        # Add labels
        for i, (x, y) in enumerate(zip(df[obj1], df[obj2])):
            ax.annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add regression line
        x_min, x_max = min(df[obj1]), max(df[obj1])
        x_line = np.linspace(x_min, x_max, 100)
        
        slope, intercept, _, _, _ = stats.linregress(df[obj1], df[obj2])
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'r--', linewidth=2)
        
        # Add correlation info
        correlation_text = f"Correlation: {corr:.2f}"
        if p_value < 0.05:
            correlation_text += " (significant)"
        else:
            correlation_text += " (not significant)"
            
        p_value_text = f"p-value: {p_value:.4f}"
        
        ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes, fontsize=10, va='top')
        ax.text(0.05, 0.90, p_value_text, transform=ax.transAxes, fontsize=10, va='top')
        
        # Format plot
        ax.set_xlabel(f"{obj1.capitalize()} (%)")
        ax.set_ylabel(f"{obj2.capitalize()} (%)")
        ax.set_title(f"Correlation: {obj1.capitalize()} vs {obj2.capitalize()}")
        
        # Add color bar showing experiment order
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Experiment Order")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
    except Exception as e:
        import traceback
        print(f"Error in objective correlation plot: {e}")
        print(traceback.format_exc())
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', fontsize=12, color='red')
    
    return ax

def _plot_parameter_links(self, ax):
    """Plot network visualization of parameter links."""
    if not self.model.experiments or len(self.model.experiments) < 4:
        ax.text(0.5, 0.5, "Need more experiments to analyze parameter relationships", 
                ha='center', va='center', fontsize=14, color='gray')
        return
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import FancyArrowPatch
        
        # Get all parameters and count their links
        params = list(self.model.parameters.keys())
        n_params = len(params)
        
        # Create a matrix to represent links
        link_matrix = np.zeros((n_params, n_params))
        link_types = {}  # Store link types (positive/negative)
        
        # Check if parameters have linked_parameters attribute
        for i, p1 in enumerate(params):
            param = self.model.parameters[p1]
            if hasattr(param, 'linked_parameters') and param.linked_parameters:
                for p2, link_info in param.linked_parameters.items():
                    if p2 in params:
                        j = params.index(p2)
                        link_matrix[i, j] = link_info.get('strength', 0.5)
                        link_types[(i, j)] = link_info.get('type', 'positive')
        
        # If no links found, try to infer them from correlations
        if np.sum(link_matrix) == 0:
            self.log(" No explicit parameter links found. Analyzing correlations...")
            
            # Extract parameter values and results from experiments
            param_values = []
            results = []
            
            for exp in self.model.experiments:
                if 'params' in exp and 'results' in exp and self.model.objectives[0] in exp['results']:
                    param_values.append(exp['params'])
                    results.append(exp['results'][self.model.objectives[0]])
                    
            if len(results) < 4:
                ax.text(0.5, 0.5, "Need more experimental results for correlation analysis", 
                      ha='center', va='center', fontsize=14, color='gray')
                return
                
            # Calculate correlations
            param_matrix = np.zeros((len(param_values), n_params))
            
            for i, param_dict in enumerate(param_values):
                for j, param_name in enumerate(params):
                    if param_name in param_dict:
                        param = self.model.parameters[param_name]
                        if param.param_type == "categorical" and param.choices:
                            # Convert categorical to numeric
                            try:
                                val = param.choices.index(param_dict[param_name]) / (len(param.choices) - 1)
                            except:
                                val = 0
                        else:
                            # Normalize continuous/discrete values
                            val = (param_dict[param_name] - param.low) / (param.high - param.low) if param.high > param.low else 0.5
                        param_matrix[i, j] = val
            
            # Calculate correlation matrix
            import pandas as pd
            corr_matrix = pd.DataFrame(param_matrix, columns=params).corr()
            
            # Use correlations to create links
            for i in range(n_params):
                for j in range(n_params):
                    if i != j:
                        # Only show strong correlations
                        if abs(corr_matrix.iloc[i, j]) > 0.4:
                            link_matrix[i, j] = abs(corr_matrix.iloc[i, j])
                            link_types[(i, j)] = 'positive' if corr_matrix.iloc[i, j] > 0 else 'negative'
        
        # If we still don't have any links, show a message
        if np.sum(link_matrix) == 0:
            ax.text(0.5, 0.5, "No significant parameter relationships detected yet", 
                  ha='center', va='center', fontsize=14, color='gray')
            return
            
        # Create network layout using spring layout
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes
            for i, param in enumerate(params):
                G.add_node(i, name=param)
                
            # Add edges
            for i in range(n_params):
                for j in range(n_params):
                    if link_matrix[i, j] > 0:
                        G.add_edge(i, j, weight=link_matrix[i, j], type=link_types.get((i, j), 'positive'))
            
            # Use spring layout with strength based on weights
            pos = nx.spring_layout(G, k=1/np.sqrt(n_params), iterations=50)
            
        except ImportError:
            # Fallback to simple circular layout
            theta = np.linspace(0, 2 * np.pi, n_params, endpoint=False)
            pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}
        
        # Draw parameter nodes
        for i, param in enumerate(params):
            x, y = pos[i]
            
            # Calculate node size based on number of connections
            node_size = 0.05 + 0.02 * np.sum(link_matrix[i, :] > 0) + 0.02 * np.sum(link_matrix[:, i] > 0)
            circle = plt.Circle((x, y), node_size, color='#4285f4')
            ax.add_patch(circle)
            
            # Add parameter name label
            ax.text(x, y, param, ha='center', va='center', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        
        # Draw links between parameters
        arrow_patches = []
        for i in range(n_params):
            for j in range(n_params):
                if link_matrix[i, j] > 0:
                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    
                    # Link color based on positive/negative relationship
                    color = 'green' if link_types.get((i, j), 'positive') == 'positive' else 'red'
                    
                    # Arrow width based on strength
                    width = 0.005 + 0.02 * link_matrix[i, j]
                    
                    # Create arrow
                    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                          arrowstyle='-|>', color=color,
                                          linewidth=width*50, alpha=0.7,
                                          shrinkA=15, shrinkB=15,
                                          mutation_scale=15,
                                          connectionstyle='arc3,rad=0.1')
                    arrow_patches.append(arrow)
                    ax.add_patch(arrow)
                    
                    # Add small label showing strength
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{link_matrix[i, j]:.2f}",
                          fontsize=7, ha='center', va='center',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Positive Relationship'),
            Line2D([0], [0], color='red', lw=2, label='Negative Relationship')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set equal aspect and remove axis
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set plot title
        ax.set_title('Parameter Relationship Network', fontsize=12)
        
        # Set limits with margin
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
    except Exception as e:
        import traceback
        print(f"Error in parameter links plot: {e}")
        print(traceback.format_exc())
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', fontsize=12, color='red')
    
    return ax
