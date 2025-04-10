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
    """Update result plots based on selected type."""
    if not self.model.experiments:
        self.result_canvas.axes.clear()
        self.result_canvas.axes.text(0.5, 0.5, "No experiments yet", 
                                  ha='center', va='center', fontsize=16, color='gray')
        self.result_canvas.draw()
        return
        
    plot_type = self.plot_type_combo.currentText()
    
    # Clear the current plot
    self.result_canvas.figure.clear()
    ax = self.result_canvas.figure.add_subplot(111)
    
    # Apply modern styling
    from matplotlib import pyplot as plt
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Custom color scheme
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22']
    
    try:
        if plot_type == "Optimization History":
            # Plot optimization history
            from ..visualizations import plot_optimization_history
            plot_optimization_history(self.model, ax)
            
        elif plot_type == "Parameter Importance":
            # Enhanced parameter importance plot
            if len(self.model.experiments) < 3:
                ax.text(0.5, 0.5, "Need at least 3 experiments for parameter importance", 
                       ha='center', va='center', fontsize=14, color='gray')
            else:
                import pandas as pd
                import numpy as np
                from sklearn.linear_model import LinearRegression, Ridge
                
                # Get the first objective
                target = self.model.objectives[0]
                
                # Extract data from experiments
                data = []
                for exp in self.model.experiments:
                    if 'params' in exp and 'results' in exp and target in exp['results']:
                        row = {'result': exp['results'][target] * 100}
                        for param, value in exp['params'].items():
                            # Handle continuous params
                            if self.model.parameters[param].param_type in ['continuous', 'discrete']:
                                row[param] = float(value)
                            else:
                                # One-hot encoding for categorical
                                row[f"{param}_{value}"] = 1
                        data.append(row)
                
                if not data:
                    ax.text(0.5, 0.5, "No complete experiment data available", 
                           ha='center', va='center', fontsize=14, color='gray')
                    self.result_canvas.draw()
                    return
                
                # Create dataframe
                df = pd.DataFrame(data).fillna(0)
                
                # Create features and target
                X = df.drop(columns=['result'])
                y = df['result']
                
                # Calculate feature importance
                model = Ridge(alpha=0.1)
                model.fit(X, y)
                importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': np.abs(model.coef_)
                }).sort_values('Importance', ascending=False)
                
                # Plot top 10 most important parameters
                top_n = min(10, len(importance))
                top_features = importance.head(top_n)
                
                # Create bar chart with gradient colors
                bars = ax.barh(
                    top_features['Feature'],
                    top_features['Importance'],
                    color=colors[:top_n],
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=0.7
                )
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width * 1.02, bar.get_y() + bar.get_height()/2, 
                           f"{width:.2f}", ha='left', va='center')
                
                # Style the plot
                ax.set_xlabel('Relative Importance', fontweight='bold')
                ax.set_title(f'Parameter Importance for {target.capitalize()}', 
                            fontsize=14, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
            
        elif plot_type == "Parameter Contour":
            # Plot parameter contour
            x_param = self.x_param_combo.currentText()
            y_param = self.y_param_combo.currentText()
            
            if not x_param or not y_param:
                ax.text(0.5, 0.5, "Please select X and Y parameters", 
                       ha='center', va='center', fontsize=14, color='gray')
            elif x_param == y_param:
                ax.text(0.5, 0.5, "X and Y parameters must be different", 
                       ha='center', va='center', fontsize=14, color='gray')
            else:
                from ..visualizations import plot_parameter_contour
                plot_parameter_contour(self.model, x_param, y_param, ax)
            
        elif plot_type == "Objective Correlation":
            # Plot objective correlation
            if len(self.model.objectives) < 2:
                ax.text(0.5, 0.5, "Need at least 2 objectives for correlation", 
                       ha='center', va='center', fontsize=14, color='gray')
            else:
                import numpy as np
                from scipy import stats
                
                # Get first two objectives
                obj1, obj2 = self.model.objectives[:2]
                
                # Extract data
                data_x = []
                data_y = []
                exp_numbers = []
                
                for i, exp in enumerate(self.model.experiments):
                    if 'results' in exp and obj1 in exp['results'] and obj2 in exp['results']:
                        if exp['results'][obj1] is not None and exp['results'][obj2] is not None:
                            data_x.append(exp['results'][obj1] * 100)
                            data_y.append(exp['results'][obj2] * 100)
                            exp_numbers.append(i+1)
                
                if len(data_x) < 3:
                    ax.text(0.5, 0.5, "Need at least 3 experiments with both objectives", 
                           ha='center', va='center', fontsize=14, color='gray')
                else:
                    # Calculate correlation
                    corr, p_value = stats.pearsonr(data_x, data_y)
                    
                    # Create gradient colormap based on experiment order
                    import matplotlib as mpl
                    norm = mpl.colors.Normalize(vmin=min(exp_numbers), vmax=max(exp_numbers))
                    cmap = mpl.cm.viridis
                    
                    # Create scatter plot with experiment numbers
                    scatter = ax.scatter(
                        data_x, data_y, 
                        c=exp_numbers, cmap=cmap, 
                        s=100, alpha=0.7, 
                        edgecolors='white', linewidths=1.5
                    )
                    
                    # Add experiment numbers as labels
                    for i, (x, y, num) in enumerate(zip(data_x, data_y, exp_numbers)):
                        ax.text(x, y, str(num), ha='center', va='center', 
                               color='white', fontweight='bold', fontsize=8)
                    
                    # Add trend line
                    x_min, x_max = min(data_x), max(data_x)
                    x_range = np.linspace(x_min, x_max, 100)
                    
                    # Calculate regression line
                    slope, intercept, _, _, _ = stats.linregress(data_x, data_y)
                    y_fit = slope * x_range + intercept
                    
                    # Plot trend line
                    ax.plot(x_range, y_fit, '--', color='#e74c3c', 
                           linewidth=2, label=f'Trend Line (r={corr:.2f})')
                    
                    # Add correlation info
                    correlation_text = [
                        f"Correlation: {corr:.2f}",
                        f"p-value: {p_value:.3f}"
                    ]
                    
                    if abs(corr) > 0.7:
                        correlation_text.append("Strong correlation")
                    elif abs(corr) > 0.3:
                        correlation_text.append("Moderate correlation")
                    else:
                        correlation_text.append("Weak correlation")
                        
                    if p_value < 0.05:
                        correlation_text.append("Statistically significant")
                    
                    # Create text box
                    textbox = "\n".join(correlation_text)
                    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                    ax.text(0.05, 0.95, textbox, transform=ax.transAxes, 
                           fontsize=10, va='top', bbox=props)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Experiment Number', fontweight='bold')
                    
                    # Style the plot
                    ax.set_xlabel(f'{obj1.capitalize()} (%)', fontweight='bold')
                    ax.set_ylabel(f'{obj2.capitalize()} (%)', fontweight='bold')
                    ax.set_title(f'Correlation between {obj1.capitalize()} and {obj2.capitalize()}', 
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='lower right')
                    ax.grid(True, linestyle='--', alpha=0.3)
            
        elif plot_type == "Parameter Links":
            # Plot parameter links
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyArrowPatch
            import networkx as nx
            import numpy as np
            
            # Extract parameter relationships
            if len(self.model.experiments) < 4:
                ax.text(0.5, 0.5, "Need at least 4 experiments to analyze parameter relationships", 
                       ha='center', va='center', fontsize=14, color='gray')
            else:
                params = list(self.model.parameters.keys())
                n_params = len(params)
                
                if n_params < 2:
                    ax.text(0.5, 0.5, "Need at least 2 parameters to show relationships", 
                           ha='center', va='center', fontsize=14, color='gray')
                else:
                    # Extract data for correlation analysis
                    data = {}
                    for p in params:
                        data[p] = []
                    
                    # Extract first objective
                    result_data = []
                    
                    for exp in self.model.experiments:
                        if 'params' in exp and 'results' in exp and self.model.objectives[0] in exp['results']:
                            # Store parameter values
                            for p in params:
                                if p in exp['params']:
                                    if self.model.parameters[p].param_type == "categorical":
                                        # Convert categorical to numerical based on index
                                        choices = self.model.parameters[p].choices
                                        value = choices.index(exp['params'][p]) / max(1, len(choices)-1)
                                    else:
                                        # Normalize between 0 and 1
                                        param_obj = self.model.parameters[p]
                                        if param_obj.high > param_obj.low:
                                            value = (float(exp['params'][p]) - param_obj.low) / (param_obj.high - param_obj.low)
                                        else:
                                            value = 0.5
                                    data[p].append(value)
                                else:
                                    data[p].append(np.nan)
                            
                            # Store result
                            result_data.append(exp['results'][self.model.objectives[0]])
                    
                    if not result_data or len(result_data) < 3:
                        ax.text(0.5, 0.5, "Insufficient experiment data to analyze relationships", 
                               ha='center', va='center', fontsize=14, color='gray')
                    else:
                        # Convert to numpy arrays
                        import pandas as pd
                        
                        df = pd.DataFrame(data)
                        
                        # Calculate correlation matrix
                        corr_matrix = df.corr().fillna(0)
                        
                        # Create graph from correlations
                        G = nx.Graph()
                        
                        # Add nodes
                        for i, param in enumerate(params):
                            G.add_node(param)
                        
                        # Add edges for strong correlations
                        for i, p1 in enumerate(params):
                            for j, p2 in enumerate(params):
                                if i < j and abs(corr_matrix.loc[p1, p2]) > 0.3:
                                    G.add_edge(p1, p2, weight=abs(corr_matrix.loc[p1, p2]), 
                                              sign=1 if corr_matrix.loc[p1, p2] > 0 else -1)
                        
                        if len(G.edges()) == 0:
                            ax.text(0.5, 0.5, "No significant parameter relationships detected yet", 
                                   ha='center', va='center', fontsize=14, color='gray')
                        else:
                            # Set up a nice layout - spring layout works well for smaller graphs
                            pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), seed=42)
                            
                            # Draw nodes
                            node_sizes = [3000 for _ in G.nodes()]
                            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                                                 node_color=colors[:len(G.nodes())], 
                                                 alpha=0.8, linewidths=1.5, edgecolors='white')
                            
                            # Draw labels
                            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="white", font_weight="bold")
                            
                            # Draw edges with varying width
                            for u, v, data in G.edges(data=True):
                                ax.add_patch(
                                    FancyArrowPatch(
                                        pos[u], pos[v],
                                        arrowstyle='-',
                                        connectionstyle='arc3,rad=0.1',
                                        mutation_scale=15,
                                        linewidth=data['weight'] * 5,
                                        color='green' if data['sign'] > 0 else 'red',
                                        alpha=0.7
                                    )
                                )
                                
                                # Add edge weight text
                                weight = data['weight']
                                edge_x = (pos[u][0] + pos[v][0]) / 2
                                edge_y = (pos[u][1] + pos[v][1]) / 2
                                
                                ax.text(edge_x, edge_y, f"{weight:.2f}", 
                                       fontsize=9, ha='center', va='center',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                            # Add legend
                            from matplotlib.lines import Line2D
                            legend_elements = [
                                Line2D([0], [0], color='green', lw=4, label='Positive Correlation'),
                                Line2D([0], [0], color='red', lw=4, label='Negative Correlation')
                            ]
                            ax.legend(handles=legend_elements, loc='upper right')
                            
                            # Remove axis
                            ax.set_axis_off()
                            
                            # Add title
                            ax.set_title("Parameter Relationship Network", fontsize=14, fontweight='bold')
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        ax.clear()
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
               ha='center', va='center', fontsize=12, color='red')
        self.log(f" Error updating results plot: {str(e)} - Error")
        print(f"Results plot error details:\n{error_details}")
    
    # Enable tight layout for better centering
    self.result_canvas.figure.tight_layout(pad=1.5)
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
            
        # Fit models using multiple approaches
        try:
            # Create subplot grid for different models
            self.model_canvas.axes.set_position([0.1, 0.1, 0.8, 0.8])
            
            # Plot title
            self.model_canvas.figure.suptitle(f'Prediction Models for {objective}', 
                                            fontsize=14, fontweight='bold')
            
            # Plot style
            plt.style.use('ggplot')
            
            # Linear Regression
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred_lr = lr.predict(X)
            r2_lr = lr.score(X, y)
            
            # Random Forest (non-linear)
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            y_pred_rf = rf.predict(X)
            r2_rf = rf.score(X, y)
            
            # Gaussian Process (non-linear with uncertainty)
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, 
                                         alpha=0.1, normalize_y=True)
            gp.fit(X, y)
            y_pred_gp, y_std_gp = gp.predict(X, return_std=True)
            r2_gp = gp.score(X, y)
                        
            # Plot diagonal line for all models
            min_val = min(min(y), min(min(y_pred_lr), min(min(y_pred_rf), min(y_pred_gp))))
            max_val = max(max(y), max(max(y_pred_lr), max(max(y_pred_rf), max(y_pred_gp))))
            margin = (max_val - min_val) * 0.1
            line_start = min_val - margin
            line_end = max_val + margin
            
            # Create a colorful style
            colors = ['#4285F4', '#EA4335', '#34A853', '#FBBC05']
            
            # Create a 2x2 grid of subplots
            self.model_canvas.figure.clear()
            gs = self.model_canvas.figure.add_gridspec(2, 2, wspace=0.3, hspace=0.3)
            
            # Plot 1: All models comparison
            ax1 = self.model_canvas.figure.add_subplot(gs[0, 0])
            ax1.plot([line_start, line_end], [line_start, line_end], 'k--', alpha=0.5)
            ax1.scatter(y, y_pred_lr, color=colors[0], alpha=0.6, label=f'Linear (R²: {r2_lr:.2f})')
            ax1.scatter(y, y_pred_rf, color=colors[1], alpha=0.6, label=f'RF (R²: {r2_rf:.2f})')
            ax1.scatter(y, y_pred_gp, color=colors[2], alpha=0.6, label=f'GP (R²: {r2_gp:.2f})')
            ax1.set_xlabel(f'Actual {objective} (%)')
            ax1.set_ylabel(f'Predicted {objective} (%)')
            ax1.set_title('Model Comparison')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Plot 2: Linear model detailed
            ax2 = self.model_canvas.figure.add_subplot(gs[0, 1])
            ax2.plot([line_start, line_end], [line_start, line_end], 'k--', alpha=0.5)
            ax2.scatter(y, y_pred_lr, color=colors[0], s=60, alpha=0.8)
            for i, txt in enumerate(range(1, len(y) + 1)):
                ax2.annotate(txt, (y[i], y_pred_lr[i]), fontsize=8, alpha=0.8)
            ax2.set_xlabel(f'Actual {objective} (%)')
            ax2.set_ylabel(f'Predicted {objective} (%)')
            ax2.set_title(f'Linear Model (R²: {r2_lr:.2f})')
            ax2.grid(True, alpha=0.3)
                
            # Plot 3: Random Forest detailed
            ax3 = self.model_canvas.figure.add_subplot(gs[1, 0])
            ax3.plot([line_start, line_end], [line_start, line_end], 'k--', alpha=0.5)
            ax3.scatter(y, y_pred_rf, color=colors[1], s=60, alpha=0.8)
            for i, txt in enumerate(range(1, len(y) + 1)):
                ax3.annotate(txt, (y[i], y_pred_rf[i]), fontsize=8, alpha=0.8)
            ax3.set_xlabel(f'Actual {objective} (%)')
            ax3.set_ylabel(f'Predicted {objective} (%)')
            ax3.set_title(f'Random Forest (R²: {r2_rf:.2f})')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Gaussian Process with uncertainty
            ax4 = self.model_canvas.figure.add_subplot(gs[1, 1])
            ax4.plot([line_start, line_end], [line_start, line_end], 'k--', alpha=0.5)
            ax4.errorbar(y, y_pred_gp, yerr=1.96*y_std_gp, fmt='o', 
                       color=colors[2], ecolor='gray', capsize=3, alpha=0.8)
            ax4.set_xlabel(f'Actual {objective} (%)')
            ax4.set_ylabel(f'Predicted {objective} (%)')
            ax4.set_title(f'Gaussian Process (R²: {r2_gp:.2f})')
            ax4.grid(True, alpha=0.3)
            
            # Best model indicator
            best_r2 = max(r2_lr, r2_rf, r2_gp)
            best_model = "Linear" if best_r2 == r2_lr else "Random Forest" if best_r2 == r2_rf else "Gaussian Process"
            self.model_canvas.figure.text(0.5, 0.01, 
                                        f"Best model: {best_model} (R² = {best_r2:.3f})", 
                                        ha='center', fontsize=12, 
                                        bbox=dict(facecolor='lightgreen', alpha=0.5))
            
            self.model_canvas.draw()
            self.log(f" Enhanced prediction models for {objective} updated - Success")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.model_canvas.axes.clear()
            self.model_canvas.axes.text(0.5, 0.5, f"Error fitting models: {str(e)}", 
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
    """Update response surface plot for selected parameters with improved error handling."""
    try:
        # Get selected parameters and objective
        objective = self.surface_obj_combo.currentText()
        x_param = self.surface_x_combo.currentText()
        y_param = self.surface_y_combo.currentText()
        
        if not objective or not x_param or not y_param:
            self.surface_canvas.axes.clear()
            self.surface_canvas.axes.text2D(0.5, 0.5, "Select parameters and objective first", 
                                         ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
            self.surface_canvas.draw()
            return
            
        if x_param == y_param:
            self.surface_canvas.axes.clear()
            self.surface_canvas.axes.text2D(0.5, 0.5, "X and Y parameters must be different", 
                                         ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
            self.surface_canvas.draw()
            return
        
        # Clear the plot and show loading message
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.text2D(0.5, 0.5, "Generating surface plot...", 
                                     ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
        self.surface_canvas.draw()
        
        # Instead of using a QThread which can crash, generate the plot in the main thread
        # but use QApplication.processEvents() to keep UI responsive
        QApplication.processEvents()
        
        # Collect data points for the plot
        x_data = []
        y_data = []
        z_data = []
        
        for exp in self.model.experiments:
            if 'params' in exp and 'results' in exp and objective in exp['results']:
                if x_param in exp['params'] and y_param in exp['params']:
                    try:
                        x_val = exp['params'][x_param]
                        y_val = exp['params'][y_param]
                        
                        # Handle categorical params
                        if self.model.parameters[x_param].param_type == "categorical":
                            choices = self.model.parameters[x_param].choices
                            x_val = choices.index(x_val) if x_val in choices else 0
                            
                        if self.model.parameters[y_param].param_type == "categorical":
                            choices = self.model.parameters[y_param].choices
                            y_val = choices.index(y_val) if y_val in choices else 0
                            
                        x_data.append(float(x_val))
                        y_data.append(float(y_val))
                        z_data.append(float(exp['results'][objective]) * 100.0)
                    except Exception as e:
                        print(f"Skipping data point due to error: {e}")
        
        if len(x_data) < 4:
            self.surface_canvas.axes.clear()
            self.surface_canvas.axes.text2D(0.5, 0.5, "Need at least 4 data points for surface plot", 
                                         ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
            self.surface_canvas.draw()
            self.log(" Surface plot not updated: insufficient data points")
            return
        
        # Create the surface plot
        try:
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            z_data = np.array(z_data)
            
            # Setup grid for interpolation
            x_min, x_max = min(x_data), max(x_data)
            y_min, y_max = min(y_data), max(y_data)
            
            # Add margins
            x_margin = max(0.1, (x_max - x_min) * 0.1)
            y_margin = max(0.1, (y_max - y_min) * 0.1)
            
            x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 20)
            y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 20)
            x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
            
            # Try different interpolation methods, falling back as needed
            try:
                from scipy.interpolate import griddata
                z_mesh = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method='cubic')
                
                # Fill NaN values using linear interpolation
                mask = np.isnan(z_mesh)
                if np.any(mask):
                    z_mesh_linear = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method='linear')
                    z_mesh[mask] = z_mesh_linear[mask]
            except Exception as e:
                print(f"Cubic interpolation failed, trying linear: {e}")
                try:
                    from scipy.interpolate import griddata
                    z_mesh = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method='linear')
                except Exception as e:
                    print(f"Linear interpolation failed, using nearest: {e}")
                    from scipy.interpolate import griddata
                    z_mesh = griddata((x_data, y_data), z_data, (x_mesh, y_mesh), method='nearest')
            
            # If there are still NaN values, use a simple nearest method to fill them
            if np.any(np.isnan(z_mesh)):
                from scipy.interpolate import NearestNDInterpolator
                interpolator = NearestNDInterpolator(list(zip(x_data, y_data)), z_data)
                for i in range(z_mesh.shape[0]):
                    for j in range(z_mesh.shape[1]):
                        if np.isnan(z_mesh[i, j]):
                            z_mesh[i, j] = interpolator(x_mesh[i, j], y_mesh[i, j])
            
            # Cap values between 0 and 100
            z_mesh = np.clip(z_mesh, 0, 100)
            
            # Clear and create plot
            self.surface_canvas.axes.clear()
            
            # Plot the surface with enhanced styling
            from matplotlib import cm
            surf = self.surface_canvas.axes.plot_surface(
                x_mesh, y_mesh, z_mesh, cmap=cm.viridis, 
                alpha=0.8, antialiased=True, edgecolor='none', 
                rstride=1, cstride=1, linewidth=0.1
            )
            
            # Add experimental data points
            scatter = self.surface_canvas.axes.scatter(
                x_data, y_data, z_data, c='r', marker='o', s=60, alpha=0.9,
                edgecolor='black', linewidth=1, label='Experiments'
            )
            
            # Add colorbar
            cbar = self.surface_canvas.figure.colorbar(surf, ax=self.surface_canvas.axes, 
                                                   pad=0.1, shrink=0.8)
            cbar.set_label(f"{objective.capitalize()} (%)", fontsize=10, fontweight='bold')
            
            # Set axis labels with units
            x_label = x_param
            if self.model.parameters[x_param].units:
                x_label += f" ({self.model.parameters[x_param].units})"
                
            y_label = y_param
            if self.model.parameters[y_param].units:
                y_label += f" ({self.model.parameters[y_param].units})"
                
            self.surface_canvas.axes.set_xlabel(x_label, fontsize=10, fontweight='bold', labelpad=10)
            self.surface_canvas.axes.set_ylabel(y_label, fontsize=10, fontweight='bold', labelpad=10)
            self.surface_canvas.axes.set_zlabel(f"{objective.capitalize()} (%)", 
                                           fontsize=10, fontweight='bold', labelpad=10)
            
            # Handle categorical parameters
            if self.model.parameters[x_param].param_type == "categorical":
                choices = self.model.parameters[x_param].choices
                positions = list(range(len(choices)))
                self.surface_canvas.axes.set_xticks(positions)
                self.surface_canvas.axes.set_xticklabels(choices, rotation=45)
                
            if self.model.parameters[y_param].param_type == "categorical":
                choices = self.model.parameters[y_param].choices
                positions = list(range(len(choices)))
                self.surface_canvas.axes.set_yticks(positions)
                self.surface_canvas.axes.set_yticklabels(choices)
            
            # Set limits with some padding
            self.surface_canvas.axes.set_xlim(x_min - x_margin, x_max + x_margin)
            self.surface_canvas.axes.set_ylim(y_min - y_margin, y_max + y_margin)
            self.surface_canvas.axes.set_zlim(0, 100)
            
            # Set title
            self.surface_canvas.axes.set_title(f"Response Surface for {objective.capitalize()}", 
                                          fontsize=12, fontweight='bold')
            
            # Set optimal view angle
            self.surface_canvas.axes.view_init(elev=30, azim=45)
            
            # Add legend
            self.surface_canvas.axes.legend(loc='upper right', fontsize=8)
            
            # Add best point annotation
            best_idx = np.argmax(z_data)
            self.surface_canvas.axes.text(
                x_data[best_idx], y_data[best_idx], z_data[best_idx] + 5, 
                f"Best: {z_data[best_idx]:.1f}%", color='red', 
                fontweight='bold', fontsize=9, ha='center'
            )
            
            self.surface_canvas.draw()
            self.log(f" Response surface plot for {objective} created successfully")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.surface_canvas.axes.clear()
            self.surface_canvas.axes.text2D(0.5, 0.5, f"Error creating surface: {str(e)}", 
                                         ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
            self.surface_canvas.draw()
            self.log(f" Error creating surface plot: {str(e)} - Error")
            print(f"Surface plot error details: {error_details}")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        self.surface_canvas.axes.clear()
        self.surface_canvas.axes.text2D(0.5, 0.5, f"Error: {str(e)}", 
                                     ha='center', va='center', transform=self.surface_canvas.axes.transAxes)
        self.surface_canvas.draw()
        self.log(f" Error updating surface plot: {str(e)} - Error")
        print(f"Surface plot error details: {error_details}")

def update_convergence_plot(self):
    """Update the convergence analysis plot with enhanced visualization."""
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
            
        self.convergence_canvas.figure.clear()
        ax = self.convergence_canvas.figure.add_subplot(111)
        
        # Apply a modern style
        plt.style.use('seaborn-v0_8-colorblind')
        
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
        
        # Plot the data with improved styling
        x = range(1, len(scores) + 1)
        
        # Plot individual points with experiment numbers
        scatter = ax.scatter(x, scores, s=80, color='#3498db', alpha=0.6, 
                          edgecolor='white', linewidth=1, zorder=10)
        
        # Add experiment numbers to points
        for i, (xi, yi) in enumerate(zip(x, scores)):
            ax.annotate(f"{i+1}", (xi, yi), fontsize=9, ha='center', va='center', 
                      color='white', weight='bold', zorder=11)
            
        # Plot best so far with area fill
        ax.plot(x, cumulative_max, color='#e74c3c', linewidth=3, label='Best So Far', zorder=9)
        ax.fill_between(x, 0, cumulative_max, color='#e74c3c', alpha=0.1)
        
        # Plot moving average as a smooth curve
        ax.plot(x, moving_avg, color='#2ecc71', linewidth=2, 
              linestyle='--', label=f'{window_size}-Experiment Average', zorder=8)
            
        # Format the plot with modern styling
        ax.set_xlabel('Experiment Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Optimization Convergence Analysis', fontsize=14, fontweight='bold')
        
        # Add grid but make it subtle
        ax.grid(True, linestyle='-', alpha=0.2)
        
        # Set y-axis limits with some headroom
        y_max = max(100.0, max(scores) * 1.1) if scores else 100.0
        ax.set_ylim(0, min(y_max, 100.0))
        
        # Add reference line at 100%
        ax.axhline(y=100, color='#95a5a6', linestyle='-.', alpha=0.5, label='Theoretical Maximum')
        
        # Add trend prediction if enough data
        if len(scores) >= 4:
            try:
                from scipy import stats
                import numpy as np
                
                # Define convergence model
                def convergence_model(x, a, b, c):
                    """Model of form: a * (1 - exp(-b * x)) + c"""
                    return a * (1 - np.exp(-b * x)) + c
                
                # Fit model to best-so-far data
                from scipy.optimize import curve_fit
                
                # Initial guesses
                p0 = [30, 0.1, max(0, cumulative_max[0])]
                
                try:
                    popt, _ = curve_fit(convergence_model, x, cumulative_max, p0=p0, 
                                      bounds=([0, 0.001, 0], [100, 1, 100]))
                    
                    # Predict future performance
                    future_x = np.linspace(1, len(scores) + 10, 100)
                    future_y = convergence_model(future_x, *popt)
                    future_y = np.minimum(future_y, 100)  # Cap at 100%
                    
                    # Plot prediction curve
                    ax.plot(future_x, future_y, color='#9b59b6', linestyle='-', 
                          linewidth=2, alpha=0.7, label='Predicted Trend', zorder=7)
                    
                    # Add confidence region
                    ax.fill_between(future_x, future_y*0.9, future_y*1.1, 
                                  color='#9b59b6', alpha=0.1, zorder=6)
                    
                    # Estimate experiments to reach 90%
                    if max(cumulative_max) < 90:
                        target = 90
                        # Numerically solve for x where convergence_model = target
                        from scipy.optimize import fsolve
                        def f(x):
                            return convergence_model(x, *popt) - target
                        
                        try:
                            x_at_target = fsolve(f, len(scores) + 5)[0]
                            if x_at_target > len(scores) and x_at_target < len(scores) + 50:
                                exps_needed = int(np.ceil(x_at_target - len(scores)))
                                ax.annotate(f"~ {exps_needed} more experiments to reach {target}%",
                                         xy=(len(scores), cumulative_max[-1]),
                                         xytext=(len(scores) - 1, target - 10),
                                         arrowprops=dict(arrowstyle="->", color='#9b59b6'),
                                         color='#9b59b6', fontweight='bold')
                        except:
                            pass
                except:
                    # Fall back to linear trend if curve fitting fails
                    slope, intercept, r_value, _, _ = stats.linregress(x, cumulative_max)
                    line = slope * np.array(x) + intercept
                    
                    # Plot trend line
                    ax.plot(x, line, color='#f39c12', linewidth=2,
                          label=f'Linear Trend (r²={r_value**2:.2f})', zorder=7)
                    
                    # Project forward
                    future_x = list(range(1, len(scores) + 6))
                    future_y = [slope * xi + intercept for xi in future_x]
                    future_y = [min(yi, 100) for yi in future_y]  # Cap at 100%
                    
                    x_extension = future_x[len(scores):]
                    y_extension = future_y[len(scores):]
                    
                    ax.plot(x_extension, y_extension, color='#f39c12', 
                          linestyle='--', linewidth=2, alpha=0.7, zorder=7)
            except Exception as e:
                print(f"Trend analysis failed: {e}")
                
        # Create a custom legend with larger markers
        legend = ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
        legend.get_frame().set_facecolor('#ffffff')
        
        # Add experiment counts
        ax.text(0.02, 0.98, f"Total experiments: {len(scores)}", 
              transform=ax.transAxes, fontsize=10, va='top', 
              bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
              
        # Add visual stats
        if scores:
            stats_text = (
                f"Best score: {max(scores):.1f}%\n"
                f"Last score: {scores[-1]:.1f}%\n"
                f"Avg score: {sum(scores)/len(scores):.1f}%"
            )
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                  fontsize=10, ha='right', va='bottom',
                  bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
        
        self.convergence_canvas.draw()
        self.log(" Convergence analysis visualization updated - Success")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        self.convergence_canvas.axes.clear()
        self.convergence_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
                                       ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
        self.convergence_canvas.draw()
        self.log(f" Error updating convergence plot: {str(e)} - Error")
        print(f"Convergence plot error details: {error_details}")

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

def update_links_plot(self, ax):
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
