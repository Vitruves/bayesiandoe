import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from .core import _calculate_parameter_distance
from sklearn.gaussian_process import GaussianProcessRegressor

def plot_optimization_history(model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if not model.experiments:
        ax.text(0.5, 0.5, "No experiment results yet", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
        
    data = []
    labels = []
    
    for obj in model.objectives:
        values = [exp.get(obj, 0.0) * 100.0 for exp in model.experiments]
        if values:
            data.append(values)
            labels.append(obj.capitalize())
            
    if not data:
        ax.text(0.5, 0.5, "No objective data available", 
               ha='center', va='center', transform=ax.transAxes)
    else:
        x = range(1, len(model.experiments) + 1)
        for i, values in enumerate(data):
            ax.plot(x, values, marker='o', linestyle='-', label=labels[i])
            
        for i, values in enumerate(data):
            best_values = np.maximum.accumulate(values)
            ax.plot(x, best_values, marker='', linestyle='', 
                   color=ax.lines[i].get_color(), alpha=0.7)
            
        ax.set_xlabel("Experiment Number")
        ax.set_ylabel("Value (%)")
        ax.set_title("Optimization Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    return ax

def plot_parameter_importance(model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if not model.experiments or len(model.experiments) < 3:
        ax.text(0.5, 0.5, "Need at least 3 experiments for importance analysis", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    try:
        importance = model.analyze_parameter_importance()
        
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        params = [p[0] for p in sorted_params]
        importance_values = [p[1] for p in sorted_params]
        
        y_pos = np.arange(len(params))
        ax.barh(y_pos, importance_values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Relative Importance')
        ax.set_title('Parameter Importance Analysis')
        ax.grid(True, alpha=0.3, axis='x')
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
    
    return ax

def plot_parameter_contour(model, x_param, y_param, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if not x_param or not y_param:
        ax.text(0.5, 0.5, "Select X and Y parameters", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
        
    if x_param == y_param:
        ax.text(0.5, 0.5, "X and Y parameters must be different", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    obj = model.objectives[0] if model.objectives else None
    
    if not obj:
        ax.text(0.5, 0.5, "No objective defined", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
        
    x_data = []
    y_data = []
    z_data = []
    
    for exp in model.experiments:
        if x_param in exp and y_param in exp and obj in exp:
            x_val = exp[x_param]
            y_val = exp[y_param]
            
            if model.parameters[x_param].param_type == "categorical":
                choices = model.parameters[x_param].choices
                x_val = choices.index(x_val) if x_val in choices else 0
                
            if model.parameters[y_param].param_type == "categorical":
                choices = model.parameters[y_param].choices
                y_val = choices.index(y_val) if y_val in choices else 0
                
            x_data.append(float(x_val))
            y_data.append(float(y_val))
            z_data.append(float(exp[obj]) * 100.0)
            
    if len(x_data) < 3:
        ax.text(0.5, 0.5, "Need at least 3 data points for contour plot", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    try:
        from scipy.interpolate import griddata
        
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        xi = np.linspace(x_min, x_max, 100)
        yi = np.linspace(y_min, y_max, 100)
        
        zi = griddata((x_data, y_data), z_data, (xi[None, :], yi[:, None]), method='cubic')
        
        contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label=f"{obj.capitalize()} (%)")
        
        ax.scatter(x_data, y_data, c=z_data, cmap='viridis', 
                 edgecolor='k', s=50, zorder=10)
        
        if model.parameters[x_param].param_type == "categorical":
            choices = model.parameters[x_param].choices
            ticks = list(range(len(choices)))
            ax.set_xticks(ticks)
            ax.set_xticklabels(choices, rotation=45, ha='right')
            
        if model.parameters[y_param].param_type == "categorical":
            choices = model.parameters[y_param].choices
            ticks = list(range(len(choices)))
            ax.set_yticks(ticks)
            ax.set_yticklabels(choices)
            
        x_units = f" ({model.parameters[x_param].units})" if model.parameters[x_param].units else ""
        y_units = f" ({model.parameters[y_param].units})" if model.parameters[y_param].units else ""
        
        ax.set_xlabel(f"{x_param}{x_units}")
        ax.set_ylabel(f"{y_param}{y_units}")
        ax.set_title(f"{obj.capitalize()} as a function of {x_param} and {y_param}")
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
    
    return ax

def plot_objective_correlation(model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(model.objectives) < 2:
        ax.text(0.5, 0.5, "Need at least 2 objectives for correlation analysis", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    data = {}
    for obj in model.objectives:
        data[obj] = []
        
    for exp in model.experiments:
        for obj in model.objectives:
            if obj in exp:
                data[obj].append(float(exp[obj]))
            else:
                data[obj].append(float('nan'))
                
    df = pd.DataFrame(data)
    df = df.dropna()
    
    if len(df) < 3:
        ax.text(0.5, 0.5, "Need at least 3 complete experiments for correlation analysis", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    corr = df.corr()
    
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels([obj.capitalize() for obj in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([obj.capitalize() for obj in corr.columns])
    
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
                                 
    ax.set_title("Objective Correlation Matrix")
    return ax

def plot_response_surface(model, x_param, y_param, obj, gp_model=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    if not model.experiments:
        ax.text2D(0.5, 0.5, "No experiment results yet", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
        
    if not x_param or not y_param or x_param == y_param:
        ax.text2D(0.5, 0.5, "Select two different parameters for X and Y", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
        
    try:
        x_data = []
        y_data = []
        z_data = []
        
        for exp in model.experiments:
            if x_param in exp and y_param in exp and obj in exp:
                x_val = exp[x_param]
                y_val = exp[y_param]
                
                if model.parameters[x_param].param_type == "categorical":
                    choices = model.parameters[x_param].choices
                    x_val = choices.index(x_val) if x_val in choices else 0
                    
                if model.parameters[y_param].param_type == "categorical":
                    choices = model.parameters[y_param].choices
                    y_val = choices.index(y_val) if y_val in choices else 0
                    
                x_data.append(float(x_val))
                y_data.append(float(y_val))
                z_data.append(float(exp[obj]) * 100.0)
                
        if len(x_data) < 4:
            ax.text2D(0.5, 0.5, "Need at least 4 data points for surface plot", 
                    ha='center', va='center', transform=ax.transAxes)
            return ax
            
        from scipy.interpolate import griddata
        
        x_min_data, x_max_data = min(x_data), max(x_data)
        y_min_data, y_max_data = min(y_data), max(y_data)
        
        x_range = x_max_data - x_min_data if x_max_data > x_min_data else 1.0
        y_range = y_max_data - y_min_data if y_max_data > y_min_data else 1.0
        
        x_grid_min = x_min_data - x_range * 0.1
        x_grid_max = x_max_data + x_range * 0.1
        y_grid_min = y_min_data - y_range * 0.1
        y_grid_max = y_max_data + y_range * 0.1

        n_grid = 50
        xi = np.linspace(x_grid_min, x_grid_max, n_grid)
        yi = np.linspace(y_grid_min, y_grid_max, n_grid)
        xi_mesh, yi_mesh = np.meshgrid(xi, yi)

        if gp_model:
            # Use GP model for prediction if provided
            grid_points = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()]).T
            
            # Need to construct full feature vectors for prediction
            # Assume other parameters are fixed at their median or mode
            fixed_params = {}
            param_order = list(model.parameters.keys())
            x_param_idx = param_order.index(x_param)
            y_param_idx = param_order.index(y_param)

            full_features_template = []
            norm_x_data = []
            
            for p_name, p_obj in model.parameters.items():
                 if p_name == x_param or p_name == y_param:
                     continue # Placeholder
                 
                 vals = [exp['params'][p_name] for exp in model.experiments if p_name in exp['params']]
                 if not vals:
                     # Use default if no data
                     fixed_val = p_obj.low + (p_obj.high - p_obj.low) / 2 if p_obj.param_type != "categorical" else p_obj.choices[0]
                 elif p_obj.param_type == "categorical":
                      from collections import Counter
                      fixed_val = Counter(vals).most_common(1)[0][0]
                 else:
                     fixed_val = np.median(vals)

                 fixed_params[p_name] = fixed_val
                 
                 # Normalize fixed param values
                 if p_obj.param_type in ["continuous", "discrete"]:
                      norm_val = (fixed_val - p_obj.low) / (p_obj.high - p_obj.low) if p_obj.high > p_obj.low else 0.5
                      full_features_template.append(norm_val)
                 elif p_obj.param_type == "categorical":
                      for choice in p_obj.choices:
                         full_features_template.append(1.0 if fixed_val == choice else 0.0)

            # Prepare grid features for prediction
            X_pred = []
            feature_len = len(model._normalize_params(model.experiments[0]['params'])) # Get expected feature length
            
            for gx, gy in zip(xi_mesh.ravel(), yi_mesh.ravel()):
                current_params = fixed_params.copy()
                current_params[x_param] = gx
                current_params[y_param] = gy
                normalized_features = model._normalize_params(current_params)
                
                # Ensure feature vector length matches training data
                if len(normalized_features) == feature_len:
                    X_pred.append(normalized_features)
                else:
                    # Handle potential length mismatch (e.g., due to categorical encoding issues)
                    print(f"Warning: Feature length mismatch in surface plot. Expected {feature_len}, got {len(normalized_features)}")
                    # Append a zero vector or handle appropriately
                    X_pred.append([0.0] * feature_len) 
            
            X_pred = np.array(X_pred)
            
            if X_pred.shape[0] > 0:
                 zi, std_zi = gp_model.predict(X_pred, return_std=True)
                 zi = zi.reshape(xi_mesh.shape)
                 zi = np.clip(zi, 0, 100) # Clip predictions to valid range
            else:
                 # Fallback if feature preparation failed
                 print("Warning: Could not prepare features for GP prediction in surface plot. Falling back.")
                 zi = griddata((x_data, y_data), z_data, (xi_mesh, yi_mesh), method='cubic', fill_value=np.mean(z_data))

        else:
             # Fallback to griddata if no GP model
            zi = griddata((x_data, y_data), z_data, (xi_mesh, yi_mesh), method='cubic', fill_value=np.mean(z_data))
            zi = np.clip(zi, 0, 100)
        
        ax.set_facecolor('#f5f5f5')
        
        surf = ax.plot_surface(xi_mesh, yi_mesh, zi, cmap='viridis', 
                             alpha=0.8, antialiased=True,
                             rstride=1, cstride=1)
                         
        cbar = plt.colorbar(surf, ax=ax, label=f"{obj.capitalize()} (%)")
        cbar.ax.tick_params(labelsize=10)
                                
        ax.scatter(x_data, y_data, z_data, color='r', 
                  marker='o', s=70, label='Experiments', 
                  edgecolor='black', linewidth=1)
             
        x_label = x_param
        if model.parameters[x_param].units:
            x_label += f" ({model.parameters[x_param].units})"
            
        y_label = y_param
        if model.parameters[y_param].units:
            y_label += f" ({model.parameters[y_param].units})"
            
        ax.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        ax.set_zlabel(f"{obj.capitalize()} (%)", fontsize=12, labelpad=10)
        
        ax.set_zlim(0, 100)
        
        ax.set_title(f"Response Surface for {obj.capitalize()}", 
                    fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10)
        
        ax.view_init(elev=30, azim=45)
        
        if model.parameters[x_param].param_type == "categorical":
            choices = model.parameters[x_param].choices
            ticks = range(len(choices))
            ax.set_xticks(ticks)
            ax.set_xticklabels(choices, rotation=45, ha='right')
            
        if model.parameters[y_param].param_type == "categorical":
            choices = model.parameters[y_param].choices
            ticks = range(len(choices))
            ax.set_yticks(ticks)
            ax.set_yticklabels(choices)
            
    except Exception as e:
        ax.clear()
        ax.text2D(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        
    return ax

def plot_convergence(model, ax=None):
    """Plot the convergence of optimization over experiments with improved analysis.
    
    Shows:
    - Individual experiment scores
    - Best score so far
    - Projected convergence curve
    - Estimated asymptotic performance
    - Confidence regions for predictions
    - Parameter importance inset
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure
    
    if not model.experiments:
        ax.text(0.5, 0.5, "No experiment results yet", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
        
    # Calculate scores for each experiment
    scores = []
    
    for exp in model.experiments:
        score = 0
        weight_sum = 0
        # Use the composite score directly if available and calculated
        if 'score' in exp and exp['score'] is not None:
             score = exp['score']
             scores.append(score * 100.0)
             continue # Skip recalculation if already present
             
        # Fallback: recalculate if 'score' is missing
        for obj in model.objectives:
            if obj in exp.get('results', {}) and exp['results'][obj] is not None:
                weight = model.objective_weights.get(obj, 1.0)
                score += exp['results'][obj] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            score = score / weight_sum
            
        scores.append(score * 100.0)
    
    best_scores = np.maximum.accumulate(scores)
    x = range(1, len(scores) + 1)
    
    # Set up the plot aesthetics
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='', alpha=0.7)
    
    # Mark experiment rounds if available
    if hasattr(model, 'round_start_indices') and model.round_start_indices:
        for idx in model.round_start_indices:
            if 0 < idx < len(scores):
                ax.axvline(idx + 1, color='#aaaaaa', linestyle='', alpha=0.5)
    
    # Plot individual experiment markers, with size based on score
    sizes = [max(20, s * 0.5) for s in scores]
    scatter = ax.scatter(x, scores, s=sizes, c=range(len(scores)), 
                       cmap='viridis', alpha=0.7, 
                       label='Experiment Scores', 
                       edgecolor='white', linewidth=0.5)
    
    # Add colorbar for experiment sequence
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="3%", height="30%", loc='upper left',
                    bbox_to_anchor=(1.02, 0.3, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Experiment Sequence')
    
    # Plot experiment connecting line
    ax.plot(x, scores, '', color='#4285f4', alpha=0.5, linewidth=1)
    
    # Plot the best-so-far scores
    ax.plot(x, best_scores, '-', linewidth=2.5, color='#ea4335', 
           label='Best Score So Far')
    
    # Add parameter importance analysis as inset
    try:
        param_importance = model.analyze_parameter_importance()
        
        if param_importance:
            # Create inset axes for parameter importance
            ax_inset = inset_axes(ax, width="30%", height="25%", loc='lower right',
                                 bbox_to_anchor=(0.98, 0.02, 1, 1),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0)
            
            # Sort parameters by importance
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 5 parameters only
            if len(sorted_params) > 5:
                sorted_params = sorted_params[:5]
                
            params = [p[0] for p in sorted_params]
            importance = [p[1] for p in sorted_params]
            
            # Plot horizontal bar chart
            y_pos = range(len(params))
            ax_inset.barh(y_pos, importance, color='#4285f4', alpha=0.7)
            ax_inset.set_yticks(y_pos)
            ax_inset.set_yticklabels(params, fontsize=8)
            ax_inset.set_xlabel('Importance', fontsize=8)
            ax_inset.set_title('Parameter Importance', fontsize=9)
            ax_inset.set_xlim(0, 1.0)
            
            # Remove spines
            for spine in ax_inset.spines.values():
                spine.set_visible(False)
                
            # Add light gray background
            ax_inset.set_facecolor('#f0f0f0')
            ax_inset.set_axisbelow(True)
            ax_inset.grid(axis='x', alpha=0.3)
    except Exception as e:
        print(f"Error plotting parameter importance: {e}")
    
    # Fit convergence model and project future performance
    try:
        from scipy.optimize import curve_fit
        
        # Define an appropriate convergence model
        def convergence_model(x, a, b, c):
            """Model of form: a * (1 - exp(-b * x)) + c"""
            return a * (1 - np.exp(-b * x)) + c
            
        # Make initial parameter guesses
        init_value = best_scores[0]
        final_value = best_scores[-1]
        growth_rate = 0.1  # Initial guess
        
        # Ensure bounds are valid for curve_fit
        lower_bound = [0, 1e-6, 0]
        upper_bound = [100.0, 1.0, 100.0] # Assume max score is 100%
        
        # Make sure initial guess p0 is within bounds
        p0 = [
            max(lower_bound[0], min(upper_bound[0], final_value - init_value)), 
            max(lower_bound[1], min(upper_bound[1], growth_rate)), 
            max(lower_bound[2], min(upper_bound[2], init_value))
        ]
        
        # Fit the convergence model to the data
        try:
            popt, pcov = curve_fit(convergence_model, x, best_scores, p0=p0, 
                                   bounds=(lower_bound, upper_bound), maxfev=10000)
            a, b, c = popt
            
            # Compute projected future values
            future_rounds = max(20, len(scores) * 2)
            x_fit = np.linspace(1, len(scores) + future_rounds, 100)
            y_fit = convergence_model(x_fit, *popt)
            
            # Cap predictions to reasonable values
            max_possible = 100.0
            y_fit = np.minimum(y_fit, max_possible)
            
            # Plot the convergence curve with confidence regions
            ax.plot(x_fit, y_fit, '', color='#34a853', linewidth=2,
                  label=f'Predicted Convergence')
                  
            # Calculate the asymptotic value (a + c)
            asymptotic_value = min(max_possible, a + c)
            
            # Add a horizontal line for the asymptotic value
            ax.axhline(asymptotic_value, color='#fbbc05', linestyle=':', linewidth=2,
                     label=f'Projected Maximum ({asymptotic_value:.1f}%)')
            
            # Calculate and show how many more rounds needed to reach 95% of maximum
            current = best_scores[-1]
            target = 0.95 * asymptotic_value
            
            if a > 0 and target > current:
                # Calculate required number of iterations for target
                from math import log
                required_x = -log(1 - (target - c) / a) / b if a > 0 else float('inf')
                remaining_rounds = max(0, int(np.ceil(required_x - len(best_scores))))
                
                # Only show if it's a reasonable number
                if 0 < remaining_rounds < 50:
                    ax.axvline(required_x, color='#34a853', linestyle='', alpha=0.5)
                    
                    # Add annotation with arrow
                    ax.annotate(
                        f'~{remaining_rounds} more rounds to reach\n95% of maximum potential', 
                        xy=(required_x, target),
                        xytext=(len(scores) + 1, current + (target - current) * 0.5),
                        arrowprops=dict(arrowstyle='->', color='#34a853', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                        fontsize=10
                    )
                    
            # Add annotation showing current best
            ax.annotate(
                f'Current best: {current:.1f}%', 
                xy=(len(scores), current), 
                xytext=(len(scores) - min(len(scores) // 2, 3), current * 0.85),
                arrowprops=dict(arrowstyle='->', color='#4285f4', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                fontsize=10
            )
            
            # Add a confidence region around the prediction if possible
            try:
                perr = np.sqrt(np.diag(pcov))
                upper = convergence_model(x_fit, *(popt + perr))
                lower = convergence_model(x_fit, *(popt - perr))
                upper = np.minimum(upper, max_possible)
                lower = np.maximum(lower, 0)
                
                ax.fill_between(x_fit, lower, upper, color='#34a853', alpha=0.1)
            except:
                pass  # Skip confidence interval if it fails
                
        except Exception as e:
            import traceback
            print(f"Warning: Failed to fit convergence model: {e}")
            print(traceback.format_exc())
            
    except ImportError:
        print("Warning: scipy.optimize not available, skipping convergence prediction")
        
    ax.set_xlabel('Experiment Number', fontsize=12)
    ax.set_ylabel('Composite Score (%)', fontsize=12)
    ax.set_title('Optimization Convergence Analysis', fontsize=14, fontweight='bold')
    
    max_score = max(100.0, max(scores) * 1.1) if scores else 100.0
    y_max = min(max_score, 100.0)
    ax.set_ylim(0, y_max)
    
    legend = ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    legend.get_frame().set_facecolor('#ffffff')
    
    return ax

def plot_diversity_analysis(model, ax=None):
    """Plot a heatmap showing distances between experiments."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(model.experiments) < 3:
        ax.text(0.5, 0.5, "Need at least 3 experiments for diversity analysis", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Calculate pairwise distances between experiments
    n_exps = len(model.experiments)
    distance_matrix = np.zeros((n_exps, n_exps))
    
    for i in range(n_exps):
        for j in range(i+1, n_exps):
            dist = model.calculate_experiment_distance(
                model.experiments[i]['params'], 
                model.experiments[j]['params']
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Plot the distance heatmap
    im = ax.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Parameter Space Distance')
    
    # Add experiment round labels
    ax.set_xticks(np.arange(n_exps))
    ax.set_yticks(np.arange(n_exps))
    
    exp_labels = [f"Exp {i+1}" for i in range(n_exps)]
    ax.set_xticklabels(exp_labels, rotation=45, ha="right")
    ax.set_yticklabels(exp_labels)
    
    ax.set_title("Experiment Diversity Analysis")
    
    # Add textual distance values
    for i in range(n_exps):
        for j in range(n_exps):
            if i != j:
                text = ax.text(j, i, f"{distance_matrix[i, j]:.2f}",
                             ha="center", va="center", 
                             color="white" if distance_matrix[i, j] > 0.5 else "black",
                             fontsize=8)
    
    return ax