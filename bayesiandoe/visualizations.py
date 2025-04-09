import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from .core import _calculate_parameter_distance

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
            ax.plot(x, best_values, marker='', linestyle='--', 
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

def plot_response_surface(model, obj, x_param, y_param, ax=None):
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
        
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        xi, yi = np.meshgrid(xi, yi)
        
        zi = griddata((x_data, y_data), z_data, (xi, yi), method='cubic')
        
        zi = np.clip(zi, 0, 100)
        
        ax.set_facecolor('#f5f5f5')
        
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', 
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if not model.experiments:
        ax.text(0.5, 0.5, "No experiment results yet", 
               ha='center', va='center', transform=ax.transAxes)
        return ax
        
    scores = []
    
    for exp in model.experiments:
        score = 0
        weight_sum = 0
        for obj in model.objectives:
            if obj in exp and exp[obj] is not None:
                weight = model.objective_weights.get(obj, 1.0)
                score += exp[obj] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            score = score / weight_sum
            
        scores.append(score * 100.0)
        
    best_scores = np.maximum.accumulate(scores)
    x = range(1, len(scores) + 1)
    
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.plot(x, scores, 'o-', color='#4285f4', alpha=0.7, 
           label='Experiment Scores', markersize=6)
    
    ax.plot(x, best_scores, 'r-', linewidth=2.5, color='#ea4335', 
           label='Best Score So Far')
    
    try:
        from scipy.optimize import curve_fit
        
        def convergence_model(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c
            
        p0 = [max(100.0 - min(best_scores), 1.0), 0.1, min(best_scores[0], best_scores[-1])]
        popt, _ = curve_fit(convergence_model, x, best_scores, p0=p0, maxfev=10000)
                          
        future_rounds = max(20, len(scores) * 2)
        x_fit = np.linspace(1, len(scores) + future_rounds, 100)
        y_fit = convergence_model(x_fit, *popt)
        
        a, b, c = popt
        
        asymptotic_value = min(150.0, a + c)
        
        y_fit = np.minimum(y_fit, 150.0)
        
        ax.plot(x_fit, y_fit, '--', color='#34a853', linewidth=2,
              label=f'Projected Convergence (Max: {asymptotic_value:.1f}%)')
                           
        ax.axhline(asymptotic_value, color='#fbbc05', linestyle=':', linewidth=2,
                 label=f'Projected Maximum ({asymptotic_value:.1f}%)')
                            
        current = best_scores[-1]
        target = 0.95 * asymptotic_value
        
        if a > 0:
            target_x = -np.log(1 - (target - c) / a) / b if a > 0 else float('inf')
            remaining_rounds = max(0, int(np.ceil(target_x - len(best_scores))))
            
            if remaining_rounds > 0 and remaining_rounds < 50:
                ax.axvline(target_x, color='#34a853', linestyle='--', alpha=0.5)
                
                ax.annotate(
                    f'{remaining_rounds} rounds to reach target', 
                    xy=(target_x, target),
                    xytext=(len(scores) + 1, current + (target - current) * 0.5),
                    arrowprops=dict(arrowstyle='->', color='#34a853', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                    fontsize=10)
        
        ax.annotate(
            f'Current: {current:.1f}%', 
            xy=(len(scores), current), 
            xytext=(len(scores) - min(len(scores) // 2, 3), current * 0.85),
            arrowprops=dict(arrowstyle='->', color='#4285f4', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            fontsize=10)
    except Exception:
        pass
        
    ax.set_xlabel('Experiment Number', fontsize=12)
    ax.set_ylabel('Composite Score (%)', fontsize=12)
    ax.set_title('Optimization Convergence Analysis', fontsize=14, fontweight='bold')
    
    max_score = max(max(best_scores), 100.0) if best_scores.size > 0 else 100.0
    y_max = min(150.0, max_score * 1.2)
    ax.set_ylim(0, y_max)
    
    legend = ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
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