import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PySide6.QtWidgets import QMessageBox
from ..visualizations import (
    plot_optimization_history, plot_parameter_importance, 
    plot_parameter_contour, plot_objective_correlation,
    plot_convergence, plot_response_surface
)

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
    objective = self.model_obj_combo.currentText()
    
    if not objective or not self.model.experiments:
        QMessageBox.warning(self, "Warning", "No experiments or objectives available")
        return
        
    self.model_canvas.axes.clear()
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel
        import numpy as np
        
        X, _ = self.model._extract_normalized_features_and_targets()
        
        y = []
        for exp in self.model.experiments:
            if 'results' in exp and objective in exp['results']:
                y.append(exp['results'][objective] * 100.0)
            else:
                y.append(0.0)
                
        y = np.array(y)
        
        if len(X) < 3 or len(np.unique(y)) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 3 diverse experiments for model fitting")
            return
            
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
        gp.fit(X, y)
        
        importances = {}
        if len(X[0]) > 1:
            from sklearn.inspection import permutation_importance
            
            r = permutation_importance(gp, X, y, n_repeats=10, random_state=42)
            
            feature_idx = 0
            for name, param in self.model.parameters.items():
                if param.param_type == "categorical":
                    n_choices = len(param.choices)
                    imp = np.sum(r.importances_mean[feature_idx:feature_idx+n_choices])
                    importances[name] = imp
                    feature_idx += n_choices
                else:
                    importances[name] = r.importances_mean[feature_idx]
                    feature_idx += 1
                    
            max_imp = max(importances.values()) if importances else 1.0
            if max_imp > 0:
                importances = {k: v / max_imp for k, v in importances.items()}
        
        y_pred, y_std = gp.predict(X, return_std=True)
        
        self.model_canvas.axes.errorbar(
            y, y_pred, yerr=2*y_std, fmt='o', alpha=0.6, 
            ecolor='gray', capsize=5, markersize=8
        )
        
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        margin = (max_val - min_val) * 0.1
        line_start = min_val - margin
        line_end = max_val + margin
        
        self.model_canvas.axes.plot([line_start, line_end], [line_start, line_end], 
                            'k--', alpha=0.7, label='Perfect Prediction')
        
        self.model_canvas.axes.set_xlabel(f'Actual {objective} (%)')
        self.model_canvas.axes.set_ylabel(f'Predicted {objective} (%)')
        self.model_canvas.axes.set_title(f'Gaussian Process Model for {objective}')
        
        from sklearn.metrics import r2_score
        r2 = r2_score(y, y_pred)
        self.model_canvas.axes.annotate(f'R² = {r2:.3f}', 
                          xy=(0.05, 0.95), xycoords='axes fraction',
                          ha='left', va='top', fontsize=12)
        
        if importances:
            imp_text = "Parameter Importance:\n"
            for name, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                imp_text += f"{name}: {imp:.3f}\n"
                
            self.model_canvas.axes.annotate(imp_text, 
                              xy=(0.05, 0.85), xycoords='axes fraction',
                              ha='left', va='top', fontsize=9)
            
        self.model_canvas.draw()
        self.log(f"-- Prediction model for {objective} updated - Success")
        
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Error updating model plot: {str(e)}")
        self.log(f"-- Error updating model plot: {str(e)}")

def update_surface_plot(self):
    objective = self.surface_obj_combo.currentText()
    
    if not objective or not self.model.experiments:
        QMessageBox.warning(self, "Warning", "No experiments or objectives available")
        return
        
    self.surface_canvas.axes.clear()
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel
        
        X, _ = self.model._extract_normalized_features_and_targets()
        
        y = []
        for exp in self.model.experiments:
            if 'results' in exp and objective in exp['results']:
                y.append(exp['results'][objective] * 100.0)
            else:
                y.append(0.0)
                
        y = np.array(y)
        
        if len(X) < 3 or len(np.unique(y)) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 3 diverse experiments for model fitting")
            return
            
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
        gp.fit(X, y)
        
        x_param = self.surface_x_combo.currentText()
        y_param = self.surface_y_combo.currentText()
        
        if not x_param or not y_param or x_param == y_param:
            QMessageBox.warning(self, "Warning", "Select different X and Y parameters")
            return
            
        from ..visualizations import plot_response_surface
        plot_response_surface(
            self.model, gp, x_param, y_param, objective, self.surface_canvas.axes
        )
        
        self.surface_canvas.draw()
        self.log(f"-- Response surface for {objective} updated - Success")
        
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Error updating surface plot: {str(e)}")
        self.log(f"-- Error updating surface plot: {str(e)}")

def update_convergence_plot(self):
    self.convergence_canvas.axes.clear()
    
    try:
        if not self.model.experiments:
            self.convergence_canvas.axes.text(0.5, 0.5, "No experiment data available", 
                ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
            self.convergence_canvas.draw()
            return
            
        from ..visualizations import plot_convergence
        plot_convergence(self.model, self.convergence_canvas.axes)
        
        self.convergence_canvas.draw()
        self.log("-- Convergence plot updated - Success")
        
    except Exception as e:
        self.convergence_canvas.axes.clear()
        self.convergence_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
            ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
        self.convergence_canvas.draw()
        self.log(f"-- Error updating convergence plot: {str(e)} - Error")

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
