from PySide6.QtWidgets import (
    QMessageBox, QTableWidgetItem, QDialog, QVBoxLayout, QGroupBox,
    QGridLayout, QLabel, QFormLayout, QTextEdit, QHBoxLayout, QPushButton,
    QListWidgetItem
)
from PySide6.QtGui import QColor
from .ui_utils import log, update_ui_from_model, update_prior_table
from .ui_visualization import update_prior_plot
from .dialogs import PriorDialog, ParameterLinkDialog

def update_objectives(self):
    """Update the objectives and weights from the UI controls"""
    try:
        objectives = {}
        for row in range(self.objectives_table.rowCount()):
            obj_item = self.objectives_table.item(row, 0)
            weight_item = self.objectives_table.item(row, 1)
            
            if obj_item and obj_item.text().strip() and weight_item and weight_item.text().strip():
                obj_name = obj_item.text().strip().lower()
                try:
                    weight = float(weight_item.text().strip())
                    objectives[obj_name] = weight
                except ValueError:
                    pass
        
        if objectives:
            # Ensure model has the method
            if not hasattr(self.model, 'set_objectives'):
                # Add the method dynamically if missing
                def set_objectives(model_self, objs):
                    model_self.objectives = list(objs.keys())
                    model_self.objective_weights = objs
                
                import types
                self.model.set_objectives = types.MethodType(set_objectives, self.model)
            
            self.model.set_objectives(objectives)
            self.log(f" Objectives updated: {', '.join(objectives.keys())} - Success")
        else:
            self.log(" No valid objectives found. Please add at least one objective - Error")
    except Exception as e:
        self.log(f" Failed to update objectives: {str(e)} - Error")

def show_registry_item_tooltip(self, item, reg_type, category):
    """Show tooltip with detailed properties for registry items"""
    if not item:
        return
        
    item_name = item.text()
    properties = self.registry_manager.get_item_properties(reg_type, category, item_name)
    
    if properties:
        tooltip = "<html><body><table>"
        tooltip += f"<tr><th colspan='2'>{item_name}</th></tr>"
        
        for key, value in properties.items():
            if key != "color":  # Skip color property
                tooltip += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
        tooltip += "</table></body></html>"
        
        item.setToolTip(tooltip)
    else:
        item.setToolTip("")

def refresh_registry(self):
    for reg_type, categories in self.registry_lists.items():
        for category, list_widget in categories.items():
            list_widget.clear()
            
            items = self.registry_manager.get_item_names(reg_type, category)
            
            for item_name in items:
                list_item = QListWidgetItem(item_name)
                list_widget.addItem(list_item)
                
                props = self.registry_manager.get_item_properties(reg_type, category, item_name)
                if props:
                    tooltip = "\n".join([f"{k}: {v}" for k, v in props.items() if k != "color"])
                    list_item.setToolTip(tooltip)
                    
                    if "color" in props:
                        color_name = props["color"]
                        list_item.setForeground(QColor(color_name))

def on_prior_selected(self):
    selected_items = self.prior_table.selectedItems()
    if not selected_items:
        return
        
    row = selected_items[0].row()
    param_name = self.prior_table.item(row, 0).text()
    if param_name in self.model.parameters:
        self.prior_param_combo.setCurrentText(param_name)
        self.update_prior_ui()
        self.update_prior_plot()

def update_best_results(self):
    n_best = self.n_best_spin.value()
    self.best_table.update_from_model(self.model, n_best)
    log(self, f" Best results table updated showing top {n_best} results - Success")

def show_result_details(self):
    selected_items = self.all_results_table.selectedItems()
    if not selected_items:
        QMessageBox.warning(self, "Warning", "Select a result first.")
        return
        
    row = selected_items[0].row()
    id_item = self.all_results_table.item(row, 0)
    
    if not id_item or not id_item.text().isdigit():
        QMessageBox.warning(self, "Warning", "Invalid result selection.")
        return
        
    exp_id = int(id_item.text()) - 1
    
    if exp_id < 0 or exp_id >= len(self.model.experiments):
        QMessageBox.warning(self, "Warning", "Invalid experiment ID.")
        return
        
    exp_data = self.model.experiments[exp_id]
    
    detail_dialog = QDialog(self)
    detail_dialog.setWindowTitle(f"Experiment #{exp_id+1} Details")
    detail_dialog.resize(600, 400)
    
    layout = QVBoxLayout(detail_dialog)
    
    param_group = QGroupBox("Parameters")
    param_layout = QGridLayout(param_group)
    
    row = 0
    col = 0
    max_cols = 2
    
    for name, value in exp_data['params'].items():
        if name in self.model.parameters:
            param = self.model.parameters[name]
            if isinstance(value, float):
                value_str = f"{value:.4g}"
            else:
                value_str = str(value)
                
            unit_str = f" {param.units}" if param.units else ""
            label_text = f"{name}: {value_str}{unit_str}"
            
            param_layout.addWidget(QLabel(label_text), row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    layout.addWidget(param_group)
    
    result_group = QGroupBox("Results")
    result_layout = QFormLayout(result_group)
    
    for obj, value in exp_data.get('results', {}).items():
        if value is not None:
            result_layout.addRow(f"{obj.capitalize()}:", QLabel(f"{value*100.0:.2f}%"))
            
    layout.addWidget(result_group)
    
    notes_group = QGroupBox("Notes")
    notes_layout = QVBoxLayout(notes_group)
    
    notes_text = QTextEdit()
    notes_text.setReadOnly(True)
    
    if 'params' in exp_data and 'notes' in exp_data['params']:
        notes_text.setText(exp_data['params']['notes'])
    else:
        notes_text.setText("No notes.")
        
    notes_layout.addWidget(notes_text)
    layout.addWidget(notes_group)
    
    buttons = QHBoxLayout()
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(detail_dialog.accept)
    buttons.addWidget(close_btn)
    
    layout.addLayout(buttons)
    
    detail_dialog.exec()

def show_prior_help(self):
    QMessageBox.information(
        self,
        "About Priors",
        "Priors represent your belief about the optimal values for each parameter.\n\n"
        "For continuous and discrete parameters:\n"
        "- Expected Optimal Value: Your best guess for the optimal value\n"
        "- Confidence: How sure you are about your guess\n"
        "- Standard Deviation: Technical parameter controlling spread of values\n\n"
        "For categorical parameters:\n"
        "- Set preference weights for each category (higher = more likely to be selected)\n\n"
        "Setting priors helps guide optimization by focusing on promising regions."
    )

def on_param_button_clicked(self, param_name):
    if not param_name or param_name not in self.model.parameters:
        return
        
    param = self.model.parameters[param_name]
    
    dialog = PriorDialog(self, self.model, param_name)
    if dialog.exec() == QDialog.Accepted and dialog.result:
        if param.param_type in ["continuous", "discrete"]:
            param.set_prior(
                mean=dialog.result.get("mean"),
                std=dialog.result.get("std")
            )
            log(self, f" Prior set for {param_name} (mean={dialog.result.get('mean')}, std={dialog.result.get('std')}) - Success")
        else:
            preferences = dialog.result.get("categorical_preferences", {})
            param.categorical_preferences = preferences
            log(self, f" Categorical prior set for {param_name} - Success")
            
        update_prior_table(self)
        if self.viz_param_combo.currentText() == param_name:
            update_prior_plot(self)

def show_parameter_links(self, param_name):
    """Show and manage parameter links dialog."""
    if not param_name or param_name not in self.model.parameters:
        return
        
    dialog = ParameterLinkDialog(self, self.model, param_name)
    dialog.exec()
    log(self, f" Parameter links updated for {param_name} - Success")

def show_design_method_help(self):
    """Show a dialog to help users choose the best design method."""
    try:
        from .design_selector import DesignMethodSelector
        from PySide6.QtGui import QColor
        
        selector = DesignMethodSelector(self)
        result = selector.exec()
        
        if result:
            # Get the selected method from the dialog
            selected_row = selector.results_table.currentRow()
            if selected_row >= 0:
                method = selector.results_table.item(selected_row, 0).text()
                self.design_method_combo.setCurrentText(method)
                self.log(f" Design method updated to: {method}")
                # Update the model's design method
                self.model.design_method = method.lower()
            else:
                # Default to BoTorch as the highest rated method
                self.design_method_combo.setCurrentText("BoTorch")
                self.model.design_method = "botorch"
                self.log(" Design method updated to: BoTorch (default recommended)")
    except Exception as e:
        self.log(f" Error showing design method help: {str(e)} - Error")
        # Show a simple message box as fallback and set BoTorch as default
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Sampling Methods Information",
            "Various sampling methods are available to optimize your experimental design.\n\n"
            "BoTorch: State-of-the-art Bayesian optimization using Gaussian Processes. "
            "Excellent for continuous parameters and chemical optimization. "
            "Incorporates prior knowledge effectively and provides uncertainty quantification.\n\n"
            "TPE: Good for categorical parameters and mixed parameter types.\n\n"
            "GPEI: Best for continuous parameters, provides good uncertainty estimates.\n\n"
            "Latin Hypercube: Good space coverage for initial designs.\n\n"
            "Sobol: Excellent for high dimensional spaces.\n\n"
            "Random: Simple baseline approach for comparison."
        )
        # Set BoTorch as the default
        self.design_method_combo.setCurrentText("BoTorch")
        self.model.design_method = "botorch"
        self.log(" Design method set to: BoTorch (recommended default)") 