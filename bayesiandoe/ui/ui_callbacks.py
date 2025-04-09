from PySide6.QtWidgets import (
    QMessageBox, QTableWidgetItem, QDialog, QVBoxLayout, QGroupBox,
    QGridLayout, QLabel, QFormLayout, QTextEdit, QHBoxLayout, QPushButton,
    QListWidgetItem
)
from PySide6.QtGui import QColor
from .ui_utils import log, update_ui_from_model, update_prior_table
from .ui_visualization import update_prior_plot
from .dialogs import PriorDialog

def update_objectives(self):
    objectives = []
    weights = {}
    
    if self.obj_yield_check.isChecked():
        objectives.append("yield")
        weights["yield"] = self.weight_yield_spin.value()
    
    if self.obj_purity_check.isChecked():
        objectives.append("purity")
        weights["purity"] = self.weight_purity_spin.value()
    
    if self.obj_selectivity_check.isChecked():
        objectives.append("selectivity")
        weights["selectivity"] = self.weight_selectivity_spin.value()
    
    if not objectives:
        QMessageBox.warning(self, "Warning", "At least one objective must be selected.")
        self.obj_yield_check.setChecked(True)
        objectives.append("yield")
        weights["yield"] = self.weight_yield_spin.value()
    
    self.model.objectives = objectives
    self.model.objective_weights = weights
    
    update_ui_from_model(self)
    
    log(self, f"-- Objectives updated: {', '.join(objectives)} - Success")

def show_registry_item_tooltip(self, item, reg_type, category):
    if not item:
        return
        
    item_name = item.text()
    properties = self.registry_manager.get_item_properties(reg_type, category, item_name)
    
    if properties:
        tooltip = "\n".join([f"{k}: {v}" for k, v in properties.items() if k != "color"])
        item.setToolTip(tooltip)

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
    log(self, f"-- Best results table updated showing top {n_best} results - Success")

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
            log(self, f"-- Prior set for {param_name} (mean={dialog.result.get('mean')}, std={dialog.result.get('std')}) - Success")
        else:
            preferences = dialog.result.get("categorical_preferences", {})
            param.categorical_preferences = preferences
            log(self, f"-- Categorical prior set for {param_name} - Success")
            
        update_prior_table(self)
        if self.viz_param_combo.currentText() == param_name:
            update_prior_plot(self) 