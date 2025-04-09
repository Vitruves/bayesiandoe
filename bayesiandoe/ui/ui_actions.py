import os
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMessageBox, QFileDialog, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QTableWidgetItem,
    QInputDialog, QFormLayout, QLineEdit, QTextEdit, QGridLayout,
    QGroupBox, QSlider, QListWidgetItem, QTabWidget, QWidget,
    QTableWidget, QComboBox, QCheckBox, QHeaderView, QScrollArea,
    QFrame, QRadioButton
)
from PySide6.QtGui import QColor, QFont, QPixmap, QImage

from ..parameters import ChemicalParameter
from ..core import OptunaBayesianExperiment, _calculate_parameter_distance
from .ui_utils import (
    log, update_ui_from_model, update_best_result_label, 
    update_parameter_combos
)
from .ui_callbacks import refresh_registry, update_best_results
from .dialogs import ResultDialog, OptimizationSettingsDialog, PriorDialog, ParameterDialog
from .canvas import MplCanvas

# Import functions needed for the visualization functions
import matplotlib.pyplot as plt
from .ui_visualization import update_results_plot

def add_parameter(self):
    dialog = ParameterDialog(self)
    if dialog.exec():
        param = dialog.result
        
        if not param:
            return
            
        if param.name in self.model.parameters:
            QMessageBox.warning(self, "Warning", f"Parameter '{param.name}' already exists")
            return
            
        self.model.add_parameter(param)
        self.param_table.update_from_model(self.model)
        update_parameter_combos(self)
        self.experiment_table.update_columns(self.model)
        self.best_table.update_columns(self.model)
        
        log(self, f"-- Parameter '{param.name}' added - Success")

def edit_parameter(self):
    selected_items = self.param_table.selectedItems()
    if not selected_items:
        QMessageBox.information(self, "Info", "Please select a parameter to edit")
        return
        
    row = selected_items[0].row()
    name = self.param_table.item(row, 0).text()
    
    if name not in self.model.parameters:
        QMessageBox.critical(self, "Error", f"Parameter '{name}' not found")
        return
        
    param = self.model.parameters[name]
    
    from .dialogs import ParameterDialog
    dialog = ParameterDialog(self, param, edit_mode=True)
    if dialog.exec():
        result = dialog.result
        
        if not result:
            return
            
        if param.param_type == result.param_type:
            if param.param_type in ["continuous", "discrete"]:
                param.low = result.low
                param.high = result.high
            elif param.param_type == "categorical":
                param.choices = result.choices
                
            param.units = result.units
            
            self.param_table.update_from_model(self.model)
            log(self, f"-- Parameter '{name}' updated - Success")

def remove_parameter(self):
    selected_items = self.param_table.selectedItems()
    if not selected_items:
        QMessageBox.information(self, "Info", "Please select a parameter to remove")
        return
        
    row = selected_items[0].row()
    name = self.param_table.item(row, 0).text()
    
    confirm = QMessageBox.question(
        self, "Confirm", f"Are you sure you want to remove the parameter '{name}'?",
        QMessageBox.Yes | QMessageBox.No
    )
    
    if confirm == QMessageBox.Yes:
        self.model.remove_parameter(name)
        self.param_table.update_from_model(self.model)
        update_parameter_combos(self)
        self.experiment_table.update_columns(self.model)
        self.best_table.update_columns(self.model)
        
        log(self, f"-- Parameter '{name}' removed - Success")

def add_from_registry(self, reg_type):
    current_tab_index = self.registry_tabs.currentIndex()
    if current_tab_index < 0 or reg_type != self.registry_manager.get_all_types()[current_tab_index]:
        QMessageBox.warning(self, "Warning", "Please select a registry tab first.")
        return
    
    categories = self.registry_lists[reg_type]
    
    for category, list_widget in categories.items():
        selected_items = list_widget.selectedItems()
        
        for item in selected_items:
            item_name = item.text()
            param_name = f"{category.lower()}_{item_name}"
            
            if reg_type == "solvents":
                param = ChemicalParameter(
                    name="Solvent",
                    param_type="categorical",
                    choices=[item_name]
                )
                
                if "Solvent" in self.model.parameters:
                    existing_choices = self.model.parameters["Solvent"].choices
                    if item_name not in existing_choices:
                        updated_choices = existing_choices + [item_name]
                        self.model.remove_parameter("Solvent")
                        param = ChemicalParameter(
                            name="Solvent",
                            param_type="categorical",
                            choices=updated_choices
                        )
                    else:
                        continue
            
            elif reg_type == "catalysts":
                param = ChemicalParameter(
                    name="Catalyst",
                    param_type="categorical",
                    choices=[item_name]
                )
                
                if "Catalyst" in self.model.parameters:
                    existing_choices = self.model.parameters["Catalyst"].choices
                    if item_name not in existing_choices:
                        updated_choices = existing_choices + [item_name]
                        self.model.remove_parameter("Catalyst")
                        param = ChemicalParameter(
                            name="Catalyst",
                            param_type="categorical",
                            choices=updated_choices
                        )
                    else:
                        continue
            
            elif reg_type == "ligands":
                param = ChemicalParameter(
                    name="Ligand",
                    param_type="categorical",
                    choices=[item_name]
                )
                
                if "Ligand" in self.model.parameters:
                    existing_choices = self.model.parameters["Ligand"].choices
                    if item_name not in existing_choices:
                        updated_choices = existing_choices + [item_name]
                        self.model.remove_parameter("Ligand")
                        param = ChemicalParameter(
                            name="Ligand",
                            param_type="categorical",
                            choices=updated_choices
                        )
                    else:
                        continue
            
            else:
                if param_name in self.model.parameters:
                    QMessageBox.warning(self, "Warning", f"Parameter '{param_name}' already exists.")
                    continue
                
                param = ChemicalParameter(
                    name=param_name,
                    param_type="categorical",
                    choices=[item_name]
                )
            
            self.model.add_parameter(param)
    
    self.param_table.update_from_model(self.model)
    update_parameter_combos(self)
    self.experiment_table.update_columns(self.model)
    self.best_table.update_columns(self.model)
    
    log(self, f"-- Added parameters from {reg_type} registry - Success")

def add_to_registry(self, reg_type):
    tab_widget = None
    for rt, categories in self.registry_lists.items():
        if rt == reg_type:
            tab_index = list(categories.keys()).index(list(categories.keys())[0])
            tab_widget = self.registry_tabs.widget(self.registry_tabs.currentIndex()).findChild(QTabWidget)
            break
    
    if not tab_widget:
        QMessageBox.warning(self, "Warning", "Could not find the registry tab.")
        return
    
    current_category = list(self.registry_lists[reg_type].keys())[tab_widget.currentIndex()]
    
    item_name, ok = QInputDialog.getText(self, "Add Item", f"Enter new {reg_type} name:")
    if not ok or not item_name.strip():
        return
        
    properties_dialog = QDialog(self)
    properties_dialog.setWindowTitle(f"Properties for {item_name}")
    properties_dialog.resize(400, 300)
    
    layout = QVBoxLayout(properties_dialog)
    form_layout = QFormLayout()
    
    property_inputs = {}
    
    if reg_type == "solvents":
        property_inputs["bp"] = QLineEdit()
        property_inputs["ε"] = QLineEdit()
        property_inputs["log P"] = QLineEdit()
        
        form_layout.addRow("Boiling Point (°C):", property_inputs["bp"])
        form_layout.addRow("Dielectric Constant (ε):", property_inputs["ε"])
        form_layout.addRow("Log P:", property_inputs["log P"])
        
    elif reg_type == "catalysts":
        property_inputs["loading"] = QLineEdit("1-5 mol%")
        property_inputs["activation"] = QLineEdit()
        property_inputs["cost"] = QLineEdit()
        
        form_layout.addRow("Loading:", property_inputs["loading"])
        form_layout.addRow("Activation:", property_inputs["activation"])
        form_layout.addRow("Cost:", property_inputs["cost"])
        
    elif reg_type == "ligands":
        property_inputs["type"] = QLineEdit()
        property_inputs["bite angle"] = QLineEdit()
        property_inputs["donor"] = QLineEdit()
        
        form_layout.addRow("Type:", property_inputs["type"])
        form_layout.addRow("Bite Angle:", property_inputs["bite angle"])
        form_layout.addRow("Donor Atoms:", property_inputs["donor"])
        
    layout.addLayout(form_layout)
    
    button_box = QHBoxLayout()
    ok_button = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    
    ok_button.clicked.connect(properties_dialog.accept)
    cancel_button.clicked.connect(properties_dialog.reject)
    
    button_box.addWidget(ok_button)
    button_box.addWidget(cancel_button)
    
    layout.addLayout(button_box)
    
    if properties_dialog.exec() == QDialog.Accepted:
        properties = {}
        for prop_name, input_widget in property_inputs.items():
            value = input_widget.text().strip()
            if value:
                properties[prop_name] = value
        
        success = self.registry_manager.add_item(reg_type, current_category, item_name, properties)
        
        if success:
            self.refresh_registry()
            log(self, f"-- Added {item_name} to {reg_type}/{current_category} registry - Success")
        else:
            QMessageBox.warning(self, "Warning", f"Could not add {item_name}. It may already exist.")

def remove_from_registry(self, reg_type):
    tab_widget = None
    for rt, categories in self.registry_lists.items():
        if rt == reg_type:
            tab_widget = self.registry_tabs.widget(self.registry_tabs.currentIndex()).findChild(QTabWidget)
            break
    
    if not tab_widget:
        QMessageBox.warning(self, "Warning", "Could not find the registry tab.")
        return
    
    current_category = list(self.registry_lists[reg_type].keys())[tab_widget.currentIndex()]
    list_widget = self.registry_lists[reg_type][current_category]
    
    selected_items = list_widget.selectedItems()
    if not selected_items:
        QMessageBox.information(self, "Information", "No items selected to remove.")
        return
    
    confirm = QMessageBox.question(
        self, 
        "Confirm Removal", 
        f"Remove {len(selected_items)} item(s) from the registry?",
        QMessageBox.Yes | QMessageBox.No, 
        QMessageBox.No
    )
    
    if confirm == QMessageBox.Yes:
        for item in selected_items:
            item_name = item.text()
            success = self.registry_manager.remove_item(reg_type, current_category, item_name)
            
            if success:
                log(self, f"-- Removed {item_name} from {reg_type}/{current_category} registry - Success")
            else:
                log(self, f"-- Failed to remove {item_name} from registry - Failed")
        
        self.refresh_registry()

def load_template(self, template_name):
    from ..parameters import ChemicalParameter
    from PySide6.QtWidgets import QMessageBox
    from .ui_utils import update_parameter_combos, log
    
    if template_name == "reaction_conditions":
        params = [
            ChemicalParameter(name="Temperature", param_type="continuous", low=25, high=100, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=1, high=16, units="h"),
            ChemicalParameter(name="Concentration", param_type="continuous", low=0.1, high=1.0, units="M"),
            ChemicalParameter(name="Solvent", param_type="categorical", choices=["Toluene", "THF", "DCM", "MeOH", "DMSO", "DMF"]),
            ChemicalParameter(name="Atmosphere", param_type="categorical", choices=["Air", "N₂", "Ar"])
        ]
    elif template_name == "catalyst":
        params = [
            ChemicalParameter(name="Catalyst", param_type="categorical", 
                choices=["Pd(OAc)₂", "Pd(PPh₃)₄", "Pd₂(dba)₃", "PdCl₂(PPh₃)₂", "Pd(dppf)Cl₂", "Ni(COD)₂", "NiCl₂·DME"]),
            ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.05, high=0.5, units="mol%"),
            ChemicalParameter(name="Ligand", param_type="categorical", 
                choices=["PPh₃", "P(t-Bu)₃", "BINAP", "XPhos", "SPhos", "dppf", "Xantphos", "None"]),
            ChemicalParameter(name="Ligand:Catalyst Ratio", param_type="continuous", low=0.5, high=5, units="L:M"),
            ChemicalParameter(name="Substrate A", param_type="continuous", low=1, high=1.0001, units="equiv"),
            ChemicalParameter(name="Substrate B", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Temperature", param_type="continuous", low=25, high=100, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=1, high=16, units="h"),
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["Toluene", "THF", "DCM", "DMF", "DMSO", "MeCN", "MeOH", "EtOH", "H₂O"])
        ]
    elif template_name == "solvent":
        params = [
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["Toluene", "THF", "DCM", "DMF", "DMSO", "MeCN", "MeOH", "EtOH", "H₂O", 
                         "Hexane", "Acetone", "Diethyl Ether", "1,4-Dioxane"]),
            ChemicalParameter(name="Temperature", param_type="continuous", low=25, high=100, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=1, high=16, units="h")
        ]
    elif template_name == "cross_coupling":
        params = [
            ChemicalParameter(name="Catalyst", param_type="categorical", 
                choices=["Pd(OAc)₂", "Pd(PPh₃)₄", "Pd₂(dba)₃", "PdCl₂(PPh₃)₂", "Pd(dppf)Cl₂"]),
            ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.05, high=0.5, units="mol%"),
            ChemicalParameter(name="Ligand", param_type="categorical", 
                choices=["PPh₃", "P(t-Bu)₃", "BINAP", "XPhos", "SPhos", "dppf", "Xantphos", "None"]),
            ChemicalParameter(name="Ligand:Catalyst Ratio", param_type="continuous", low=0.5, high=5, units="L:M"),
            ChemicalParameter(name="Base", param_type="categorical", 
                choices=["K₂CO₃", "Cs₂CO₃", "K₃PO₄", "NaOt-Bu", "NEt₃", "KOAc", "K₃PO₄"]),
            ChemicalParameter(name="Base Equiv", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Substrate A", param_type="continuous", low=1, high=1.0001, units="equiv"),
            ChemicalParameter(name="Substrate B", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Temperature", param_type="continuous", low=25, high=100, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=1, high=16, units="h"),
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["Toluene", "THF", "DCM", "DMF", "DMSO", "MeCN", "MeOH", "EtOH", "H₂O", "1,4-Dioxane"])
        ]
    elif template_name == "oxidation":
        params = [
            ChemicalParameter(name="Oxidant", param_type="categorical", 
                choices=["H₂O₂", "m-CPBA", "DMDO", "TBHP", "NaOCl", "O₂", "O₂/catalyst", "OsO₄/NMO"]),
            ChemicalParameter(name="Oxidant Equiv", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Catalyst", param_type="categorical", 
                choices=["None", "VO(acac)₂", "Mn(OAc)₃", "Fe(acac)₃", "RuCl₃", "KMnO₄", "SeO₂"]),
            ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.05, high=0.5, units="mol%"),
            ChemicalParameter(name="Additive", param_type="categorical", 
                choices=["None", "AcOH", "TFA", "Base", "Phase transfer catalyst"]),
            ChemicalParameter(name="Substrate", param_type="continuous", low=1, high=1.0001, units="equiv"),
            ChemicalParameter(name="Temperature", param_type="continuous", low=0, high=75, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=0.5, high=16, units="h"),
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["DCM", "MeOH", "H₂O", "MeOH/H₂O", "DCM/H₂O", "Acetone", "AcOH"])
        ]
    elif template_name == "reduction":
        params = [
            ChemicalParameter(name="Reductant", param_type="categorical", 
                choices=["NaBH₄", "LiAlH₄", "DIBAL-H", "BH₃·THF", "H₂/Pd-C", "H₂/Pt", "H₂/Raney Ni", "Na/NH₃(l)"]),
            ChemicalParameter(name="Reductant Equiv", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Catalyst", param_type="categorical", 
                choices=["None", "Pd/C", "Pt/C", "Raney Ni", "RuCl₃", "Rh/Al₂O₃", "PtO₂"]),
            ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.05, high=0.5, units="mol%"),
            ChemicalParameter(name="Temperature", param_type="continuous", low=0, high=100, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=1, high=16, units="h"),
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["THF", "Et₂O", "MeOH", "EtOH", "EtOAc", "DCM", "MeOH/DCM"])
        ]
    elif template_name == "amide_coupling":
        params = [
            ChemicalParameter(name="Coupling Agent", param_type="categorical", 
                choices=["EDC·HCl", "DCC", "PyBOP", "HATU", "HBTU", "T3P", "CDI", "SOCl₂"]),
            ChemicalParameter(name="Coupling Agent Equiv", param_type="continuous", low=1, high=2, units="equiv"),
            ChemicalParameter(name="Additive", param_type="categorical", 
                choices=["None", "HOBt", "HOAt", "NHS", "DMAP", "Oxyma"]),
            ChemicalParameter(name="Base", param_type="categorical", 
                choices=["DIPEA", "NEt₃", "NMM", "DMAP", "Pyridine", "K₂CO₃"]),
            ChemicalParameter(name="Base Equiv", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Acid", param_type="continuous", low=1, high=1.0001, units="equiv"),
            ChemicalParameter(name="Amine", param_type="continuous", low=1, high=3, units="equiv"),  
            ChemicalParameter(name="Temperature", param_type="continuous", low=0, high=50, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=1, high=16, units="h"),
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["DCM", "DMF", "THF", "MeCN", "CHCl₃"])
        ]
    elif template_name == "organocatalysis":
        params = [
            ChemicalParameter(name="Catalyst", param_type="categorical", 
                choices=["L-Proline", "MacMillan catalyst", "Cinchona alkaloid", "Thiourea", "Phosphoric acid", "DMAP", "Imidazole"]),
            ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.05, high=0.5, units="mol%"),
            ChemicalParameter(name="Additive", param_type="categorical", 
                choices=["None", "AcOH", "TFA", "Benzoic acid", "Water", "MgCl₂", "ZnCl₂"]),
            ChemicalParameter(name="Substrate A", param_type="continuous", low=1, high=1.0001, units="equiv"),
            ChemicalParameter(name="Substrate B", param_type="continuous", low=1, high=5, units="equiv"),
            ChemicalParameter(name="Temperature", param_type="continuous", low=-20, high=60, units="°C"),
            ChemicalParameter(name="Time", param_type="continuous", low=4, high=96, units="h"),
            ChemicalParameter(name="Solvent", param_type="categorical", 
                choices=["DCM", "DMSO", "Toluene", "MeOH", "THF", "DMF", "Water", "CHCl₃"])
        ]
        
    else:
        QMessageBox.warning(self, "Warning", f"Template '{template_name}' not found.")
        return
    
    if self.model.parameters:
        confirm = QMessageBox.question(
            self, 
            "Confirm Template Load", 
            "Loading this template will replace all current parameters. Continue?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
            
        # Clear existing parameters
        self.model.parameters = {}
    
    # Add each parameter from the template
    try:
        for param in params:
            self.model.add_parameter(param)
            
        # Update UI elements
        self.param_table.update_from_model(self.model)
        update_parameter_combos(self)
        
        # Make sure experiment table and best results table are updated with new parameters
        if hasattr(self, 'experiment_table'):
            self.experiment_table.update_columns(self.model)
        if hasattr(self, 'best_table'):
            self.best_table.update_columns(self.model)
        
        # Log success message
        log(self, f"-- Loaded {template_name} template with {len(params)} parameters - Success")
    except Exception as e:
        import traceback
        error_msg = f"Error loading template: {str(e)}"
        traceback.print_exc()
        log(self, f"-- {error_msg} - Error")
        QMessageBox.critical(self, "Error", error_msg)

def add_substrate_parameter(self):
    substrate_name, ok = QInputDialog.getText(self, "Add Substrate", "Enter substrate name:")
    
    if ok and substrate_name.strip():
        smiles, ok = QInputDialog.getText(self, "Add SMILES (Optional)", 
                                      "Enter SMILES string (optional):", 
                                      text="")
        
        param = ChemicalParameter(
            name="Substrate",
            param_type="categorical",
            choices=[substrate_name.strip()]
        )
        
        if ok and smiles.strip():
            param.smiles = smiles.strip()
        
        if "Substrate" in self.model.parameters:
            existing_choices = self.model.parameters["Substrate"].choices
            if substrate_name.strip() not in existing_choices:
                updated_choices = existing_choices + [substrate_name.strip()]
                
                self.model.remove_parameter("Substrate")
                
                param = ChemicalParameter(
                    name="Substrate",
                    param_type="categorical",
                    choices=updated_choices
                )
                
        self.model.add_parameter(param)
        self.param_table.update_from_model(self.model)
        update_parameter_combos(self)
        self.experiment_table.update_columns(self.model)
        self.best_table.update_columns(self.model)
        
        log(self, f"-- Substrate '{substrate_name.strip()}' added - Success")

def generate_initial_experiments(self):
    try:
        n_experiments = self.n_initial_spin.value()
        
        if not self.model.parameters:
            self.log("-- No parameters defined. Add parameters first - Error")
            return
            
        if not self.model.objectives:
            self.log("-- No objectives defined. Define objectives first - Error")
            return
            
        # Generate initial experiments
        method = self.design_method_combo.currentText().lower()
        self.log(f"-- Using {method} design method")
        
        # Update model's design method
        self.model.design_method = method
        
        # Clear any existing planned experiments
        if hasattr(self.model, 'planned_experiments'):
            self.model.planned_experiments = []
        else:
            self.model.planned_experiments = []
            
        # Reset round indices
        self.round_start_indices = []
        
        # Call the appropriate suggestion method based on design method
        if method == "botorch":
            suggestions = self.model._suggest_with_botorch(n_experiments)
        elif method == "tpe":
            suggestions = self.model._suggest_with_tpe(n_experiments)
        elif method == "gpei":
            suggestions = self.model._suggest_with_gp(n_experiments)
        elif method == "random":
            suggestions = self.model._suggest_random(n_experiments)
        elif method == "latin hypercube":
            suggestions = self.model._suggest_with_lhs(n_experiments)
        elif method == "sobol":
            suggestions = self.model._suggest_with_sobol(n_experiments)
        else:
            # Default to BoTorch if method not recognized
            self.log(f"-- Unknown design method '{method}', using BoTorch - Warning")
            suggestions = self.model._suggest_with_botorch(n_experiments)
            
        self.model.planned_experiments = suggestions
        self.current_round = 1
        self.current_round_label.setText("1")
        
        # Update the UI
        update_ui_from_model(self)
        
        self.log(f"-- Generated {n_experiments} initial experiments with {method.upper()} sampling - Success")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        self.log(f"-- Failed to generate experiments: {str(e)} - Error")

def generate_next_experiments(self):
    """Generate the next round of experiments"""
    from ..core import settings
    
    # Get number of new experiments to generate
    n_next = self.n_next_spin.value()
    
    # Set exploitation weight based on slider
    exploitation_weight = self.exploit_slider.value() / 100.0
    self.model.exploitation_weight = exploitation_weight
    
    try:
        # Store initial count for tracking
        initial_count = len(self.model.planned_experiments)
        print(f"Before adding: {initial_count} planned experiments")
        
        # Generate new suggestions
        method = self.design_method_combo.currentText().lower()
        
        # Get start time for performance measurement
        import time
        start_time = time.time()
        
        # Call the appropriate suggestion method based on design method
        if method == "botorch":
            suggestions = self.model._suggest_with_botorch(n_next)
        elif method == "tpe":
            suggestions = self.model._suggest_with_tpe(n_next)
        elif method == "gpei":
            suggestions = self.model._suggest_with_gp(n_next)
        elif method == "random":
            suggestions = self.model._suggest_random(n_next)
        elif method == "latin hypercube":
            suggestions = self.model._suggest_with_lhs(n_next)
        elif method == "sobol":
            suggestions = self.model._suggest_with_sobol(n_next)
        else:
            # Default to BoTorch if method not recognized
            self.log(f"-- Unknown design method '{method}', using BoTorch - Warning")
            suggestions = self.model._suggest_with_botorch(n_next)
            
        # Add suggestions to planned experiments
        self.model.planned_experiments.extend(suggestions)
        
        # Log timing
        elapsed = time.time() - start_time
        self.log(f"-- Generated {n_next} suggestions in {elapsed:.2f}s")
        
        final_count = len(self.model.planned_experiments)
        print(f"After adding: {final_count} planned experiments")
        
        # If suggestions were returned, update UI
        if suggestions and len(suggestions) > 0:
            print(f"Generated {len(suggestions)} new suggestions")
            
            # Update round start indices - record the index where the new round starts
            self.round_start_indices.append(initial_count)
            
            # Increment round number
            self.current_round += 1
            self.current_round_label.setText(str(self.current_round))
            
            # Update experiment table
            self.experiment_table.update_from_planned(self.model, self.round_start_indices)
            
            log(self, f"-- Generated {n_next} experiments for round {self.current_round} - Success")
        else:
            # Handle case where no suggestions were returned
            log(self, f"-- No experiments generated. Check parameters and try again - Warning")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(self, f"-- Failed to generate experiments: {str(e)} - Error")
        
        # Show error dialog to user with helpful message
        from PySide6.QtWidgets import QMessageBox
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error Generating Experiments")
        error_box.setText("Failed to generate next experiments")
        
        # Provide a more specific message based on the error type
        if "X_pending" in str(e):
            error_box.setInformativeText(
                "The optimization algorithm encountered an issue with the BoTorch library. "
                "This is likely due to version compatibility. Trying with a different method."
            )
        else:
            error_box.setInformativeText("The optimization algorithm encountered an error. Let's try a different approach.")
            
        error_box.setDetailedText(error_details)
        error_box.exec()

def add_result_for_selected(self):
    """Add result for selected experiment with enhanced error detection and reporting."""
    try:
        # Print debug info about table
        print(f"Experiment table has {self.experiment_table.rowCount()} rows")
        
        selected_items = self.experiment_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select an experiment row first.")
            return
            
        # Get the row number of the first selected item
        row = selected_items[0].row()
        print(f"Selected row: {row}")
        
        # Check if this is a separator row by checking column span
        if self.experiment_table.columnSpan(row, 0) > 1:
            QMessageBox.warning(self, "Warning", "You selected a round separator. Please select an experiment row.")
            return
            
        # Make sure we have both row and ID columns populated
        round_item = self.experiment_table.item(row, 0)
        id_item = self.experiment_table.item(row, 1)
        
        if not round_item or not id_item:
            QMessageBox.warning(self, "Warning", "Selected row is missing critical data.")
            return
            
        print(f"Round: {round_item.text() if round_item else 'None'}, ID: {id_item.text() if id_item else 'None'}")
        
        if not id_item or not id_item.text().isdigit():
            QMessageBox.warning(self, "Warning", "Selected row doesn't have a valid experiment ID.")
            return
            
        exp_id = int(id_item.text()) - 1
        print(f"Experiment ID: {exp_id+1}")
        
        # Make sure experiment exists in planned experiments
        if exp_id < 0 or exp_id >= len(self.model.planned_experiments):
            QMessageBox.warning(self, "Warning", f"Experiment #{exp_id+1} doesn't exist in planned experiments.")
            return
            
        # Get parameters for this experiment
        exp_params = self.model.planned_experiments[exp_id]
        print(f"Experiment parameters: {exp_params}")
            
        # Check if result already exists
        existing_exp = None
        for i, exp in enumerate(self.model.experiments):
            if 'params' not in exp:
                continue
                
            # Count matching parameters
            matching_params = 0
            total_params = 0
            for k, v in exp_params.items():
                if k in exp['params']:
                    total_params += 1
                    if isinstance(v, float):
                        # For floats, allow small differences
                        if abs(float(exp['params'][k]) - float(v)) < 1e-6:
                            matching_params += 1
                    elif exp['params'][k] == v:
                        matching_params += 1
            
            # If all parameters match, consider it the same experiment
            if matching_params == total_params and total_params > 0:
                existing_exp = exp
                print(f"Found existing result (experiment #{i+1})")
                break
        
        if existing_exp:
            confirm = QMessageBox.question(self, "Confirm", 
                            f"Result already exists for experiment #{exp_id+1}. Replace it?",
                            QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.No:
                return
                
            # Remove existing result
            self.model.experiments.remove(existing_exp)
            print("Removed existing result")
            
        # Show dialog to input results
        dialog = ResultDialog(self, self.model, exp_id, exp_params)
        if dialog.exec() == QDialog.Accepted and dialog.result:
            # Add result to model
            self.model.experiments.append(dialog.result)
            print(f"Added new result: {dialog.result}")
            
            # Calculate score
            results = dialog.result['results']
            score = self.model._calculate_composite_score(results)
            dialog.result['score'] = score
            print(f"Calculated score: {score}")
            
            # Update experiment table and highlight result
            self.experiment_table.update_from_planned(self.model, self.round_start_indices)
            
            # Update all result tables using the central update method
            self.update_result_tables()
            
            # Log success
            self.log(f"-- Added result for experiment #{exp_id+1} - Success")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        self.log(f"-- Failed to add result: {str(e)} - Error")
        print(f"Error details: {error_details}")

def new_project(self):
    confirm = QMessageBox.question(
        self, 
        "Confirm New Project", 
        "Creating a new project will discard all current data. Continue?",
        QMessageBox.Yes | QMessageBox.No, 
        QMessageBox.No
    )
    
    if confirm == QMessageBox.Yes:
        self.model = OptunaBayesianExperiment()
        self.current_round = 0
        self.round_start_indices = []
        self.working_directory = os.getcwd()
        
        # Explicitly update columns on tables first
        self.experiment_table.update_columns(self.model)
        if hasattr(self, 'best_table'):
            self.best_table.update_columns(self.model)
            
        update_ui_from_model(self)
        self.tab_widget.setCurrentIndex(0)
        log(self, "-- New project created - Success")

def open_project(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self,
        "Open Project",
        self.working_directory,
        "Bayesian DOE Project Files (*.bdoe);;All Files (*)"
    )
    
    if file_path:
        try:
            self.model = OptunaBayesianExperiment()
            self.model.load_model(file_path)
            
            self.working_directory = os.path.dirname(file_path)
            
            self.current_round = 0
            self.round_start_indices = []
            
            if self.model.experiments:
                rounds_per_experiment = 5
                self.current_round = (len(self.model.experiments) - 1) // rounds_per_experiment + 1
                
                for i in range(1, self.current_round + 1):
                    self.round_start_indices.append((i - 1) * rounds_per_experiment)
            
            update_ui_from_model(self)
            log(self, f"-- Project loaded from {file_path} - Success")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project: {str(e)}")
            log(self, f"-- Error loading project: {str(e)} - Failed")

def save_project(self):
    if not self.model.parameters:
        QMessageBox.warning(self, "Warning", "No parameters defined. Define parameters before saving.")
        return
        
    file_path, _ = QFileDialog.getSaveFileName(
        self,
        "Save Project",
        self.working_directory,
        "Bayesian DOE Project Files (*.bdoe);;All Files (*)"
    )
    
    if file_path:
        if not file_path.lower().endswith('.bdoe'):
            file_path += '.bdoe'
            
        try:
            self.model.save_model(file_path)
            
            self.working_directory = os.path.dirname(file_path)
            
            log(self, f"-- Project saved to {file_path} - Success")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
            log(self, f"-- Error saving project: {str(e)} - Failed")

def import_data(self):
    filepath, _ = QFileDialog.getOpenFileName(
        self,
        "Import Data",
        self.working_directory,
        "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
    )
    
    if not filepath:
        return
        
    try:
        log(self, f"-- Importing data from {filepath}")
        self.progress_bar.setValue(10)
        
        if filepath.lower().endswith(".xlsx"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
            
        self.progress_bar.setValue(30)
        
        required_obj = self.model.objectives[0] if self.model.objectives else "yield"
        param_columns = [col for col in df.columns if col in self.model.parameters]
        
        if not param_columns:
            QMessageBox.warning(self, "Warning", 
                            "Could not find any parameter columns matching the current parameters")
            self.progress_bar.setValue(0)
            return
            
        objective_columns = []
        for obj in self.model.objectives:
            candidates = [
                obj,
                obj.capitalize(),
                f"{obj} (%)",
                f"{obj.capitalize()} (%)"
            ]
            
            for candidate in candidates:
                if candidate in df.columns:
                    objective_columns.append((obj, candidate))
                    break
                    
        if not objective_columns:
            QMessageBox.warning(self, "Warning", 
                            "Could not find any objective columns matching the current objectives")
            self.progress_bar.setValue(0)
            return
            
        self.progress_bar.setValue(50)
        
        imported_count = 0
        for _, row in df.iterrows():
            params = {}
            for param_name in param_columns:
                if pd.notnull(row[param_name]):
                    param = self.model.parameters[param_name]
                    value = row[param_name]
                    
                    if param.param_type == "continuous":
                        value = float(value)
                    elif param.param_type == "discrete":
                        value = int(value)
                    
                    params[param_name] = value
            
            results = {}
            for obj, col_name in objective_columns:
                if pd.notnull(row[col_name]):
                    value = float(row[col_name])
                    if value > 1.0:
                        value /= 100.0
                    results[obj] = value
            
            if params and results:
                self.model.add_experiment_result(params, results)
                imported_count += 1
                
        self.progress_bar.setValue(80)
        
        self.experiment_table.update_from_planned(self.model, self.round_start_indices)
        self.best_table.update_from_model(self.model, self.n_best_spin.value())
        self.all_results_table.update_from_model(self.model)
        
        completed_count = len(self.model.experiments)
        planned_count = len(self.model.planned_experiments)
        self.results_count_label.setText(f"{completed_count} / {planned_count}")
        
        best_exps = self.model.get_best_experiments(n=1)
        if best_exps:
            best_exp = best_exps[0]
            best_score = best_exp.get('score', 0) * 100.0
            self.best_result_label.setText(f"{best_score:.2f}%")
            
        self.progress_bar.setValue(100)
        log(self, f"-- Imported {imported_count} experiments from {filepath} - Success")
        
        QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))
        
    except Exception as e:
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Error", f"Error importing data: {str(e)}")
        log(self, f"-- Data import failed: {str(e)} - Error")

def export_results(self):
    if not self.model.experiments:
        QMessageBox.warning(self, "Warning", "No experiments to export")
        return
        
    filepath, filter_used = QFileDialog.getSaveFileName(
        self,
        "Export Results",
        os.path.join(self.working_directory, "results.csv"),
        "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
    )
    
    if not filepath:
        return
        
    try:
        log(self, f"-- Exporting results to {filepath}")
        self.progress_bar.setValue(10)
        
        from ..core import settings
        
        param_names = list(self.model.parameters.keys())
        objective_names = self.model.objectives
        
        data = []
        for i, exp in enumerate(self.model.experiments):
            row = {
                "Experiment ID": i + 1,
                "Round": (i // 5) + 1,
                "Timestamp": exp.get("timestamp", "").replace("T", " ")
            }
            
            params = exp.get("params", {})
            for param in param_names:
                if param in params:
                    value = params[param]
                    if isinstance(value, float):
                        value = settings.format_value(value)
                    row[param] = value
                else:
                    row[param] = ""
            
            results = exp.get("results", {})
            for obj in objective_names:
                if obj in results and results[obj] is not None:
                    value = results[obj] * 100.0
                    row[f"{obj.capitalize()} (%)"] = settings.format_value(value)
                else:
                    row[f"{obj.capitalize()} (%)"] = ""
            
            if "score" in exp:
                row["Composite Score (%)"] = settings.format_value(exp["score"] * 100.0)
            
            data.append(row)
        
        self.progress_bar.setValue(50)
        
        df = pd.DataFrame(data)
        
        if filepath.lower().endswith(".xlsx"):
            df.to_excel(filepath, index=False)
        else:
            if not filepath.lower().endswith(".csv"):
                filepath += ".csv"
            df.to_csv(filepath, index=False)
        
        self.progress_bar.setValue(100)
        log(self, f"-- Results exported to {filepath} - Success")
        
        QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))
        
        self.working_directory = os.path.dirname(filepath)
        
    except Exception as e:
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
        log(self, f"-- Results export failed: {str(e)} - Error")

def statistical_analysis(self):
    if not self.model.experiments:
        QMessageBox.information(self, "Information", "No experiment data available for analysis.")
        return
    
    analysis_dialog = QDialog(self)
    analysis_dialog.setWindowTitle("Statistical Analysis")
    analysis_dialog.resize(800, 600)
    
    layout = QVBoxLayout(analysis_dialog)
    
    tabs = QTabWidget()
    
    summary_tab = QWidget()
    summary_layout = QVBoxLayout(summary_tab)
    
    summary_text = QTextEdit()
    summary_text.setReadOnly(True)
    summary_layout.addWidget(summary_text)
    
    generate_summary_btn = QPushButton("Generate Summary Statistics")
    generate_summary_btn.clicked.connect(lambda: generate_summary())
    summary_layout.addWidget(generate_summary_btn)
    
    tabs.addTab(summary_tab, "Summary Statistics")
    
    regression_tab = QWidget()
    regression_layout = QVBoxLayout(regression_tab)
    
    control_layout = QHBoxLayout()
    control_layout.addWidget(QLabel("Target:"))
    
    target_combo = QComboBox()
    target_combo.addItems(self.model.objectives)
    control_layout.addWidget(target_combo)
    
    run_regression_btn = QPushButton("Run Regression Analysis")
    run_regression_btn.clicked.connect(lambda: run_regression())
    control_layout.addWidget(run_regression_btn)
    
    regression_layout.addLayout(control_layout)
    
    regression_result = QTextEdit()
    regression_result.setReadOnly(True)
    regression_layout.addWidget(regression_result)
    
    tabs.addTab(regression_tab, "Regression Analysis")
    
    anova_tab = QWidget()
    anova_layout = QVBoxLayout(anova_tab)
    
    anova_control = QHBoxLayout()
    anova_control.addWidget(QLabel("Response:"))
    
    response_combo = QComboBox()
    response_combo.addItems(self.model.objectives)
    anova_control.addWidget(response_combo)
    
    anova_control.addWidget(QLabel("Factor:"))
    factor_combo = QComboBox()
    factor_combo.addItems(list(self.model.parameters.keys()))
    anova_control.addWidget(factor_combo)
    
    run_anova_btn = QPushButton("Run ANOVA")
    run_anova_btn.clicked.connect(lambda: run_anova())
    anova_control.addWidget(run_anova_btn)
    
    anova_layout.addLayout(anova_control)
    
    anova_result = QTextEdit()
    anova_result.setReadOnly(True)
    anova_layout.addWidget(anova_result)
    
    tabs.addTab(anova_tab, "ANOVA")
    
    layout.addWidget(tabs)
    
    def generate_summary():
        summary_text.clear()
        
        data = []
        for exp in self.model.experiments:
            row = {}
            for param_name in self.model.parameters:
                if param_name in exp['params']:
                    row[param_name] = exp['params'][param_name]
            
            for obj in self.model.objectives:
                if obj in exp['results']:
                    row[obj] = exp['results'][obj] * 100.0
            
            data.append(row)
        
        if not data:
            summary_text.setText("No data available for analysis.")
            return
        
        df = pd.DataFrame(data)
        
        summary = "# Summary Statistics\n\n"
        summary += "## Objectives\n\n"
        
        for obj in self.model.objectives:
            if obj in df.columns:
                summary += f"### {obj.capitalize()}\n"
                summary += f"- Count: {df[obj].count()}\n"
                summary += f"- Mean: {df[obj].mean():.2f}%\n"
                summary += f"- Std Dev: {df[obj].std():.2f}%\n"
                summary += f"- Min: {df[obj].min():.2f}%\n"
                summary += f"- 25%: {df[obj].quantile(0.25):.2f}%\n"
                summary += f"- Median: {df[obj].median():.2f}%\n"
                summary += f"- 75%: {df[obj].quantile(0.75):.2f}%\n"
                summary += f"- Max: {df[obj].max():.2f}%\n\n"
        
        summary += "## Parameters\n\n"
        
        for param_name, param in self.model.parameters.items():
            if param_name in df.columns:
                summary += f"### {param_name}\n"
                
                if param.param_type == "categorical":
                    value_counts = df[param_name].value_counts()
                    summary += "Value counts:\n"
                    for val, count in value_counts.items():
                        summary += f"- {val}: {count}\n"
                else:
                    summary += f"- Count: {df[param_name].count()}\n"
                    summary += f"- Mean: {df[param_name].mean():.4f}\n"
                    summary += f"- Std Dev: {df[param_name].std():.4f}\n"
                    summary += f"- Min: {df[param_name].min():.4f}\n"
                    summary += f"- Max: {df[param_name].max():.4f}\n"
                    
                summary += "\n"
        
        summary_text.setText(summary)
    
    def run_regression():
        target = target_combo.currentText()
        regression_result.clear()
        
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            
            data = []
            for exp in self.model.experiments:
                row = {}
                for param_name in self.model.parameters:
                    if param_name in exp['params']:
                        param_val = exp['params'][param_name]
                        if self.model.parameters[param_name].param_type == "categorical":
                            row[param_name] = str(param_val)
                        else:
                            row[param_name] = float(param_val)
                
                if target in exp['results']:
                    row[target] = exp['results'][target] * 100.0
                    data.append(row)
            
            if not data:
                regression_result.setText(f"No data available with {target} results.")
                return
            
            df = pd.DataFrame(data)
            
            cat_columns = []
            for param_name, param in self.model.parameters.items():
                if param.param_type == "categorical" and param_name in df.columns:
                    cat_columns.append(param_name)
            
            if cat_columns:
                df = pd.get_dummies(df, columns=cat_columns, drop_first=True)
            
            formula_parts = []
            for col in df.columns:
                if col != target:
                    formula_parts.append(col)
            
            if not formula_parts:
                regression_result.setText("No parameters available for regression.")
                return
            
            formula = f"{target} ~ " + " + ".join(formula_parts)
            
            model = ols(formula, data=df).fit()
            
            result_text = "## Regression Analysis\n\n"
            result_text += f"Target: {target}\n\n"
            result_text += f"R-squared: {model.rsquared:.4f}\n"
            result_text += f"Adjusted R-squared: {model.rsquared_adj:.4f}\n\n"
            
            result_text += "### Coefficient Summary\n\n"
            coef_df = pd.DataFrame({
                'Coef': model.params,
                'Std Err': model.bse,
                'P-value': model.pvalues
            })
            
            result_text += coef_df.to_string() + "\n\n"
            
            result_text += "### Significant Factors (p < 0.05):\n"
            sig_factors = coef_df[coef_df['P-value'] < 0.05].index.tolist()
            
            if sig_factors:
                for factor in sig_factors:
                    if factor != "Intercept":
                        result_text += f"- {factor}: {coef_df.loc[factor, 'Coef']:.4f} (p={coef_df.loc[factor, 'P-value']:.4f})\n"
            else:
                result_text += "- No significant factors found\n"
            
            regression_result.setText(result_text)
            
        except Exception as e:
            regression_result.setText(f"Error during regression analysis: {str(e)}")
    
    def run_anova():
        response = response_combo.currentText()
        factor = factor_combo.currentText()
        anova_result.clear()
        
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            
            data = []
            for exp in self.model.experiments:
                row = {}
                if factor in exp['params'] and response in exp['results']:
                    row[factor] = str(exp['params'][factor])
                    row[response] = exp['results'][response] * 100.0
                    data.append(row)
            
            if not data:
                anova_result.setText(f"No data available with both {factor} and {response}.")
                return
            
            df = pd.DataFrame(data)
            
            formula = f"{response} ~ C({factor})"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            result_text = "## ANOVA Results\n\n"
            result_text += f"Response Variable: {response}\n"
            result_text += f"Factor: {factor}\n\n"
            
            result_text += anova_table.to_string() + "\n\n"
            
            f_value = anova_table.iloc[0, 2]
            p_value = anova_table.iloc[0, 3]
            
            result_text += "### Interpretation\n\n"
            
            if p_value < 0.05:
                result_text += f"The factor '{factor}' has a statistically significant effect on {response} (F={f_value:.4f}, p={p_value:.4f}).\n\n"
            else:
                result_text += f"The factor '{factor}' does not have a statistically significant effect on {response} (F={f_value:.4f}, p={p_value:.4f}).\n\n"
            
            result_text += "### Group Means\n\n"
            group_means = df.groupby(factor)[response].mean().reset_index()
            group_means.columns = [factor, f"Mean {response} (%)"]
            
            result_text += group_means.to_string(index=False) + "\n"
            
            anova_result.setText(result_text)
            
        except Exception as e:
            anova_result.setText(f"Error during ANOVA: {str(e)}")
    
    analysis_dialog.exec()

def plan_parallel_experiments(self):
    if not self.model.parameters:
        QMessageBox.information(self, "Information", "Define parameters first.")
        return
        
    parallel_dialog = QDialog(self)
    parallel_dialog.setWindowTitle("Parallel Experiment Planning")
    parallel_dialog.resize(800, 600)
    
    layout = QVBoxLayout(parallel_dialog)
    
    control_panel = QHBoxLayout()
    
    control_panel.addWidget(QLabel("Number of experiments:"))
    n_experiments = QSpinBox()
    n_experiments.setRange(1, 50)
    n_experiments.setValue(10)
    control_panel.addWidget(n_experiments)
    
    control_panel.addWidget(QLabel("Diversity weight:"))
    diversity_slider = QSlider(Qt.Horizontal)
    diversity_slider.setRange(0, 100)
    diversity_slider.setValue(70)
    control_panel.addWidget(diversity_slider)
    
    generate_btn = QPushButton("Generate Plan")
    generate_btn.clicked.connect(lambda: generate_plan())
    control_panel.addWidget(generate_btn)
    
    export_btn = QPushButton("Export Plan")
    export_btn.clicked.connect(lambda: export_plan(exp_table))
    control_panel.addWidget(export_btn)
    
    layout.addLayout(control_panel)
    
    exp_table = QTableWidget()
    exp_table.setSelectionBehavior(QTableWidget.SelectRows)
    
    layout.addWidget(exp_table)
    
    viz_panel = QHBoxLayout()
    
    viz_canvas = MplCanvas(width=6, height=4)
    viz_panel.addWidget(viz_canvas, 1)
    
    info_panel = QVBoxLayout()
    
    diversity_info = QTextEdit()
    diversity_info.setReadOnly(True)
    diversity_info.setMaximumWidth(300)
    info_panel.addWidget(diversity_info)
    
    viz_panel.addLayout(info_panel)
    
    layout.addLayout(viz_panel)
    
    def generate_plan():
        exp_table.clear()
        
        n = n_experiments.value()
        diversity_weight = diversity_slider.value() / 100.0
        
        pool_size = n * 10
        candidate_pool = []
        
        try:
            for _ in range(pool_size):
                params = {}
                for name, param in self.model.parameters.items():
                    params[name] = param.suggest_value()
                
                if self.model.experiments:
                    distances = []
                    for exp in self.model.experiments:
                        dist = self.model.calculate_experiment_distance(params, exp['params'])
                        if 'score' in exp:
                            distances.append((dist, exp['score']))
                    
                    if distances:
                        weights = [1 / (d + 0.01) for d, _ in distances]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            predicted_score = sum(w * s for (_, s), w in zip(distances, weights)) / total_weight
                        else:
                            predicted_score = 0.5
                    else:
                        predicted_score = 0.5
                else:
                    predicted_score = 0.5
                
                candidate_pool.append({
                    'params': params,
                    'predicted_score': predicted_score
                })
            
            candidate_pool.sort(key=lambda x: x['predicted_score'], reverse=True)
            
            selected = self.model.select_diverse_subset(candidate_pool, n, diversity_weight)
            
            columns = ["ID"] + list(self.model.parameters.keys()) + ["Predicted"]
            exp_table.setColumnCount(len(columns))
            exp_table.setHorizontalHeaderLabels(columns)
            
            for i, exp in enumerate(selected):
                row = exp_table.rowCount()
                exp_table.insertRow(row)
                
                exp_table.setItem(row, 0, QTableWidgetItem(str(i+1)))
                
                for col, param_name in enumerate(self.model.parameters.keys(), 1):
                    value = exp['params'][param_name]
                    if isinstance(value, float):
                        value_str = f"{value:.4g}"
                    else:
                        value_str = str(value)
                    exp_table.setItem(row, col, QTableWidgetItem(value_str))
                
                pred_value = exp['predicted_score'] * 100.0
                exp_table.setItem(row, len(columns)-1, QTableWidgetItem(f"{pred_value:.2f}%"))
            
            exp_table.resizeColumnsToContents()
            
            viz_canvas.axes.clear()
            
            if n > 1:
                distance_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        dist = self.model.calculate_experiment_distance(
                            selected[i]['params'],
                            selected[j]['params']
                        )
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist
            
                im = viz_canvas.axes.imshow(distance_matrix, cmap='viridis')
                plt.colorbar(im, ax=viz_canvas.axes, label='Distance')
                
                viz_canvas.axes.set_xticks(np.arange(n))
                viz_canvas.axes.set_yticks(np.arange(n))
                
                exp_labels = [f"#{i+1}" for i in range(n)]
                viz_canvas.axes.set_xticklabels(exp_labels, rotation=45, ha="right")
                viz_canvas.axes.set_yticklabels(exp_labels)
                
                viz_canvas.axes.set_title("Experiment Distance Matrix")
                
                avg_dist = np.sum(distance_matrix) / (n * (n-1))
                min_dist = np.min(distance_matrix[distance_matrix > 0])
                max_dist = np.max(distance_matrix)
                
                diversity_text = "## Diversity Statistics\n\n"
                diversity_text += f"- Average distance: {avg_dist:.4f}\n"
                diversity_text += f"- Minimum distance: {min_dist:.4f}\n"
                diversity_text += f"- Maximum distance: {max_dist:.4f}\n\n"
                
                diversity_text += f"Diversity weight: {diversity_weight:.2f}\n"
                diversity_text += f"Number of experiments: {n}\n"
                
                diversity_info.setText(diversity_text)
            
            viz_canvas.draw()
            export_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(parallel_dialog, "Error", f"Error generating plan: {str(e)}")
            export_btn.setEnabled(False)
    
    def export_plan(exp_table):
        if exp_table.rowCount() == 0:
            QMessageBox.information(self, "Information", "No plan to export.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Parallel Experiment Plan",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            data = []
            headers = [exp_table.horizontalHeaderItem(c).text() for c in range(exp_table.columnCount())]
            
            for row in range(exp_table.rowCount()):
                row_data = {}
                for col in range(exp_table.columnCount()):
                    item = exp_table.item(row, col)
                    if item:
                        row_data[headers[col]] = item.text()
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            if filepath.lower().endswith('.xlsx'):
                df.to_excel(filepath, index=False)
                log(self, f"-- Exported plan to {filepath} - Success")
            else:
                if not filepath.lower().endswith('.csv'):
                    filepath += '.csv'
                df.to_csv(filepath, index=False)
                log(self, f"-- Exported plan to {filepath} - Success")
            
            QMessageBox.information(self, "Success", f"Plan exported to {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting plan: {str(e)}")
    
    generate_plan()
    parallel_dialog.exec()

def open_structure_editor(self):
    try:
        import rdkit
        from rdkit.Chem import Draw
        from rdkit import Chem
        
        editor_dialog = QDialog(self)
        editor_dialog.setWindowTitle("Molecular Structure Editor")
        editor_dialog.resize(800, 600)
        
        layout = QVBoxLayout(editor_dialog)
        
        smiles_layout = QHBoxLayout()
        smiles_layout.addWidget(QLabel("SMILES:"))
        
        smiles_input = QLineEdit()
        smiles_layout.addWidget(smiles_input)
        
        render_btn = QPushButton("Render")
        smiles_layout.addWidget(render_btn)
        
        layout.addLayout(smiles_layout)
        
        structure_view = QLabel("Enter SMILES to view structure")
        structure_view.setAlignment(Qt.AlignCenter)
        structure_view.setMinimumHeight(300)
        
        layout.addWidget(structure_view)
        
        info_panel = QTextEdit()
        info_panel.setReadOnly(True)
        layout.addWidget(info_panel)
        
        def render_structure():
            smiles = smiles_input.text().strip()
            if not smiles:
                return
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 300))
                    
                    height, width, channel = img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    structure_view.setPixmap(pixmap)
                    
                    from rdkit.Chem import Descriptors, Lipinski
                    
                    props = {
                        "Molecular Weight": Descriptors.MolWt(mol),
                        "LogP": Descriptors.MolLogP(mol),
                        "H-Bond Donors": Lipinski.NumHDonors(mol),
                        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
                        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
                        "TPSA": Descriptors.TPSA(mol),
                        "Heavy Atoms": mol.GetNumHeavyAtoms(),
                        "Rings": Descriptors.RingCount(mol)
                    }
                    
                    info_text = "## Molecular Properties\n\n"
                    for prop, value in props.items():
                        info_text += f"**{prop}**: {value:.2f}\n"
                        
                    info_panel.setText(info_text)
                else:
                    structure_view.setText("Invalid SMILES")
                    info_panel.setText("Error: Could not parse SMILES string")
                    
            except Exception as e:
                structure_view.setText(f"Error: {str(e)}")
                info_panel.setText(f"Error: {str(e)}")
        
        render_btn.clicked.connect(render_structure)
        smiles_input.returnPressed.connect(render_structure)
        
        editor_dialog.exec()
        
    except ImportError:
        QMessageBox.warning(
            self, 
            "RDKit Required", 
            "The molecular editor requires RDKit to be installed.\n\n"
            "Install it with: pip install rdkit"
        )
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to open structure editor: {str(e)}")

def show_optimization_settings(self):
    dialog = OptimizationSettingsDialog(self, self.model)
    if dialog.exec():
        log(self, "-- Optimization settings updated - Success")
        
        self.exploit_slider.setValue(int(self.model.exploitation_weight * 100))

def show_preferences(self):
    prefs_dialog = QDialog(self)
    prefs_dialog.setWindowTitle("Preferences")
    prefs_dialog.resize(550, 450)
    
    layout = QVBoxLayout(prefs_dialog)
    
    tabs = QTabWidget()
    
    display_tab = QWidget()
    display_layout = QFormLayout(display_tab)
    
    from ..core import settings
    
    auto_round_check = QCheckBox("Automatically round numerical values")
    auto_round_check.setChecked(settings.auto_round)
    display_layout.addRow("Auto Rounding:", auto_round_check)
    
    precision_spin = QSpinBox()
    precision_spin.setRange(0, 8)
    precision_spin.setValue(settings.rounding_precision)
    display_layout.addRow("Decimal Precision:", precision_spin)
    
    smart_round_check = QCheckBox("Use smart rounding based on value magnitude")
    smart_round_check.setChecked(settings.smart_rounding)
    display_layout.addRow("Smart Rounding:", smart_round_check)
    
    theme_combo = QComboBox()
    theme_combo.addItems(["Light", "Dark", "System"])
    theme_combo.setCurrentText("Light")
    display_layout.addRow("UI Theme:", theme_combo)
    
    font_size_spin = QSpinBox()
    font_size_spin.setRange(8, 18)
    font_size_spin.setValue(10)
    display_layout.addRow("UI Font Size:", font_size_spin)
    
    tabs.addTab(display_tab, "Display")
    
    # Add Experiment Design tab with logical unit rounding settings
    design_tab = QWidget()
    design_layout = QVBoxLayout(design_tab)
    
    # Logical unit rounding settings
    logical_units_box = QGroupBox("Logical Unit Rounding")
    logical_units_layout = QFormLayout(logical_units_box)
    
    use_logical_units = QCheckBox("Use logical unit rounding for experiment parameters")
    use_logical_units.setChecked(settings.use_logical_units)
    logical_units_layout.addRow(use_logical_units)
    
    shrink_intervals = QCheckBox("Progressively shrink intervals during optimization")
    shrink_intervals.setChecked(settings.shrink_intervals)
    logical_units_layout.addRow(shrink_intervals)
    
    shrink_factor = QDoubleSpinBox()
    shrink_factor.setRange(0.1, 0.9)
    shrink_factor.setSingleStep(0.05)
    shrink_factor.setDecimals(2)
    shrink_factor.setValue(settings.interval_shrink_factor)
    logical_units_layout.addRow("Interval Shrink Factor:", shrink_factor)
    
    experiments_before_shrinking = QSpinBox()
    experiments_before_shrinking.setRange(5, 50)
    experiments_before_shrinking.setValue(settings.experiments_before_shrinking)
    logical_units_layout.addRow("Experiments Before Shrinking:", experiments_before_shrinking)
    
    # Parameter interval table
    parameter_intervals_label = QLabel("Parameter Type Intervals:")
    parameter_intervals_layout = QGridLayout()
    parameter_intervals_layout.addWidget(QLabel("<b>Parameter Type</b>"), 0, 0)
    parameter_intervals_layout.addWidget(QLabel("<b>Interval</b>"), 0, 1)
    parameter_intervals_layout.addWidget(QLabel("<b>Min</b>"), 0, 2)
    parameter_intervals_layout.addWidget(QLabel("<b>Max</b>"), 0, 3)
    
    # Time settings
    parameter_intervals_layout.addWidget(QLabel("Time (h)"), 1, 0)
    time_interval = QDoubleSpinBox()
    time_interval.setRange(0.05, 1.0)
    time_interval.setSingleStep(0.05)
    time_interval.setDecimals(2)
    time_interval.setValue(settings.param_rounding["time"]["interval"])
    parameter_intervals_layout.addWidget(time_interval, 1, 1)
    
    time_min = QDoubleSpinBox()
    time_min.setRange(0.1, 10.0)
    time_min.setSingleStep(0.5)
    time_min.setDecimals(1)
    time_min.setValue(settings.param_rounding["time"]["min"])
    parameter_intervals_layout.addWidget(time_min, 1, 2)
    
    time_max = QDoubleSpinBox()
    time_max.setRange(5.0, 48.0)
    time_max.setSingleStep(1.0)
    time_max.setDecimals(1)
    time_max.setValue(settings.param_rounding["time"]["max"])
    parameter_intervals_layout.addWidget(time_max, 1, 3)
    
    # Temperature settings
    parameter_intervals_layout.addWidget(QLabel("Temperature (°C)"), 2, 0)
    temp_interval = QDoubleSpinBox()
    temp_interval.setRange(1.0, 25.0)
    temp_interval.setSingleStep(5.0)
    temp_interval.setDecimals(0)
    temp_interval.setValue(settings.param_rounding["temperature"]["interval"])
    parameter_intervals_layout.addWidget(temp_interval, 2, 1)
    
    temp_min = QDoubleSpinBox()
    temp_min.setRange(-20.0, 50.0)
    temp_min.setSingleStep(5.0)
    temp_min.setDecimals(0)
    temp_min.setValue(settings.param_rounding["temperature"]["min"])
    parameter_intervals_layout.addWidget(temp_min, 2, 2)
    
    temp_max = QDoubleSpinBox()
    temp_max.setRange(50.0, 150.0)
    temp_max.setSingleStep(5.0)
    temp_max.setDecimals(0)
    temp_max.setValue(settings.param_rounding["temperature"]["max"])
    parameter_intervals_layout.addWidget(temp_max, 2, 3)
    
    # Equivalents settings
    parameter_intervals_layout.addWidget(QLabel("Equivalents (eq)"), 3, 0)
    equiv_interval = QDoubleSpinBox()
    equiv_interval.setRange(0.1, 1.0)
    equiv_interval.setSingleStep(0.1)
    equiv_interval.setDecimals(1)
    equiv_interval.setValue(settings.param_rounding["eq"]["interval"])
    parameter_intervals_layout.addWidget(equiv_interval, 3, 1)
    
    equiv_min = QDoubleSpinBox()
    equiv_min.setRange(0.1, 2.0)
    equiv_min.setSingleStep(0.5)
    equiv_min.setDecimals(1)
    equiv_min.setValue(settings.param_rounding["eq"]["min"])
    parameter_intervals_layout.addWidget(equiv_min, 3, 2)
    
    equiv_max = QDoubleSpinBox()
    equiv_max.setRange(2.0, 10.0)
    equiv_max.setSingleStep(0.5)
    equiv_max.setDecimals(1)
    equiv_max.setValue(settings.param_rounding["eq"]["max"])
    parameter_intervals_layout.addWidget(equiv_max, 3, 3)
    
    # Catalyst loading settings
    parameter_intervals_layout.addWidget(QLabel("Catalyst Loading (mol%)"), 4, 0)
    cat_interval = QDoubleSpinBox()
    cat_interval.setRange(0.01, 0.25)
    cat_interval.setSingleStep(0.01)
    cat_interval.setDecimals(2)
    cat_interval.setValue(settings.param_rounding["catalyst"]["interval"])
    parameter_intervals_layout.addWidget(cat_interval, 4, 1)
    
    cat_min = QDoubleSpinBox()
    cat_min.setRange(0.01, 0.25)
    cat_min.setSingleStep(0.05)
    cat_min.setDecimals(2)
    cat_min.setValue(settings.param_rounding["catalyst"]["min"])
    parameter_intervals_layout.addWidget(cat_min, 4, 2)
    
    cat_max = QDoubleSpinBox()
    cat_max.setRange(0.1, 5.0)
    cat_max.setSingleStep(0.1)
    cat_max.setDecimals(2)
    cat_max.setValue(settings.param_rounding["catalyst"]["max"])
    parameter_intervals_layout.addWidget(cat_max, 4, 3)
    
    # Concentration settings
    parameter_intervals_layout.addWidget(QLabel("Concentration (M)"), 5, 0)
    conc_interval = QDoubleSpinBox()
    conc_interval.setRange(0.05, 0.5)
    conc_interval.setSingleStep(0.05)
    conc_interval.setDecimals(2)
    conc_interval.setValue(settings.param_rounding["concentration"]["interval"])
    parameter_intervals_layout.addWidget(conc_interval, 5, 1)
    
    conc_min = QDoubleSpinBox()
    conc_min.setRange(0.05, 0.5)
    conc_min.setSingleStep(0.05)
    conc_min.setDecimals(2)
    conc_min.setValue(settings.param_rounding["concentration"]["min"])
    parameter_intervals_layout.addWidget(conc_min, 5, 2)
    
    conc_max = QDoubleSpinBox()
    conc_max.setRange(0.5, 3.0)
    conc_max.setSingleStep(0.1)
    conc_max.setDecimals(1)
    conc_max.setValue(settings.param_rounding["concentration"]["max"])
    parameter_intervals_layout.addWidget(conc_max, 5, 3)
    
    # Add parameter rounding settings to layout
    logical_units_layout.addRow(parameter_intervals_label)
    logical_units_layout.addRow(parameter_intervals_layout)
    
    design_layout.addWidget(logical_units_box)
    design_layout.addStretch()
    
    tabs.addTab(design_tab, "Experiment Design")
    
    model_tab = QWidget()
    model_layout = QFormLayout(model_tab)
    
    acq_function_combo = QComboBox()
    acq_function_combo.addItems([
        "Expected Improvement (EI)", 
        "Probability of Improvement (PI)", 
        "Upper Confidence Bound (UCB)"
    ])
    
    if self.model.acquisition_function == "ei":
        acq_function_combo.setCurrentIndex(0)
    elif self.model.acquisition_function == "pi":
        acq_function_combo.setCurrentIndex(1)
    elif self.model.acquisition_function == "ucb":
        acq_function_combo.setCurrentIndex(2)
        
    model_layout.addRow("Default Acquisition Function:", acq_function_combo)
    
    exploit_spin = QDoubleSpinBox()
    exploit_spin.setRange(0.0, 1.0)
    exploit_spin.setSingleStep(0.05)
    exploit_spin.setDecimals(2)
    exploit_spin.setValue(self.model.exploitation_weight)
    model_layout.addRow("Default Exploitation Weight:", exploit_spin)
    
    tabs.addTab(model_tab, "Modeling")
    
    layout.addWidget(tabs)
    
    button_box = QHBoxLayout()
    save_btn = QPushButton("Save")
    cancel_btn = QPushButton("Cancel")
    
    save_btn.clicked.connect(prefs_dialog.accept)
    cancel_btn.clicked.connect(prefs_dialog.reject)
    
    button_box.addWidget(save_btn)
    button_box.addWidget(cancel_btn)
    
    layout.addLayout(button_box)
    
    if prefs_dialog.exec():
        # Display settings
        settings.auto_round = auto_round_check.isChecked()
        settings.rounding_precision = precision_spin.value()
        settings.smart_rounding = smart_round_check.isChecked()
        
        # Logical unit settings
        settings.use_logical_units = use_logical_units.isChecked()
        settings.shrink_intervals = shrink_intervals.isChecked()
        settings.interval_shrink_factor = shrink_factor.value()
        settings.experiments_before_shrinking = experiments_before_shrinking.value()
        
        # Update parameter rounding settings
        settings.param_rounding["time"]["interval"] = time_interval.value()
        settings.param_rounding["time"]["min"] = time_min.value()
        settings.param_rounding["time"]["max"] = time_max.value()
        
        settings.param_rounding["temperature"]["interval"] = temp_interval.value()
        settings.param_rounding["temperature"]["min"] = temp_min.value()
        settings.param_rounding["temperature"]["max"] = temp_max.value()
        
        settings.param_rounding["eq"]["interval"] = equiv_interval.value()
        settings.param_rounding["eq"]["min"] = equiv_min.value()
        settings.param_rounding["eq"]["max"] = equiv_max.value()
        settings.param_rounding["equiv"]["interval"] = equiv_interval.value()
        settings.param_rounding["equiv"]["min"] = equiv_min.value()
        settings.param_rounding["equiv"]["max"] = equiv_max.value()
        
        settings.param_rounding["catalyst"]["interval"] = cat_interval.value()
        settings.param_rounding["catalyst"]["min"] = cat_min.value()
        settings.param_rounding["catalyst"]["max"] = cat_max.value()
        settings.param_rounding["load"]["interval"] = cat_interval.value()
        settings.param_rounding["load"]["min"] = cat_min.value()
        settings.param_rounding["load"]["max"] = cat_max.value()
        
        settings.param_rounding["concentration"]["interval"] = conc_interval.value()
        settings.param_rounding["concentration"]["min"] = conc_min.value()
        settings.param_rounding["concentration"]["max"] = conc_max.value()
        settings.param_rounding["conc"]["interval"] = conc_interval.value()
        settings.param_rounding["conc"]["min"] = conc_min.value()
        settings.param_rounding["conc"]["max"] = conc_max.value()
        
        # Update model settings
        self.auto_round_check.setChecked(settings.auto_round)
        self.precision_spin.setValue(settings.rounding_precision)
        self.smart_round_check.setChecked(settings.smart_rounding)
        
        acq_index = acq_function_combo.currentIndex()
        if acq_index == 0:
            self.model.acquisition_function = "ei"
        elif acq_index == 1:
            self.model.acquisition_function = "pi"
        elif acq_index == 2:
            self.model.acquisition_function = "ucb"
            
        self.model.exploitation_weight = exploit_spin.value()
        
        log(self, "-- Preferences updated - Success")

def show_documentation(self):
    doc_dialog = QDialog(self)
    doc_dialog.setWindowTitle("Documentation")
    doc_dialog.resize(700, 500)
    
    layout = QVBoxLayout(doc_dialog)
    
    tabs = QTabWidget()
    
    overview_tab = QWidget()
    overview_layout = QVBoxLayout(overview_tab)
    
    overview_text = QTextEdit()
    overview_text.setReadOnly(True)
    overview_text.setHtml("""
    <h2>Bayesian DOE - Chemical Reaction Optimizer</h2>
    
    <p>This application uses Bayesian optimization to efficiently optimize chemical reactions with minimal experiments.</p>
    
    <h3>Key Features:</h3>
    <ul>
        <li>Bayesian optimization using acquisition functions like Expected Improvement</li>
        <li>Support for continuous, discrete, and categorical parameters</li>
        <li>Prior knowledge integration for expert-guided optimization</li>
        <li>Analysis tools to understand parameter importance</li>
        <li>Chemical registry with property information</li>
        <li>Result visualization and analysis</li>
    </ul>
    
    <h3>Workflow:</h3>
    <ol>
        <li>Define reaction parameters</li>
        <li>Set prior knowledge (optional)</li>
        <li>Generate initial experiments</li>
        <li>Run experiments and input results</li>
        <li>Generate next round of experiments guided by the model</li>
        <li>Repeat until optimal conditions are found</li>
    </ol>
    """)
    
    overview_layout.addWidget(overview_text)
    tabs.addTab(overview_tab, "Overview")
    
    params_tab = QWidget()
    params_layout = QVBoxLayout(params_tab)
    
    params_text = QTextEdit()
    params_text.setReadOnly(True)
    params_text.setHtml("""
    <h2>Parameter Configuration</h2>
    
    <h3>Parameter Types:</h3>
    <ul>
        <li><b>Continuous</b>: Floating-point values within a range (e.g., temperature: 25.0-100.0°C)</li>
        <li><b>Discrete</b>: Integer values within a range (e.g., reaction time: 1-24 hours)</li>
        <li><b>Categorical</b>: Selection from a list of options (e.g., solvent: THF, toluene, etc.)</li>
    </ul>
    
    <h3>Setting Prior Knowledge:</h3>
    <p>For continuous and discrete parameters:</p>
    <ul>
        <li>Expected optimal value: Your best guess of the optimal value</li>
        <li>Confidence level: How certain you are about this guess</li>
    </ul>
    
    <p>For categorical parameters:</p>
    <ul>
        <li>Set preference weights for each option (higher = more likely to be selected early)</li>
    </ul>
    
    <h3>Templates:</h3>
    <p>Use templates to quickly set up parameters for common reaction types:</p>
    <ul>
        <li>Reaction Conditions (temperature, time, solvent)</li>
        <li>Catalyst Screening</li>
        <li>Solvent Screening</li>
        <li>Cross-Coupling</li>
    </ul>
    """)
    
    params_layout.addWidget(params_text)
    tabs.addTab(params_tab, "Parameters")
    
    layout.addWidget(tabs)
    
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(doc_dialog.accept)
    layout.addWidget(close_btn)
    
    doc_dialog.exec()

def show_about(self):
    about_dialog = QDialog(self)
    about_dialog.setWindowTitle("About")
    about_dialog.resize(500, 400)
    
    layout = QVBoxLayout(about_dialog)
    
    about_text = QTextEdit()
    about_text.setReadOnly(True)
    about_text.setHtml("""
    <div align="center">
    <h1>Bayesian DOE</h1>
    <h3>Chemical Reaction Optimizer</h3>
    <p>Version 1.0.5</p>
    
    <p>A powerful tool for optimizing chemical reactions with minimal experiments<br>
    using Bayesian optimization techniques.</p>
    
    <p>&copy; 2025 Johan H.G. Natter</p>
    
    <h3>Technologies</h3>
    <p>Python, PySide6, Optuna, scikit-learn, numpy, pandas, matplotlib</p>
    
    <h3>Open Source Licenses</h3>
    <p>This software is released under the MIT License</p>
    
    <p>Powered by:</p>
    <ul style="list-style-type: none; padding: 0;">
        <li>Optuna (MIT License)</li>
        <li>scikit-learn (BSD 3-Clause)</li>
        <li>PySide6 (LGPL)</li>
        <li>matplotlib (PSF-based)</li>
    </ul>
    </div>
    """)
    
    layout.addWidget(about_text)
    
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(about_dialog.accept)
    layout.addWidget(close_btn)
    
    about_dialog.exec()