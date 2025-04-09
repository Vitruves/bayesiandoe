import sys
import os
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
import datetime
import random
from scipy import stats
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap, QImage, QAction, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QGroupBox, QHeaderView, QTableWidgetItem,
    QScrollArea, QFrame, QSizePolicy, QTextEdit, QMessageBox, QFileDialog,
    QSlider, QFormLayout, QGridLayout, QListWidget, QInputDialog,
    QStatusBar, QProgressBar, QStackedWidget, QTableWidget, QDialog, QListWidgetItem
)

from ..core import _calculate_parameter_distance

from ..core import OptunaBayesianExperiment
from ..parameters import ChemicalParameter
from ..visualizations import (
    plot_optimization_history, plot_parameter_importance, 
    plot_parameter_contour, plot_objective_correlation, 
    plot_response_surface, plot_convergence
)
from .widgets import (
    LogDisplay, SplashScreen, ParameterTable, ExperimentTable,
    BestResultsTable, AllResultsTable
)
from .canvas import MplCanvas, Mpl3DCanvas
from .dialogs import (
    ParameterDialog, ResultDialog, TemplateSelector, PriorDialog,
    OptimizationSettingsDialog
)
from ..registry import ChemicalRegistry

class BayesianDOEApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = OptunaBayesianExperiment()
        self.current_round = 0
        self.round_start_indices = []
        self.working_directory = os.getcwd()
        
        # Initialize the ChemicalRegistry with proper normalization
        self.registry_manager = ChemicalRegistry()
        self.registry_manager.initialize_registry()  # Ensure properties are normalized
        self.registry = self.registry_manager.get_full_registry()
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Bayesian DOE - Chemical Reaction Optimizer")
        self.setMinimumSize(1200, 800)
        
        self.log_display = LogDisplay()
        
        self.create_menu_bar()
        self.create_status_bar()
        self.create_central_widget()
        
        self.log("-- Application started successfully")
        
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(QAction("New Project", self, triggered=self.new_project))
        file_menu.addAction(QAction("Open Project", self, triggered=self.open_project))
        file_menu.addAction(QAction("Save Project", self, triggered=self.save_project))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Import Data", self, triggered=self.import_data))
        file_menu.addAction(QAction("Export Results", self, triggered=self.export_results))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))
        
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(QAction("Optimization Settings", self, triggered=self.show_optimization_settings))
        edit_menu.addAction(QAction("Preferences", self, triggered=self.show_preferences))
        
        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(QAction("Structure Editor", self, triggered=self.open_structure_editor))
        tools_menu.addAction(QAction("Parallel Experiments", self, triggered=self.plan_parallel_experiments))
        tools_menu.addAction(QAction("Statistical Analysis", self, triggered=self.statistical_analysis))
        
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("Documentation", self, triggered=self.show_documentation))
        help_menu.addAction(QAction("About", self, triggered=self.show_about))
        
    def create_status_bar(self):
        from ..core import settings
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Left side - status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)  # stretch=1 makes it take available space
        
        # Add rounding settings section
        rounding_section = QWidget()
        rounding_layout = QHBoxLayout(rounding_section)
        rounding_layout.setContentsMargins(0, 0, 0, 0)
        
        # Auto rounding checkbox
        self.auto_round_check = QCheckBox("Auto Round")
        self.auto_round_check.setChecked(settings.auto_round)
        self.auto_round_check.setToolTip("Automatically round numerical values")
        self.auto_round_check.stateChanged.connect(self.update_rounding_settings)
        rounding_layout.addWidget(self.auto_round_check)
        
        # Rounding precision label and spinner
        rounding_layout.addWidget(QLabel("Precision:"))
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(0, 8)
        self.precision_spin.setValue(settings.rounding_precision)
        self.precision_spin.setToolTip("Number of decimal places for rounding")
        self.precision_spin.valueChanged.connect(self.update_rounding_settings)
        rounding_layout.addWidget(self.precision_spin)
        
        # Smart rounding checkbox
        self.smart_round_check = QCheckBox("Smart")
        self.smart_round_check.setChecked(settings.smart_rounding)
        self.smart_round_check.setToolTip("Use smart rounding based on value magnitude")
        self.smart_round_check.stateChanged.connect(self.update_rounding_settings)
        rounding_layout.addWidget(self.smart_round_check)
        
        self.status_bar.addPermanentWidget(rounding_section)
        
        # Add progress bar at the end
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setValue(0)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def update_rounding_settings(self):
        """Update the global rounding settings based on UI controls"""
        from ..core import settings
        
        settings.auto_round = self.auto_round_check.isChecked()
        settings.rounding_precision = self.precision_spin.value()
        settings.smart_rounding = self.smart_round_check.isChecked()
        
        # Update UI elements that display values
        self.update_ui_from_model()
        
        # Log the change
        auto_status = "enabled" if settings.auto_round else "disabled"
        smart_status = "enabled" if settings.smart_rounding else "disabled"
        self.log(f"-- Rounding settings updated: Auto-round {auto_status}, Precision {settings.rounding_precision}, Smart rounding {smart_status}")
        
    def create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        self.tab_widget = QTabWidget()
        self.setup_setup_tab()
        self.setup_prior_tab()
        self.setup_experiment_tab()
        self.setup_results_tab()
        self.setup_analysis_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        self.setCentralWidget(central_widget)
        
    def setup_setup_tab(self):
        setup_tab = QWidget()
        layout = QHBoxLayout(setup_tab)
        
        left_panel = QVBoxLayout()
        
        param_group = QGroupBox("Reaction Parameters")
        param_layout = QVBoxLayout(param_group)
        
        self.param_table = ParameterTable()
        param_layout.addWidget(self.param_table)
        
        button_layout = QHBoxLayout()
        self.add_param_button = QPushButton("Add Parameter")
        self.edit_param_button = QPushButton("Edit Parameter")
        self.remove_param_button = QPushButton("Remove Parameter")
        
        self.add_param_button.clicked.connect(self.add_parameter)
        self.edit_param_button.clicked.connect(self.edit_parameter)
        self.remove_param_button.clicked.connect(self.remove_parameter)
        
        button_layout.addWidget(self.add_param_button)
        button_layout.addWidget(self.edit_param_button)
        button_layout.addWidget(self.remove_param_button)
        
        param_layout.addLayout(button_layout)
        
        left_panel.addWidget(param_group)
        
        templates_group = QGroupBox("Parameter Templates")
        templates_layout = QGridLayout(templates_group)
        
        templates = [
            ("Reaction Conditions", lambda: self.load_template("reaction_conditions")),
            ("Catalyst Screening", lambda: self.load_template("catalyst")),
            ("Solvent Screening", lambda: self.load_template("solvent")),
            ("Cross-Coupling", lambda: self.load_template("cross_coupling")),
            ("Oxidation", lambda: self.load_template("oxidation")),
            ("Reduction", lambda: self.load_template("reduction")),
            ("Amide Coupling", lambda: self.load_template("amide_coupling")),
            ("Organocatalysis", lambda: self.load_template("organocatalysis"))
        ]
        
        row, col = 0, 0
        max_cols = 3
        
        for text, func in templates:
            button = QPushButton(text)
            button.clicked.connect(func)
            templates_layout.addWidget(button, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        left_panel.addWidget(templates_group)
        
        right_panel = QVBoxLayout()
        
        registry_group = QGroupBox("Parameter Registry")
        registry_layout = QVBoxLayout(registry_group)
        
        self.registry_tabs = QTabWidget()
        self.registry_lists = {}
        self.registry_widgets = {}
        
        # Build UI from registry manager
        for reg_type in self.registry_manager.get_all_types():
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            category_tabs = QTabWidget()
            category_widgets = {}
            
            categories = self.registry_manager.get_categories(reg_type)
            for category in categories:
                cat_tab = QWidget()
                cat_layout = QVBoxLayout(cat_tab)
                
                list_widget = QListWidget()
                list_widget.setSelectionMode(QListWidget.ExtendedSelection)
                list_widget.setMouseTracking(True)
                list_widget.itemEntered.connect(lambda item, rt=reg_type, cat=category: self.show_registry_item_tooltip(item, rt, cat))
                
                cat_layout.addWidget(list_widget)
                
                category_tabs.addTab(cat_tab, category)
                category_widgets[category] = list_widget
            
            tab_layout.addWidget(category_tabs)
            
            btn_layout = QHBoxLayout()
            add_to_param_btn = QPushButton("Add to Parameters")
            add_to_reg_btn = QPushButton("Add New")
            remove_btn = QPushButton("Remove")
            
            rt = reg_type  # Create a local variable to capture the current value
            add_to_param_btn.clicked.connect(lambda checked, rt=rt: self.add_from_registry(rt))
            add_to_reg_btn.clicked.connect(lambda checked, rt=rt: self.add_to_registry(rt))
            remove_btn.clicked.connect(lambda checked, rt=rt: self.remove_from_registry(rt))
            
            btn_layout.addWidget(add_to_param_btn)
            btn_layout.addWidget(add_to_reg_btn)
            btn_layout.addWidget(remove_btn)
            
            tab_layout.addLayout(btn_layout)
            
            self.registry_tabs.addTab(tab, reg_type.capitalize())
            
            self.registry_lists[reg_type] = category_widgets
            self.registry_widgets[reg_type] = {
                "add_to_param": add_to_param_btn,
                "add_to_reg": add_to_reg_btn,
                "remove": remove_btn
            }
        
        registry_layout.addWidget(self.registry_tabs)
        right_panel.addWidget(registry_group)
        
        # Populate registry lists with initial data
        self.refresh_registry()
        
        obj_group = QGroupBox("Optimization Objectives")
        obj_layout = QGridLayout(obj_group)
        
        self.obj_yield_check = QCheckBox("Yield")
        self.obj_purity_check = QCheckBox("Purity")
        self.obj_selectivity_check = QCheckBox("Selectivity")
        
        self.weight_yield_spin = QDoubleSpinBox()
        self.weight_purity_spin = QDoubleSpinBox()
        self.weight_selectivity_spin = QDoubleSpinBox()
        
        self.weight_yield_spin.setRange(0.1, 10)
        self.weight_purity_spin.setRange(0.1, 10)
        self.weight_selectivity_spin.setRange(0.1, 10)
        
        self.weight_yield_spin.setValue(1.0)
        self.weight_purity_spin.setValue(1.0)
        self.weight_selectivity_spin.setValue(1.0)
        
        self.obj_yield_check.setChecked(True)
        self.obj_purity_check.setChecked(True)
        self.obj_selectivity_check.setChecked(True)
        
        obj_layout.addWidget(self.obj_yield_check, 0, 0)
        obj_layout.addWidget(QLabel("Weight:"), 0, 1)
        obj_layout.addWidget(self.weight_yield_spin, 0, 2)
        
        obj_layout.addWidget(self.obj_purity_check, 1, 0)
        obj_layout.addWidget(QLabel("Weight:"), 1, 1)
        obj_layout.addWidget(self.weight_purity_spin, 1, 2)
        
        obj_layout.addWidget(self.obj_selectivity_check, 2, 0)
        obj_layout.addWidget(QLabel("Weight:"), 2, 1)
        obj_layout.addWidget(self.weight_selectivity_spin, 2, 2)
        
        apply_obj_btn = QPushButton("Apply Objectives")
        apply_obj_btn.clicked.connect(self.update_objectives)
        obj_layout.addWidget(apply_obj_btn, 3, 0, 1, 3)
        
        right_panel.addWidget(obj_group)
        
        layout.addLayout(left_panel, 65)
        layout.addLayout(right_panel, 35)
        
        self.tab_widget.addTab(setup_tab, "Experiment Setup")
        
    def refresh_registry(self):
        """Refresh the registry lists with current data."""
        # Clear all lists first
        for reg_type, categories in self.registry_lists.items():
            for category, list_widget in categories.items():
                list_widget.clear()
                
                # Get items for this category
                items = self.registry_manager.get_item_names(reg_type, category)
                
                # Add items to list widget
                for item_name in items:
                    list_item = QListWidgetItem(item_name)
                    list_widget.addItem(list_item)
                    
                    # Get properties to use for tooltip
                    props = self.registry_manager.get_item_properties(reg_type, category, item_name)
                    if props:
                        tooltip = "\n".join([f"{k}: {v}" for k, v in props.items() if k != "color"])
                        list_item.setToolTip(tooltip)
                        
                        # Set color if available
                        if "color" in props:
                            color_name = props["color"]
                            list_item.setForeground(QColor(color_name))
        
    def update_objectives(self):
        """Update objectives based on user selections."""
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
        
        # Update model with new objectives and weights
        self.model.objectives = objectives
        self.model.objective_weights = weights
        
        # Update UI elements that depend on objectives
        self.update_ui_from_model()
        
        self.log(f"-- Objectives updated: {', '.join(objectives)} - Success")
    
    def show_registry_item_tooltip(self, item, reg_type, category):
        """Show tooltip for registry item when hovered."""
        if not item:
            return
            
        item_name = item.text()
        properties = self.registry_manager.get_item_properties(reg_type, category, item_name)
        
        if properties:
            tooltip = "\n".join([f"{k}: {v}" for k, v in properties.items() if k != "color"])
            item.setToolTip(tooltip)
    
    def add_from_registry(self, reg_type):
        """Add selected items from registry to parameters."""
        # Get current active tab for this registry type
        current_tab_index = self.registry_tabs.currentIndex()
        if current_tab_index < 0 or reg_type != self.registry_manager.get_all_types()[current_tab_index]:
            QMessageBox.warning(self, "Warning", "Please select a registry tab first.")
            return
        
        # Get category tabs for this type
        categories = self.registry_lists[reg_type]
        
        for category, list_widget in categories.items():
            selected_items = list_widget.selectedItems()
            
            for item in selected_items:
                item_name = item.text()
                param_name = f"{category.lower()}_{item_name}"
                
                # Create parameter based on registry type
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
                            continue  # Skip if already in choices
                
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
                            continue  # Skip if already in choices
                
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
                            continue  # Skip if already in choices
                
                else:
                    # For other types, add as a separate parameter
                    if param_name in self.model.parameters:
                        QMessageBox.warning(self, "Warning", f"Parameter '{param_name}' already exists.")
                        continue
                    
                    param = ChemicalParameter(
                        name=param_name,
                        param_type="categorical",
                        choices=[item_name]
                    )
                
                # Add the parameter to the model
                self.model.add_parameter(param)
        
        # Update UI
        self.param_table.update_from_model(self.model)
        self.update_parameter_combos()
        self.experiment_table.update_columns(self.model)
        self.best_table.update_columns(self.model)
        
        self.log(f"-- Added parameters from {reg_type} registry - Success")
    
    def add_to_registry(self, reg_type):
        """Add a new item to the registry."""
        # Get current category from the active tab
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
        
        # Ask for item name
        item_name, ok = QInputDialog.getText(self, "Add Item", f"Enter new {reg_type} name:")
        if not ok or not item_name.strip():
            return
            
        # Ask for properties
        properties_dialog = QDialog(self)
        properties_dialog.setWindowTitle(f"Properties for {item_name}")
        properties_dialog.resize(400, 300)
        
        layout = QVBoxLayout(properties_dialog)
        form_layout = QFormLayout()
        
        # Add default properties based on registry type
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
        
        # OK/Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(properties_dialog.accept)
        cancel_button.clicked.connect(properties_dialog.reject)
        
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        
        layout.addLayout(button_box)
        
        if properties_dialog.exec() == QDialog.Accepted:
            # Collect properties
            properties = {}
            for prop_name, input_widget in property_inputs.items():
                value = input_widget.text().strip()
                if value:
                    properties[prop_name] = value
            
            # Add to registry
            success = self.registry_manager.add_item(reg_type, current_category, item_name, properties)
            
            if success:
                self.refresh_registry()
                self.log(f"-- Added {item_name} to {reg_type}/{current_category} registry - Success")
            else:
                QMessageBox.warning(self, "Warning", f"Could not add {item_name}. It may already exist.")
    
    def remove_from_registry(self, reg_type):
        """Remove selected items from registry."""
        # Get current category from the active tab
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
                    self.log(f"-- Removed {item_name} from {reg_type}/{current_category} registry - Success")
                else:
                    self.log(f"-- Failed to remove {item_name} from registry - Failed")
            
            self.refresh_registry()
    
    def load_template(self, template_name):
        """Load parameter template."""
        if template_name == "reaction_conditions":
            params = [
                ChemicalParameter(name="Temperature", param_type="continuous", low=0, high=150, units="°C"),
                ChemicalParameter(name="Time", param_type="continuous", low=0.5, high=24, units="h"),
                ChemicalParameter(name="Concentration", param_type="continuous", low=0.05, high=2.0, units="M"),
                ChemicalParameter(name="Solvent", param_type="categorical", choices=["Toluene", "THF", "DCM", "MeOH", "DMSO", "DMF"]),
                ChemicalParameter(name="Atmosphere", param_type="categorical", choices=["Air", "N₂", "Ar"])
            ]
            
        elif template_name == "catalyst":
            params = [
                ChemicalParameter(name="Catalyst", param_type="categorical", choices=["Pd(OAc)₂", "Pd(PPh₃)₄", "Pd/C", "Pd₂(dba)₃", "PdCl₂"]),
                ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.1, high=10, units="mol%"),
                ChemicalParameter(name="Ligand", param_type="categorical", choices=["PPh₃", "BINAP", "Xantphos", "dppf", "None"]),
                ChemicalParameter(name="Ligand:Metal Ratio", param_type="continuous", low=0.5, high=5, units="L:M")
            ]
            
        elif template_name == "solvent":
            params = [
                ChemicalParameter(name="Solvent", param_type="categorical", 
                    choices=["Water", "Methanol", "Ethanol", "Acetone", "Acetonitrile", "DMF", 
                            "DMSO", "THF", "Toluene", "DCM", "EtOAc", "Hexane"]),
                ChemicalParameter(name="Co-solvent", param_type="categorical", 
                    choices=["None", "Water", "Methanol", "DMSO", "THF"]),
                ChemicalParameter(name="Co-solvent Ratio", param_type="continuous", low=0, high=50, units="%vol")
            ]
            
        elif template_name == "cross_coupling":
            params = [
                ChemicalParameter(name="Catalyst", param_type="categorical", 
                    choices=["Pd(OAc)₂", "Pd(PPh₃)₄", "Pd₂(dba)₃", "PdCl₂", "Ni(COD)₂", "NiCl₂·DME"]),
                ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0.1, high=10, units="mol%"),
                ChemicalParameter(name="Ligand", param_type="categorical", 
                    choices=["PPh₃", "SPhos", "XPhos", "RuPhos", "BINAP", "Xantphos", "dppf", "None"]),
                ChemicalParameter(name="Base", param_type="categorical", 
                    choices=["K₂CO₃", "Cs₂CO₃", "K₃PO₄", "KOH", "Et₃N", "KOt-Bu"]),
                ChemicalParameter(name="Temperature", param_type="continuous", low=20, high=150, units="°C"),
                ChemicalParameter(name="Time", param_type="continuous", low=0.5, high=48, units="h"),
                ChemicalParameter(name="Solvent", param_type="categorical", 
                    choices=["Toluene", "THF", "1,4-Dioxane", "DMF", "DMSO", "MeCN", "2-MeTHF"]),
            ]
            
        elif template_name == "oxidation":
            params = [
                ChemicalParameter(name="Oxidant", param_type="categorical", 
                    choices=["H₂O₂", "mCPBA", "TBHP", "NaOCl", "Oxone", "O₂", "TEMPO", "IBX", "DMP"]),
                ChemicalParameter(name="Oxidant Equiv", param_type="continuous", low=1, high=5, units="equiv"),
                ChemicalParameter(name="Catalyst", param_type="categorical", 
                    choices=["None", "Cu(OAc)₂", "FeCl₃", "VO(acac)₂", "Mn(OAc)₃", "RuCl₃"]),
                ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0, high=20, units="mol%"),
                ChemicalParameter(name="Temperature", param_type="continuous", low=0, high=100, units="°C"),
                ChemicalParameter(name="Time", param_type="continuous", low=0.5, high=24, units="h"),
                ChemicalParameter(name="Solvent", param_type="categorical", 
                    choices=["DCM", "MeCN", "Water", "Acetic Acid", "MeOH", "EtOH"])
            ]
            
        elif template_name == "reduction":
            params = [
                ChemicalParameter(name="Reductant", param_type="categorical", 
                    choices=["NaBH₄", "LiAlH₄", "DIBAL-H", "H₂", "Zn/HCl", "Na/NH₃", "SmI₂", "L-Selectride"]),
                ChemicalParameter(name="Reductant Equiv", param_type="continuous", low=0.5, high=5, units="equiv"),
                ChemicalParameter(name="Catalyst", param_type="categorical", 
                    choices=["None", "Pd/C", "Pt/C", "Raney Ni", "Rh/C", "PtO₂"]),
                ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=0, high=20, units="mol%"),
                ChemicalParameter(name="Temperature", param_type="continuous", low=-78, high=80, units="°C"),
                ChemicalParameter(name="Time", param_type="continuous", low=0.5, high=24, units="h"),
                ChemicalParameter(name="Solvent", param_type="categorical", 
                    choices=["THF", "Et₂O", "DCM", "MeOH", "EtOH", "EtOAc"])
            ]
            
        elif template_name == "amide_coupling":
            params = [
                ChemicalParameter(name="Coupling Agent", param_type="categorical", 
                    choices=["EDC·HCl", "DCC", "PyBOP", "HATU", "TBTU", "T3P", "CDI", "COMU"]),
                ChemicalParameter(name="Coupling Agent Equiv", param_type="continuous", low=1, high=2, units="equiv"),
                ChemicalParameter(name="Additive", param_type="categorical", 
                    choices=["None", "HOBt", "HOAt", "DMAP", "6-Cl-HOBt", "Oxyma"]),
                ChemicalParameter(name="Additive Equiv", param_type="continuous", low=0, high=1.5, units="equiv"),
                ChemicalParameter(name="Base", param_type="categorical", 
                    choices=["DIPEA", "Et₃N", "NMM", "DMAP", "Pyridine", "K₂CO₃"]),
                ChemicalParameter(name="Base Equiv", param_type="continuous", low=1, high=5, units="equiv"),
                ChemicalParameter(name="Temperature", param_type="continuous", low=0, high=60, units="°C"),
                ChemicalParameter(name="Time", param_type="continuous", low=1, high=48, units="h"),
                ChemicalParameter(name="Solvent", param_type="categorical", 
                    choices=["DCM", "DMF", "THF", "MeCN", "DMSO"])
            ]
            
        elif template_name == "organocatalysis":
            params = [
                ChemicalParameter(name="Organocatalyst", param_type="categorical", 
                    choices=["L-Proline", "MacMillan", "Cinchonidine", "Thiourea", "Squaramide", "DMAP", "Quinine"]),
                ChemicalParameter(name="Catalyst Loading", param_type="continuous", low=1, high=30, units="mol%"),
                ChemicalParameter(name="Additive", param_type="categorical", 
                    choices=["None", "Benzoic Acid", "TFA", "AcOH", "PhCOOH", "Imidazole"]),
                ChemicalParameter(name="Additive Equiv", param_type="continuous", low=0, high=1, units="equiv"),
                ChemicalParameter(name="Temperature", param_type="continuous", low=-20, high=60, units="°C"),
                ChemicalParameter(name="Time", param_type="continuous", low=4, high=96, units="h"),
                ChemicalParameter(name="Solvent", param_type="categorical", 
                    choices=["DCM", "DMSO", "Toluene", "MeOH", "THF", "DMF", "Water", "CHCl₃"])
            ]
            
        else:
            QMessageBox.warning(self, "Warning", f"Template '{template_name}' not found.")
            return
        
        # Ask for confirmation if there are existing parameters
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
        
        # Add parameters from template
        for param in params:
            self.model.add_parameter(param)
            
        # Update UI
        self.param_table.update_from_model(self.model)
        self.update_parameter_combos()
        self.experiment_table.update_columns(self.model)
        self.best_table.update_columns(self.model)
        
        self.log(f"-- Loaded {template_name} template with {len(params)} parameters - Success")

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
            self.update_parameter_combos()
            self.experiment_table.update_columns(self.model)
            self.best_table.update_columns(self.model)
            
            self.log(f"-- Substrate '{substrate_name.strip()}' added - Success")
        
    def setup_prior_tab(self):
        prior_tab = QWidget()
        layout = QHBoxLayout(prior_tab)
        
        left_panel = QVBoxLayout()
        
        prior_set_group = QGroupBox("Set Prior Knowledge")
        prior_set_layout = QVBoxLayout(prior_set_group)
        
        # Add info label
        info_label = QLabel("Click on a parameter to set prior knowledge:")
        info_label.setWordWrap(True)
        prior_set_layout.addWidget(info_label)
        
        # Create scrollable area for parameter buttons
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setMinimumHeight(300)
        
        param_content = QWidget()
        self.param_buttons_layout = QVBoxLayout(param_content)
        
        # Parameter buttons will be added dynamically in update_ui_from_model
        param_scroll.setWidget(param_content)
        prior_set_layout.addWidget(param_scroll)
        
        left_panel.addWidget(prior_set_group)
        
        prior_table_group = QGroupBox("Current Prior Knowledge")
        prior_table_layout = QVBoxLayout(prior_table_group)
        
        self.prior_table = QTableWidget()
        self.prior_table.setColumnCount(4)
        self.prior_table.setHorizontalHeaderLabels(["Parameter", "Type", "Value", "Confidence"])
        self.prior_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.prior_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.prior_table.itemSelectionChanged.connect(self.on_prior_selected)
        
        prior_table_layout.addWidget(self.prior_table)
        
        left_panel.addWidget(prior_table_group)
        
        right_panel = QVBoxLayout()
        
        viz_group = QGroupBox("Prior Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.prior_canvas = MplCanvas(self, width=5, height=4)
        viz_layout.addWidget(self.prior_canvas)
        
        viz_control_layout = QHBoxLayout()
        viz_control_layout.addWidget(QLabel("Visualize Parameter:"))
        
        self.viz_param_combo = QComboBox()
        viz_control_layout.addWidget(self.viz_param_combo, 1)
        
        self.update_viz_btn = QPushButton("Update Visualization")
        self.update_viz_btn.clicked.connect(self.update_prior_plot)
        viz_control_layout.addWidget(self.update_viz_btn)
        
        viz_layout.addLayout(viz_control_layout)
        
        right_panel.addWidget(viz_group)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 1)
        
        self.tab_widget.addTab(prior_tab, "Prior Knowledge")

    def setup_experiment_tab(self):
        experiment_tab = QWidget()
        layout = QHBoxLayout(experiment_tab)
        
        left_panel = QVBoxLayout()
        
        round_group = QGroupBox("Current Round")
        round_layout = QVBoxLayout(round_group)
        
        self.current_round_label = QLabel("0")
        self.current_round_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.current_round_label.setAlignment(Qt.AlignCenter)
        
        round_frame = QFrame()
        round_frame.setFrameShape(QFrame.StyledPanel)
        round_frame.setStyleSheet("background-color: #e6f2ff; border-radius: 8px;")
        round_frame_layout = QVBoxLayout(round_frame)
        round_frame_layout.addWidget(self.current_round_label)
        
        round_layout.addWidget(round_frame)
        
        left_panel.addWidget(round_group)
        
        workflow_group = QGroupBox("Optimization Workflow")
        workflow_layout = QVBoxLayout(workflow_group)
        
        step1_group = QGroupBox("Step 1: Initial Experiments")
        step1_layout = QFormLayout(step1_group)
        
        self.n_initial_spin = QSpinBox()
        self.n_initial_spin.setRange(3, 20)
        self.n_initial_spin.setValue(5)
        
        self.design_method_combo = QComboBox()
        self.design_method_combo.addItems(["TPE", "Latin Hypercube", "Random", "Sobol"])
        
        self.generate_initial_btn = QPushButton("Generate Initial Experiments")
        self.generate_initial_btn.clicked.connect(self.generate_initial_experiments)
        
        step1_layout.addRow("Number of initial experiments:", self.n_initial_spin)
        step1_layout.addRow("Design method:", self.design_method_combo)
        step1_layout.addRow(self.generate_initial_btn)
        
        workflow_layout.addWidget(step1_group)
        
        step2_group = QGroupBox("Step 2: Run & Input Results")
        step2_layout = QVBoxLayout(step2_group)
        
        step2_layout.addWidget(QLabel("Select an experiment from the table and click 'Add Result'"))
        
        result_layout = QHBoxLayout()
        result_layout.addWidget(QLabel("Experiments with results:"))
        self.results_count_label = QLabel("0 / 0")
        result_layout.addWidget(self.results_count_label)
        
        step2_layout.addLayout(result_layout)
        
        self.add_result_btn = QPushButton("Add Result for Selected")
        self.add_result_btn.clicked.connect(self.add_result_for_selected)
        step2_layout.addWidget(self.add_result_btn)
        
        workflow_layout.addWidget(step2_group)
        
        step3_group = QGroupBox("Step 3: Generate Next Round")
        step3_layout = QVBoxLayout(step3_group)
        
        next_layout = QHBoxLayout()
        next_layout.addWidget(QLabel("Additional experiments:"))
        self.n_next_spin = QSpinBox()
        self.n_next_spin.setRange(1, 10)
        self.n_next_spin.setValue(3)
        next_layout.addWidget(self.n_next_spin)
        
        step3_layout.addLayout(next_layout)
        
        explore_layout = QVBoxLayout()
        explore_layout.addWidget(QLabel("Exploration-Exploitation Balance:"))
        
        slider_layout = QHBoxLayout()
        self.exploit_slider = QSlider(Qt.Horizontal)
        self.exploit_slider.setRange(0, 100)
        self.exploit_slider.setValue(70)
        slider_layout.addWidget(self.exploit_slider)
        
        explore_layout.addLayout(slider_layout)
        
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Explore"))
        label_layout.addStretch()
        label_layout.addWidget(QLabel("Exploit"))
        
        explore_layout.addLayout(label_layout)
        
        step3_layout.addLayout(explore_layout)
        
        self.generate_next_btn = QPushButton("Generate Next Experiments")
        self.generate_next_btn.clicked.connect(self.generate_next_experiments)
        step3_layout.addWidget(self.generate_next_btn)
        
        workflow_layout.addWidget(step3_group)
        
        workflow_layout.addWidget(QFrame(frameShape=QFrame.HLine))
        
        stats_group = QGroupBox("Optimization Progress")
        stats_layout = QFormLayout(stats_group)
        
        self.total_exp_label = QLabel("0")
        self.best_result_label = QLabel("N/A")
        self.est_rounds_label = QLabel("N/A")
        
        stats_layout.addRow("Total experiments:", self.total_exp_label)
        stats_layout.addRow("Best result so far:", self.best_result_label)
        stats_layout.addRow("Estimated rounds to converge:", self.est_rounds_label)
        
        workflow_layout.addWidget(stats_group)
        
        left_panel.addWidget(workflow_group)
        
        right_panel = QVBoxLayout()
        
        exp_group = QGroupBox("Experimental Plan")
        exp_layout = QVBoxLayout(exp_group)
        
        self.experiment_table = ExperimentTable()
        self.experiment_table.update_columns(self.model)
        
        exp_layout.addWidget(self.experiment_table)
        
        right_panel.addWidget(exp_group)
        
        layout.addLayout(left_panel)
        layout.addLayout(right_panel, 1)
        
        self.tab_widget.addTab(experiment_tab, "Experiment Design")
        
    def setup_results_tab(self):
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        tabs = QTabWidget()
        
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Optimization History", "Parameter Importance", "Parameter Contour", "Objective Correlation"])
        control_layout.addWidget(self.plot_type_combo)
        
        control_layout.addWidget(QLabel("X-axis:"))
        self.x_param_combo = QComboBox()
        control_layout.addWidget(self.x_param_combo)
        
        control_layout.addWidget(QLabel("Y-axis:"))
        self.y_param_combo = QComboBox()
        control_layout.addWidget(self.y_param_combo)
        
        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self.update_results_plot)
        control_layout.addWidget(self.update_plot_btn)
        
        viz_layout.addLayout(control_layout)
        
        self.result_canvas = MplCanvas(self, width=8, height=6)
        viz_layout.addWidget(self.result_canvas)
        
        tabs.addTab(viz_tab, "Result Visualizations")
        
        best_tab = QWidget()
        best_layout = QVBoxLayout(best_tab)
        
        best_control = QHBoxLayout()
        best_control.addWidget(QLabel("Number of top results:"))
        
        self.n_best_spin = QSpinBox()
        self.n_best_spin.setRange(1, 20)
        self.n_best_spin.setValue(5)
        best_control.addWidget(self.n_best_spin)
        
        self.update_best_btn = QPushButton("Update")
        self.update_best_btn.clicked.connect(self.update_best_results)
        best_control.addWidget(self.update_best_btn)
        
        best_layout.addLayout(best_control)
        
        self.best_table = BestResultsTable()
        self.best_table.update_columns(self.model)
        
        best_layout.addWidget(self.best_table)
        
        tabs.addTab(best_tab, "Best Results")
        
        all_tab = QWidget()
        all_layout = QVBoxLayout(all_tab)
        
        all_control = QHBoxLayout()
        
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_results)
        all_control.addWidget(self.export_results_btn)
        
        self.show_details_btn = QPushButton("Show Details")
        self.show_details_btn.clicked.connect(self.show_result_details)
        all_control.addWidget(self.show_details_btn)
        
        all_layout.addLayout(all_control)
        
        self.all_results_table = AllResultsTable()
        
        all_layout.addWidget(self.all_results_table)
        
        tabs.addTab(all_tab, "All Results")
        
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        
        self.corr_canvas = MplCanvas(self, width=8, height=6)
        corr_layout.addWidget(self.corr_canvas)
        
        self.update_corr_btn = QPushButton("Update Correlation Plot")
        self.update_corr_btn.clicked.connect(self.update_correlation_plot)
        corr_layout.addWidget(self.update_corr_btn)
        
        tabs.addTab(corr_tab, "Parameter Correlations")
        
        layout.addWidget(tabs)
        
        self.tab_widget.addTab(results_tab, "Results")
        
    def setup_analysis_tab(self):
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        tabs = QTabWidget()
        
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        model_control = QHBoxLayout()
        model_control.addWidget(QLabel("Objective:"))
        
        self.model_obj_combo = QComboBox()
        for obj in self.model.objectives:
            self.model_obj_combo.addItem(obj)
        model_control.addWidget(self.model_obj_combo)
        
        self.update_model_btn = QPushButton("Update Model Plot")
        self.update_model_btn.clicked.connect(self.update_model_plot)
        model_control.addWidget(self.update_model_btn)
        
        model_layout.addLayout(model_control)
        
        self.model_canvas = MplCanvas(self, width=8, height=6)
        model_layout.addWidget(self.model_canvas)
        
        tabs.addTab(model_tab, "Prediction Model")
        
        surface_tab = QWidget()
        surface_layout = QVBoxLayout(surface_tab)
        
        surface_control = QHBoxLayout()
        surface_control.addWidget(QLabel("Objective:"))
        
        self.surface_obj_combo = QComboBox()
        for obj in self.model.objectives:
            self.surface_obj_combo.addItem(obj)
        surface_control.addWidget(self.surface_obj_combo)
        
        surface_control.addWidget(QLabel("X Parameter:"))
        self.surface_x_combo = QComboBox()
        surface_control.addWidget(self.surface_x_combo)
        
        surface_control.addWidget(QLabel("Y Parameter:"))
        self.surface_y_combo = QComboBox()
        surface_control.addWidget(self.surface_y_combo)
        
        self.update_surface_btn = QPushButton("Update Surface")
        self.update_surface_btn.clicked.connect(self.update_surface_plot)
        surface_control.addWidget(self.update_surface_btn)
        
        surface_layout.addLayout(surface_control)
        
        self.surface_canvas = Mpl3DCanvas(self, width=8, height=6)
        surface_layout.addWidget(self.surface_canvas)
        
        tabs.addTab(surface_tab, "Response Surface")
        
        convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(convergence_tab)
        
        self.update_convergence_btn = QPushButton("Update Convergence Plot")
        self.update_convergence_btn.clicked.connect(self.update_convergence_plot)
        convergence_layout.addWidget(self.update_convergence_btn)
        
        self.convergence_canvas = MplCanvas(self, width=8, height=6)
        convergence_layout.addWidget(self.convergence_canvas)
        
        tabs.addTab(convergence_tab, "Convergence Analysis")
        
        layout.addWidget(tabs)
        
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
    def log(self, message):
        self.log_display.log(message)
        self.status_label.setText(message.strip())
        QApplication.processEvents()
        
    def update_ui_from_model(self):
        """Update all UI elements based on current model state."""
        # Update parameter table
        self.param_table.update_from_model(self.model)
        
        # Update prior table
        self.update_prior_table()
        
        # Update parameter combos
        self.update_parameter_combos()
        
        # Update parameter buttons in prior tab
        self.update_prior_param_buttons()
        
        # Update experiment table
        self.experiment_table.update_columns(self.model)
        self.experiment_table.update_from_planned(self.model, self.round_start_indices)
        
        # Update best results
        self.best_table.update_columns(self.model)
        self.best_table.update_from_model(self.model, self.n_best_spin.value())
        
        # Update results table
        self.all_results_table.update_from_model(self.model)
        
        # Update experiment counters
        self.current_round_label.setText(str(self.current_round))
        if self.model.experiments and self.model.planned_experiments:
            self.results_count_label.setText(f"{len(self.model.experiments)} / {len(self.model.planned_experiments)}")
            self.total_exp_label.setText(str(len(self.model.planned_experiments)))
            
        # Update progress
        completed = len(self.model.experiments)
        total = len(self.model.planned_experiments) if self.model.planned_experiments else 0
        if total > 0:
            progress = int(100 * completed / total)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)
            
    def update_parameter_combos(self):
        """Update parameter dropdown selections in various UI elements."""
        # Store current selections
        prior_param = self.prior_param_combo.currentText() if self.prior_param_combo.count() > 0 else ""
        viz_param = self.viz_param_combo.currentText() if self.viz_param_combo.count() > 0 else ""
        x_param = self.x_param_combo.currentText() if self.x_param_combo.count() > 0 else ""
        y_param = self.y_param_combo.currentText() if self.y_param_combo.count() > 0 else ""
        
        # Clear existing items
        self.prior_param_combo.clear()
        self.viz_param_combo.clear()
        self.x_param_combo.clear()
        self.y_param_combo.clear()
        
        # Add parameters to combos
        param_names = list(self.model.parameters.keys())
        
        self.prior_param_combo.addItems(param_names)
        self.viz_param_combo.addItems(param_names)
        self.x_param_combo.addItems(param_names)
        self.y_param_combo.addItems(param_names)
        
        # Restore selections if possible
        if prior_param in param_names:
            self.prior_param_combo.setCurrentText(prior_param)
            
        if viz_param in param_names:
            self.viz_param_combo.setCurrentText(viz_param)
            
        if x_param in param_names:
            self.x_param_combo.setCurrentText(x_param)
            
        if y_param in param_names:
            self.y_param_combo.setCurrentText(y_param)
            
        # Update prior UI if needed
        if self.prior_param_combo.count() > 0:
            self.update_prior_ui()
    
    def update_prior_table(self):
        self.prior_table.setRowCount(0)
        
        for name, param in self.model.parameters.items():
            # Check for numerical prior or categorical preferences
            has_prior = param.prior_mean is not None or (
                param.param_type == "categorical" and 
                hasattr(param, 'categorical_preferences') and 
                param.categorical_preferences
            )
            
            if has_prior:
                row = self.prior_table.rowCount()
                self.prior_table.insertRow(row)
                
                if param.param_type == "categorical":
                    value_str = "Preferences set"
                    confidence = "N/A"
                else:
                    value_str = f"{param.prior_mean:.4f}"
                    
                    param_range = param.high - param.low
                    std_ratio = param.prior_std / param_range
                    
                    if std_ratio < 0.05:
                        confidence = "Very High"
                    elif std_ratio < 0.1:
                        confidence = "High"
                    elif std_ratio < 0.2:
                        confidence = "Medium"
                    elif std_ratio < 0.3:
                        confidence = "Low"
                    else:
                        confidence = "Very Low"
                
                self.prior_table.setItem(row, 0, QTableWidgetItem(name))
                self.prior_table.setItem(row, 1, QTableWidgetItem(param.param_type.capitalize()))
                self.prior_table.setItem(row, 2, QTableWidgetItem(value_str))
                self.prior_table.setItem(row, 3, QTableWidgetItem(confidence))
                
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
            self.update_parameter_combos()
            self.experiment_table.update_columns(self.model)
            self.best_table.update_columns(self.model)
            
            self.log(f"-- Parameter '{param.name}' added - Success")
            
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
                self.log(f"-- Parameter '{name}' updated - Success")
                
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
            self.update_parameter_combos()
            self.experiment_table.update_columns(self.model)
            self.best_table.update_columns(self.model)
            
            self.log(f"-- Parameter '{name}' removed - Success")

    def statistical_analysis(self):
        """Perform statistical analysis on the experimental results"""
        if not self.model.experiments:
            QMessageBox.information(self, "Information", "No experiment data available for analysis.")
            return
        
        analysis_dialog = QDialog(self)
        analysis_dialog.setWindowTitle("Statistical Analysis")
        analysis_dialog.resize(800, 600)
        
        layout = QVBoxLayout(analysis_dialog)
        
        # Create tabs for different analyses
        tabs = QTabWidget()
        
        # Summary statistics tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_layout.addWidget(summary_text)
        
        generate_summary_btn = QPushButton("Generate Summary Statistics")
        generate_summary_btn.clicked.connect(lambda: generate_summary())
        summary_layout.addWidget(generate_summary_btn)
        
        tabs.addTab(summary_tab, "Summary Statistics")
        
        # Regression analysis tab
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
        
        # ANOVA tab
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
            
            # Create a DataFrame from experiments
            import pandas as pd
            
            data = []
            for exp in self.model.experiments:
                row = {}
                for param_name in self.model.parameters:
                    if param_name in exp['params']:
                        row[param_name] = exp['params'][param_name]
                
                for obj in self.model.objectives:
                    if obj in exp['results']:
                        row[obj] = exp['results'][obj] * 100.0  # Convert to percentage
                
                data.append(row)
            
            if not data:
                summary_text.setText("No data available for analysis.")
                return
            
            df = pd.DataFrame(data)
            
            # Generate summary statistics
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
                import pandas as pd
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                
                # Create a DataFrame from experiments
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
                        row[target] = exp['results'][target] * 100.0  # Convert to percentage
                        data.append(row)
                
                if not data:
                    regression_result.setText(f"No data available with {target} results.")
                    return
                
                df = pd.DataFrame(data)
                
                # Filter categorical variables and create dummy variables
                cat_columns = []
                for param_name, param in self.model.parameters.items():
                    if param.param_type == "categorical" and param_name in df.columns:
                        cat_columns.append(param_name)
                
                if cat_columns:
                    df = pd.get_dummies(df, columns=cat_columns, drop_first=True)
                
                # Build formula for regression
                formula_parts = []
                for col in df.columns:
                    if col != target:
                        formula_parts.append(col)
                
                if not formula_parts:
                    regression_result.setText("No parameters available for regression.")
                    return
                
                formula = f"{target} ~ " + " + ".join(formula_parts)
                
                # Run regression
                model = ols(formula, data=df).fit()
                
                # Format results
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
                
                # Add significance indicators
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
                import pandas as pd
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                
                # Create a DataFrame from experiments
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
                
                # Run ANOVA
                formula = f"{response} ~ C({factor})"
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # Format results
                result_text = "## ANOVA Results\n\n"
                result_text += f"Response Variable: {response}\n"
                result_text += f"Factor: {factor}\n\n"
                
                result_text += anova_table.to_string() + "\n\n"
                
                # Add interpretation
                f_value = anova_table.iloc[0, 2]
                p_value = anova_table.iloc[0, 3]
                
                result_text += "### Interpretation\n\n"
                
                if p_value < 0.05:
                    result_text += f"The factor '{factor}' has a statistically significant effect on {response} (F={f_value:.4f}, p={p_value:.4f}).\n\n"
                else:
                    result_text += f"The factor '{factor}' does not have a statistically significant effect on {response} (F={f_value:.4f}, p={p_value:.4f}).\n\n"
                
                # Add group means
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
        
        # Control panel
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
        
        # Experiment table
        exp_table = QTableWidget()
        exp_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(exp_table)
        
        # Diversity visualization
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
            # Clear existing items
            exp_table.clear()
            
            # Get parameters
            n = n_experiments.value()
            diversity_weight = diversity_slider.value() / 100.0
            
            # Generate a large pool of candidate experiments
            pool_size = n * 10
            candidate_pool = []
            
            try:
                for _ in range(pool_size):
                    params = {}
                    for name, param in self.model.parameters.items():
                        params[name] = param.suggest_value()
                    
                    # Predict outcome (simplified prediction)
                    if self.model.experiments:
                        distances = []
                        for exp in self.model.experiments:
                            dist = self.model.calculate_experiment_distance(params, exp['params'])
                            if 'score' in exp:
                                distances.append((dist, exp['score']))
                        
                        if distances:
                            # Use inverse distance weighting for prediction
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
                
                # Sort by predicted performance
                candidate_pool.sort(key=lambda x: x['predicted_score'], reverse=True)
                
                # Select diverse subset
                selected = self.model.select_diverse_subset(candidate_pool, n, diversity_weight)
                
                # Update table
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
                
                # Update diversity visualization
                viz_canvas.axes.clear()
                
                if n > 1:
                    # Create distance matrix between selected experiments
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
                    
                    # Add diversity stats
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
            except Exception as e:
                QMessageBox.critical(parallel_dialog, "Error", f"Error generating plan: {str(e)}")
        
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
                # Convert table to DataFrame
                import pandas as pd
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
                
                # Export based on file extension
                if filepath.lower().endswith('.xlsx'):
                    df.to_excel(filepath, index=False)
                    self.log(f"-- Exported plan to {filepath} - Success")
                else:
                    # Default to CSV
                    if not filepath.lower().endswith('.csv'):
                        filepath += '.csv'
                    df.to_csv(filepath, index=False)
                    self.log(f"-- Exported plan to {filepath} - Success")
                
                QMessageBox.information(self, "Success", f"Plan exported to {filepath}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting plan: {str(e)}")
        
        generate_plan()
        export_btn.setEnabled(False)  # Disable export button until plan is generated
        
        parallel_dialog.exec()

    def calculate_experiment_distance(self, exp1, exp2, parameters):
        return _calculate_parameter_distance(exp1, exp2, parameters)
        
    def new_project(self):
        """Create a new project by resetting the current model."""
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
            
            # Reset UI elements
            self.update_ui_from_model()
            self.tab_widget.setCurrentIndex(0)
            self.log("-- New project created - Success")
    
    def open_project(self):
        """Open a project from a file."""
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
                
                # Update working directory
                self.working_directory = os.path.dirname(file_path)
                
                # Reset round information
                self.current_round = 0
                self.round_start_indices = []
                
                # Calculate round indices from experiments
                if self.model.experiments:
                    rounds_per_experiment = 5  # Assuming 5 experiments per round
                    self.current_round = (len(self.model.experiments) - 1) // rounds_per_experiment + 1
                    
                    for i in range(1, self.current_round + 1):
                        self.round_start_indices.append((i - 1) * rounds_per_experiment)
                
                # Update UI
                self.update_ui_from_model()
                self.log(f"-- Project loaded from {file_path} - Success")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project: {str(e)}")
                self.log(f"-- Error loading project: {str(e)} - Failed")
    
    def save_project(self):
        """Save the current project to a file."""
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
            # Add .bdoe extension if not provided
            if not file_path.lower().endswith('.bdoe'):
                file_path += '.bdoe'
                
            try:
                self.model.save_model(file_path)
                
                # Update working directory
                self.working_directory = os.path.dirname(file_path)
                
                self.log(f"-- Project saved to {file_path} - Success")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
                self.log(f"-- Error saving project: {str(e)} - Failed")
    
    def open_structure_editor(self):
        """Open a molecular structure editor."""
        try:
            # Check if RDKit is available
            import rdkit
            from rdkit.Chem import Draw
            from rdkit import Chem
            
            editor_dialog = QDialog(self)
            editor_dialog.setWindowTitle("Molecular Structure Editor")
            editor_dialog.resize(800, 600)
            
            layout = QVBoxLayout(editor_dialog)
            
            # Add SMILES input
            smiles_layout = QHBoxLayout()
            smiles_layout.addWidget(QLabel("SMILES:"))
            
            smiles_input = QLineEdit()
            smiles_layout.addWidget(smiles_input)
            
            render_btn = QPushButton("Render")
            smiles_layout.addWidget(render_btn)
            
            layout.addLayout(smiles_layout)
            
            # Structure view
            structure_view = QLabel("Enter SMILES to view structure")
            structure_view.setAlignment(Qt.AlignCenter)
            structure_view.setMinimumHeight(300)
            
            layout.addWidget(structure_view)
            
            # Info panel
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
                        # Generate molecule image
                        img = Draw.MolToImage(mol, size=(400, 300))
                        
                        # Convert to QPixmap
                        height, width, channel = img.shape
                        bytes_per_line = 3 * width
                        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        
                        structure_view.setPixmap(pixmap)
                        
                        # Calculate properties
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
        from .dialogs import OptimizationSettingsDialog
        
        dialog = OptimizationSettingsDialog(self, self.model)
        if dialog.exec():
            self.log("-- Optimization settings updated - Success")
            
            # Update the exploit slider with the model's current value
            self.exploit_slider.setValue(int(self.model.exploitation_weight * 100))
    
    def update_results_plot(self):
        """Update the results visualization plot."""
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
                # Use first objective for contour plot
                obj = self.model.objectives[0] if self.model.objectives else "yield"
                self.result_canvas.axes = plot_parameter_contour(
                    self.model, x_param, y_param, self.result_canvas.axes
                )
                
        elif plot_type == "Objective Correlation":
            self.result_canvas.axes = plot_objective_correlation(self.model, self.result_canvas.axes)
        
        self.result_canvas.draw()
        
    def update_best_results(self):
        """Update the best results table."""
        n_best = self.n_best_spin.value()
        self.best_table.update_from_model(self.model, n_best)
        self.log(f"-- Best results table updated showing top {n_best} results - Success")

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
            
            # Get data
            X, _ = self.model._extract_normalized_features_and_targets()
            
            # Extract objective values
            y = []
            for exp in self.model.experiments:
                if 'results' in exp and objective in exp['results']:
                    y.append(exp['results'][objective] * 100.0)  # Convert to percentage
                else:
                    y.append(0.0)
                    
            y = np.array(y)
            
            if len(X) < 3 or len(np.unique(y)) < 2:
                QMessageBox.warning(self, "Warning", "Need at least 3 diverse experiments for model fitting")
                return
                
            # Fit GP model
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
            gp.fit(X, y)
            
            # Get parameter importance for this objective
            importances = {}
            if len(X[0]) > 1:  # At least 2 parameters
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
                        
                # Normalize importances
                max_imp = max(importances.values()) if importances else 1.0
                if max_imp > 0:
                    importances = {k: v / max_imp for k, v in importances.items()}
            
            # Plot actual vs predicted
            y_pred, y_std = gp.predict(X, return_std=True)
            
            self.model_canvas.axes.errorbar(
                y, y_pred, yerr=2*y_std, fmt='o', alpha=0.6, 
                ecolor='gray', capsize=5, markersize=8
            )
            
            # Add perfect prediction line
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            margin = (max_val - min_val) * 0.1
            line_start = min_val - margin
            line_end = max_val + margin
            
            self.model_canvas.axes.plot([line_start, line_end], [line_start, line_end], 
                                'k--', alpha=0.7, label='Perfect Prediction')
            
            # Format plot
            self.model_canvas.axes.set_xlabel(f'Actual {objective} (%)')
            self.model_canvas.axes.set_ylabel(f'Predicted {objective} (%)')
            self.model_canvas.axes.set_title(f'Gaussian Process Model for {objective}')
            
            # Add R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            self.model_canvas.axes.annotate(f'R² = {r2:.3f}', 
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              ha='left', va='top', fontsize=12)
            
            # Add parameter importance text
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
            return

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
            self.log(f"-- Exporting results to {filepath}")
            self.progress_bar.setValue(10)
            
            # Create a pandas DataFrame
            import pandas as pd
            from ..core import settings
            
            # Collect all parameter names and objective names
            param_names = list(self.model.parameters.keys())
            objective_names = self.model.objectives
            
            # Create the data rows
            data = []
            for i, exp in enumerate(self.model.experiments):
                row = {
                    "Experiment ID": i + 1,
                    "Round": (i // 5) + 1,  # Estimate round based on experiment
                    "Timestamp": exp.get("timestamp", "").replace("T", " ")
                }
                
                # Add parameters
                params = exp.get("params", {})
                for param in param_names:
                    if param in params:
                        value = params[param]
                        if isinstance(value, float):
                            value = settings.format_value(value)
                        row[param] = value
                    else:
                        row[param] = ""
                
                # Add results
                results = exp.get("results", {})
                for obj in objective_names:
                    if obj in results and results[obj] is not None:
                        value = results[obj] * 100.0  # Convert to percentage
                        row[f"{obj.capitalize()} (%)"] = settings.format_value(value)
                    else:
                        row[f"{obj.capitalize()} (%)"] = ""
                
                # Add composite score
                if "score" in exp:
                    row["Composite Score (%)"] = settings.format_value(exp["score"] * 100.0)
                
                data.append(row)
            
            self.progress_bar.setValue(50)
            
            # Create DataFrame and export
            df = pd.DataFrame(data)
            
            if filepath.lower().endswith(".xlsx"):
                df.to_excel(filepath, index=False)
            else:
                # Default to CSV
                if not filepath.lower().endswith(".csv"):
                    filepath += ".csv"
                df.to_csv(filepath, index=False)
            
            self.progress_bar.setValue(100)
            self.log(f"-- Results exported to {filepath} - Success")
            
            # Reset progress bar after a delay
            QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))
            
            # Update working directory
            self.working_directory = os.path.dirname(filepath)
            
        except Exception as e:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
            self.log(f"-- Results export failed: {str(e)} - Error")
            
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
            self.log(f"-- Importing data from {filepath}")
            self.progress_bar.setValue(10)
            
            # Import data using pandas
            import pandas as pd
            
            if filepath.lower().endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
                
            self.progress_bar.setValue(30)
            
            # Check required columns
            required_obj = self.model.objectives[0] if self.model.objectives else "yield"
            param_columns = [col for col in df.columns if col in self.model.parameters]
            
            if not param_columns:
                QMessageBox.warning(self, "Warning", 
                                "Could not find any parameter columns matching the current parameters")
                self.progress_bar.setValue(0)
                return
                
            objective_columns = []
            for obj in self.model.objectives:
                # Try various formats
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
            
            # Process the data
            imported_count = 0
            for _, row in df.iterrows():
                params = {}
                for param_name in param_columns:
                    if pd.notnull(row[param_name]):
                        param = self.model.parameters[param_name]
                        value = row[param_name]
                        
                        # Convert to appropriate type
                        if param.param_type == "continuous":
                            value = float(value)
                        elif param.param_type == "discrete":
                            value = int(value)
                        
                        params[param_name] = value
                
                results = {}
                for obj, col_name in objective_columns:
                    if pd.notnull(row[col_name]):
                        # Convert percentage to 0-1 range
                        value = float(row[col_name])
                        if value > 1.0:
                            value /= 100.0
                        results[obj] = value
                
                # Add the experiment
                if params and results:
                    self.model.add_experiment_result(params, results)
                    imported_count += 1
                    
            self.progress_bar.setValue(80)
            
            # Update UI
            self.experiment_table.update_from_planned(self.model, self.round_start_indices)
            self.best_table.update_from_model(self.model, self.n_best_spin.value())
            self.all_results_table.update_from_model(self.model)
            
            completed_count = len(self.model.experiments)
            planned_count = len(self.model.planned_experiments)
            self.results_count_label.setText(f"{completed_count} / {planned_count}")
            
            # Update best result label
            best_exps = self.model.get_best_experiments(n=1)
            if best_exps:
                best_exp = best_exps[0]
                best_score = best_exp.get('score', 0) * 100.0
                self.best_result_label.setText(f"{best_score:.2f}%")
                
            self.progress_bar.setValue(100)
            self.log(f"-- Imported {imported_count} experiments from {filepath} - Success")
            
            # Reset progress bar after a delay
            QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))
            
        except Exception as e:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Error", f"Error importing data: {str(e)}")
            self.log(f"-- Data import failed: {str(e)} - Error")
            
    def show_preferences(self):
        prefs_dialog = QDialog(self)
        prefs_dialog.setWindowTitle("Preferences")
        prefs_dialog.resize(500, 400)
        
        layout = QVBoxLayout(prefs_dialog)
        
        # Create tabs for different preference categories
        tabs = QTabWidget()
        
        # Display preferences tab
        display_tab = QWidget()
        display_layout = QFormLayout(display_tab)
        
        # Rounding settings (mirror the status bar controls)
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
        
        # UI theme selection
        theme_combo = QComboBox()
        theme_combo.addItems(["Light", "Dark", "System"])
        theme_combo.setCurrentText("Light")  # Default
        display_layout.addRow("UI Theme:", theme_combo)
        
        # Font size
        font_size_spin = QSpinBox()
        font_size_spin.setRange(8, 18)
        font_size_spin.setValue(10)
        display_layout.addRow("UI Font Size:", font_size_spin)
        
        tabs.addTab(display_tab, "Display")
        
        # Modeling preferences tab
        model_tab = QWidget()
        model_layout = QFormLayout(model_tab)
        
        # Acquisition function (similar to optimization settings)
        acq_function_combo = QComboBox()
        acq_function_combo.addItems([
            "Expected Improvement (EI)", 
            "Probability of Improvement (PI)", 
            "Upper Confidence Bound (UCB)"
        ])
        
        # Set current selection based on model
        if self.model.acquisition_function == "ei":
            acq_function_combo.setCurrentIndex(0)
        elif self.model.acquisition_function == "pi":
            acq_function_combo.setCurrentIndex(1)
        elif self.model.acquisition_function == "ucb":
            acq_function_combo.setCurrentIndex(2)
            
        model_layout.addRow("Default Acquisition Function:", acq_function_combo)
        
        # Default exploitation weight
        exploit_spin = QDoubleSpinBox()
        exploit_spin.setRange(0.0, 1.0)
        exploit_spin.setSingleStep(0.05)
        exploit_spin.setDecimals(2)
        exploit_spin.setValue(self.model.exploitation_weight)
        model_layout.addRow("Default Exploitation Weight:", exploit_spin)
        
        tabs.addTab(model_tab, "Modeling")
        
        # Add tabs to layout
        layout.addWidget(tabs)
        
        # Buttons
        button_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        
        save_btn.clicked.connect(prefs_dialog.accept)
        cancel_btn.clicked.connect(prefs_dialog.reject)
        
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        
        layout.addLayout(button_box)
        
        # Handle dialog result
        if prefs_dialog.exec():
            # Save display preferences
            settings.auto_round = auto_round_check.isChecked()
            settings.rounding_precision = precision_spin.value()
            settings.smart_rounding = smart_round_check.isChecked()
            
            # Update status bar controls
            self.auto_round_check.setChecked(settings.auto_round)
            self.precision_spin.setValue(settings.rounding_precision)
            self.smart_round_check.setChecked(settings.smart_rounding)
            
            # Save modeling preferences
            acq_index = acq_function_combo.currentIndex()
            if acq_index == 0:
                self.model.acquisition_function = "ei"
            elif acq_index == 1:
                self.model.acquisition_function = "pi"
            elif acq_index == 2:
                self.model.acquisition_function = "ucb"
                
            self.model.exploitation_weight = exploit_spin.value()
            
            # UI theme (placeholder for future implementation)
            theme = theme_combo.currentText()
            
            self.log("-- Preferences updated - Success")
        
    def show_documentation(self):
        doc_dialog = QDialog(self)
        doc_dialog.setWindowTitle("Documentation")
        doc_dialog.resize(700, 500)
        
        layout = QVBoxLayout(doc_dialog)
        
        tabs = QTabWidget()
        
        # Overview tab
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
        
        # Parameters tab
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
        
        # Optimization tab
        opt_tab = QWidget()
        opt_layout = QVBoxLayout(opt_tab)
        
        opt_text = QTextEdit()
        opt_text.setReadOnly(True)
        opt_text.setHtml("""
        <h2>Optimization Process</h2>
        
        <h3>Acquisition Functions:</h3>
        <ul>
            <li><b>Expected Improvement (EI)</b>: Balances exploration and exploitation, good default choice</li>
            <li><b>Probability of Improvement (PI)</b>: More focused on exploitation around promising areas</li>
            <li><b>Upper Confidence Bound (UCB)</b>: Can be tuned for more exploration of uncertain areas</li>
        </ul>
        
        <h3>Exploration vs. Exploitation:</h3>
        <p>The exploration-exploitation slider controls how the algorithm balances:</p>
        <ul>
            <li><b>Exploration</b>: Trying diverse conditions to avoid missing global optima</li>
            <li><b>Exploitation</b>: Focusing on areas known to be promising</li>
        </ul>
        <p>Higher values favor exploitation, while lower values favor exploration.</p>
        
        <h3>Advanced Settings:</h3>
        <ul>
            <li><b>Thompson Sampling</b>: Adds controlled randomness to improve diversity</li>
            <li><b>GP Modeling</b>: Uses Gaussian Processes to model the response surface</li>
            <li><b>Sigmoid Transformation</b>: Accelerates convergence to 100% yield</li>
        </ul>
        """)
        
        opt_layout.addWidget(opt_text)
        tabs.addTab(opt_tab, "Optimization")
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        analysis_text = QTextEdit()
        analysis_text.setReadOnly(True)
        analysis_text.setHtml("""
        <h2>Analysis Tools</h2>
        
        <h3>Available Visualizations:</h3>
        <ul>
            <li><b>Optimization History</b>: Shows improvement of objectives over time</li>
            <li><b>Parameter Importance</b>: Reveals which parameters have the most impact</li>
            <li><b>Parameter Contour</b>: Shows how two parameters interact to affect an objective</li>
            <li><b>Objective Correlation</b>: Shows relationships between different objectives</li>
            <li><b>Response Surface</b>: 3D visualization of parameter effects</li>
            <li><b>Convergence Analysis</b>: Tracks optimization progress</li>
        </ul>
        
        <h3>Statistical Analysis:</h3>
        <ul>
            <li><b>Summary Statistics</b>: Basic statistical measures for parameters and objectives</li>
            <li><b>Regression Analysis</b>: Fit models to understand parameter effects</li>
            <li><b>ANOVA</b>: Test significance of categorical parameter effects</li>
        </ul>
        
        <h3>Model Analysis:</h3>
        <p>The prediction model shows how well the Bayesian model is fitting the data and provides parameter importance scores.</p>
        """)
        
        analysis_layout.addWidget(analysis_text)
        tabs.addTab(analysis_tab, "Analysis")
        
        layout.addWidget(tabs)
        
        # Close button
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
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(about_dialog.accept)
        layout.addWidget(close_btn)
        
        about_dialog.exec()

    def update_std_from_confidence(self, confidence):
        """Update standard deviation based on confidence level."""
        param_name = self.prior_param_combo.currentText()
        if not param_name or param_name not in self.model.parameters:
            return
            
        param = self.model.parameters[param_name]
        
        if param.param_type in ["continuous", "discrete"]:
            param_range = param.high - param.low
            
            if confidence == "Very High":
                std = param_range * 0.05
            elif confidence == "High":
                std = param_range * 0.1
            elif confidence == "Medium":
                std = param_range * 0.2
            elif confidence == "Low":
                std = param_range * 0.3
            else:  # Very Low
                std = param_range * 0.4
                
            if param.param_type == "discrete":
                std = max(1, int(std))
                
            self.prior_std_spin.setValue(std)
            self.update_prior_plot()
    
    def add_or_update_prior(self):
        """Add or update prior for the selected parameter."""
        param_name = self.prior_param_combo.currentText()
        if not param_name or param_name not in self.model.parameters:
            QMessageBox.warning(self, "Warning", "Please select a parameter first.")
            return
            
        param = self.model.parameters[param_name]
        
        if param.param_type in ["continuous", "discrete"]:
            mean = self.prior_mean_spin.value()
            std = self.prior_std_spin.value()
            
            if mean < param.low or mean > param.high:
                QMessageBox.warning(self, "Warning", 
                    f"Mean value {mean} is outside parameter range [{param.low}, {param.high}].")
                return
                
            param.set_prior(mean=mean, std=std)
            self.log(f"-- Prior set for {param_name} (mean={mean}, std={std}) - Success")
            
        else:  # categorical
            # Get values from UI elements for categorical parameter
            preferences = {}
            
            # This assumes sliders are created in update_prior_ui method
            for i in range(self.categorical_layout.count()):
                item = self.categorical_layout.itemAt(i)
                if isinstance(item, QHBoxLayout):
                    label_item = item.itemAt(0)
                    slider_item = item.itemAt(1)
                    value_item = item.itemAt(2)
                    
                    if label_item and slider_item:
                        label = label_item.widget()
                        slider = slider_item.widget()
                        
                        if label and slider and isinstance(slider, QSlider):
                            category = label.text()
                            value = slider.value()
                            preferences[category] = value
            
            param.categorical_preferences = preferences
            self.log(f"-- Categorical prior set for {param_name} - Success")
        
        self.update_prior_table()
        self.update_prior_plot()
    
    def remove_prior(self):
        """Remove prior for the selected parameter."""
        param_name = self.prior_param_combo.currentText()
        if not param_name or param_name not in self.model.parameters:
            QMessageBox.warning(self, "Warning", "Please select a parameter first.")
            return
            
        param = self.model.parameters[param_name]
        
        if param.param_type in ["continuous", "discrete"]:
            param.set_prior(None, None)
        else:
            param.categorical_preferences = None
            
        self.log(f"-- Prior removed for {param_name} - Success")
        self.update_prior_table()
        self.update_prior_plot()
    
    def on_prior_selected(self):
        """Handle selection of a prior from the table."""
        selected_items = self.prior_table.selectedItems()
        if not selected_items:
            return
            
        param_name = selected_items[0].text()
        if param_name in self.model.parameters:
            # Select this parameter in the combo box
            self.prior_param_combo.setCurrentText(param_name)
            self.update_prior_ui()
            self.update_prior_plot()
    
    def update_prior_ui(self):
        """Update the prior UI based on the selected parameter."""
        param_name = self.prior_param_combo.currentText()
        if not param_name or param_name not in self.model.parameters:
            return
            
        param = self.model.parameters[param_name]
        
        # Clear categorical layout
        for i in reversed(range(self.categorical_layout.count())):
            item = self.categorical_layout.itemAt(i)
            if item:
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    # Clear sub-layout widgets
                    for j in reversed(range(item.layout().count())):
                        sub_item = item.layout().itemAt(j)
                        if sub_item and sub_item.widget():
                            sub_item.widget().deleteLater()
                    self.categorical_layout.removeItem(item)
        
        if param.param_type in ["continuous", "discrete"]:
            # Set continuous widget
            self.prior_ui_stack.setCurrentIndex(0)
            
            # Update range for mean spin
            self.prior_mean_spin.setRange(param.low, param.high)
            
            # Set values if there's a prior
            if param.prior_mean is not None and param.prior_std is not None:
                self.prior_mean_spin.setValue(param.prior_mean)
                self.prior_std_spin.setValue(param.prior_std)
                
                # Set confidence level based on std
                param_range = param.high - param.low
                std_ratio = param.prior_std / param_range
                
                if std_ratio <= 0.05:
                    self.prior_confidence_combo.setCurrentText("Very High")
                elif std_ratio <= 0.1:
                    self.prior_confidence_combo.setCurrentText("High")
                elif std_ratio <= 0.2:
                    self.prior_confidence_combo.setCurrentText("Medium")
                elif std_ratio <= 0.3:
                    self.prior_confidence_combo.setCurrentText("Low")
                else:
                    self.prior_confidence_combo.setCurrentText("Very Low")
            else:
                # Default values
                self.prior_mean_spin.setValue((param.high + param.low) / 2)
                self.prior_confidence_combo.setCurrentText("Medium")
                self.update_std_from_confidence("Medium")
                
        else:  # categorical
            # Set categorical widget
            self.prior_ui_stack.setCurrentIndex(1)
            
            # Add slider for each category
            self.categorical_layout.addWidget(QLabel("Set preference weights for each category:"))
            
            for i, choice in enumerate(param.choices):
                row_layout = QHBoxLayout()
                
                label = QLabel(choice)
                label.setMinimumWidth(100)
                
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 10)
                
                # Set value if there's a prior
                value = 5
                if hasattr(param, 'categorical_preferences') and param.categorical_preferences:
                    value = param.categorical_preferences.get(choice, 5)
                    
                slider.setValue(value)
                
                value_label = QLabel(str(value))
                slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
                
                row_layout.addWidget(label)
                row_layout.addWidget(slider, 1)
                row_layout.addWidget(value_label)
                
                self.categorical_layout.addLayout(row_layout)
    
    def update_prior_plot(self):
        """Update the prior visualization plot."""
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
            import numpy as np
            from scipy import stats
            
            # Check if this parameter has a prior
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
                
        else:  # categorical
            # Check if this parameter has preferences
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
                
                # Rotate x labels if there are many categories
                if len(categories) > 5:
                    plt.setp(self.prior_canvas.axes.get_xticklabels(), rotation=45, ha="right")
            else:
                self.prior_canvas.axes.text(0.5, 0.5, "No preferences defined for this parameter", 
                    ha='center', va='center', transform=self.prior_canvas.axes.transAxes)
        
        self.prior_canvas.draw()
        
    def update_prior_table(self):
        """Update the prior table with current prior settings."""
        self.prior_table.setRowCount(0)
        
        for name, param in self.model.parameters.items():
            if param.param_type in ["continuous", "discrete"] and param.prior_mean is not None:
                row = self.prior_table.rowCount()
                self.prior_table.insertRow(row)
                
                self.prior_table.setItem(row, 0, QTableWidgetItem(name))
                self.prior_table.setItem(row, 1, QTableWidgetItem(param.param_type.capitalize()))
                
                value_str = f"{param.prior_mean:.4g}"
                if param.units:
                    value_str += f" {param.units}"
                self.prior_table.setItem(row, 2, QTableWidgetItem(value_str))
                
                # Calculate confidence level
                param_range = param.high - param.low
                std_ratio = param.prior_std / param_range
                
                confidence = "Medium"
                if std_ratio <= 0.05:
                    confidence = "Very High"
                elif std_ratio <= 0.1:
                    confidence = "High"
                elif std_ratio <= 0.2:
                    confidence = "Medium"
                elif std_ratio <= 0.3:
                    confidence = "Low"
                else:
                    confidence = "Very Low"
                    
                self.prior_table.setItem(row, 3, QTableWidgetItem(confidence))
                
            elif param.param_type == "categorical" and hasattr(param, 'categorical_preferences') and param.categorical_preferences:
                row = self.prior_table.rowCount()
                self.prior_table.insertRow(row)
                
                self.prior_table.setItem(row, 0, QTableWidgetItem(name))
                self.prior_table.setItem(row, 1, QTableWidgetItem("Categorical"))
                
                # Find highest preference item
                max_pref = 0
                max_item = ""
                
                for choice, pref in param.categorical_preferences.items():
                    if pref > max_pref:
                        max_pref = pref
                        max_item = choice
                
                if max_item:
                    self.prior_table.setItem(row, 2, QTableWidgetItem(f"Prefer: {max_item}"))
                else:
                    self.prior_table.setItem(row, 2, QTableWidgetItem("Equal weights"))
                    
                self.prior_table.setItem(row, 3, QTableWidgetItem("Custom"))
    
    def generate_initial_experiments(self):
        """Generate initial experiments for the first round."""
        if not self.model.parameters:
            QMessageBox.warning(self, "Warning", "Define parameters first before generating experiments.")
            return
            
        n_exps = self.n_initial_spin.value()
        method = self.design_method_combo.currentText()
        
        # Reset model planned experiments
        self.model.planned_experiments = []
        self.round_start_indices = [0]
        self.current_round = 1
        
        # Generate initial experiments
        try:
            # For now, use Optuna's built-in mechanism
            experiments = self.model.suggest_experiments(n_suggestions=n_exps)
            self.model.planned_experiments.extend(experiments)
            
            # Update UI
            self.experiment_table.update_columns(self.model)
            self.experiment_table.update_from_planned(self.model, self.round_start_indices)
            self.current_round_label.setText(str(self.current_round))
            self.results_count_label.setText(f"0 / {n_exps}")
            self.total_exp_label.setText(str(len(self.model.planned_experiments)))
            
            self.log(f"-- Generated {n_exps} initial experiments using {method} - Success")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate experiments: {str(e)}")
            self.log(f"-- Failed to generate experiments: {str(e)} - Error")
    
    def generate_next_experiments(self):
        """Generate the next round of experiments."""
        if not self.model.parameters:
            QMessageBox.warning(self, "Warning", "Define parameters first before generating experiments.")
            return
            
        if not self.model.experiments:
            QMessageBox.warning(self, "Warning", "Add results for initial experiments before generating next round.")
            return
            
        n_exps = self.n_next_spin.value()
        exploit_weight = self.exploit_slider.value() / 100.0
        
        # Set exploitation weight in model
        self.model.exploitation_weight = exploit_weight
        
        # Generate next experiments
        try:
            experiments = self.model.suggest_experiments(n_suggestions=n_exps)
            
            # Add to planned experiments and advance round
            self.round_start_indices.append(len(self.model.planned_experiments))
            self.model.planned_experiments.extend(experiments)
            self.current_round += 1
            
            # Update UI
            self.experiment_table.update_from_planned(self.model, self.round_start_indices)
            self.current_round_label.setText(str(self.current_round))
            
            completed = len(self.model.experiments)
            self.results_count_label.setText(f"{completed} / {n_exps}")
            self.total_exp_label.setText(str(len(self.model.planned_experiments)))
            
            self.log(f"-- Generated {n_exps} next experiments - Success")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate next experiments: {str(e)}")
            self.log(f"-- Failed to generate next experiments: {str(e)} - Error")
    
    def update_surface_plot(self):
        objective = self.surface_obj_combo.currentText()
        
        if not objective or not self.model.experiments:
            QMessageBox.warning(self, "Warning", "No experiments or objectives available")
            return
            
        self.surface_canvas.axes.clear()
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            import numpy as np
            
            # Get data
            X, _ = self.model._extract_normalized_features_and_targets()
            
            # Extract objective values
            y = []
            for exp in self.model.experiments:
                if 'results' in exp and objective in exp['results']:
                    y.append(exp['results'][objective] * 100.0)  # Convert to percentage
                else:
                    y.append(0.0)
                    
            y = np.array(y)
            
            if len(X) < 3 or len(np.unique(y)) < 2:
                QMessageBox.warning(self, "Warning", "Need at least 3 diverse experiments for model fitting")
                return
                
            # Fit GP model
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
            gp.fit(X, y)
            
            # Get parameter importance for this objective
            importances = {}
            if len(X[0]) > 1:  # At least 2 parameters
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
                        
                # Normalize importances
                max_imp = max(importances.values()) if importances else 1.0
                if max_imp > 0:
                    importances = {k: v / max_imp for k, v in importances.items()}
            
            # Plot actual vs predicted
            y_pred, y_std = gp.predict(X, return_std=True)
            
            self.surface_canvas.axes.errorbar(
                X[:, 0], X[:, 1], y, yerr=2*y_std, fmt='o', alpha=0.6, 
                ecolor='gray', capsize=5, markersize=8
            )
            
            # Add perfect prediction line
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            margin = (max_val - min_val) * 0.1
            line_start = min_val - margin
            line_end = max_val + margin
            
            self.surface_canvas.axes.plot([line_start, line_end], [line_start, line_end], 
                                'k--', alpha=0.7, label='Perfect Prediction')
            
            # Format plot
            self.surface_canvas.axes.set_xlabel(f'{self.surface_x_combo.currentText()} {f"({self.model.parameters[self.surface_x_combo.currentText()].units})" if self.model.parameters[self.surface_x_combo.currentText()].units else ""}')
            self.surface_canvas.axes.set_ylabel(f'{self.surface_y_combo.currentText()} {f"({self.model.parameters[self.surface_y_combo.currentText()].units})" if self.model.parameters[self.surface_y_combo.currentText()].units else ""}')
            self.surface_canvas.axes.set_title(f'Gaussian Process Model for {objective}')
            
            # Add R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            self.surface_canvas.axes.annotate(f'R² = {r2:.3f}', 
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              ha='left', va='top', fontsize=12)
            
            # Add parameter importance text
            if importances:
                imp_text = "Parameter Importance:\n"
                for name, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                    imp_text += f"{name}: {imp:.3f}\n"
                    
                self.surface_canvas.axes.annotate(imp_text, 
                                  xy=(0.05, 0.85), xycoords='axes fraction',
                                  ha='left', va='top', fontsize=9)
                
            self.surface_canvas.draw()
            self.log(f"-- Prediction model for {objective} updated - Success")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating surface plot: {str(e)}")
            self.log(f"-- Error updating surface plot: {str(e)}")
            return
    
    def update_convergence_plot(self):
        self.convergence_canvas.axes.clear()
        
        try:
            # Calculate convergence progress
            convergence_progress = []
            for exp in self.model.experiments:
                if 'results' in exp and 'score' in exp:
                    convergence_progress.append(exp['results']['score'])
            
            if not convergence_progress:
                self.convergence_canvas.axes.text(0.5, 0.5, "No convergence data available", 
                    ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
                self.convergence_canvas.draw()
                return
            
            # Plot convergence progress
            self.convergence_canvas.axes.plot(convergence_progress, label='Convergence Progress')
            self.convergence_canvas.axes.set_xlabel('Experiment Number')
            self.convergence_canvas.axes.set_ylabel('Objective Value')
            self.convergence_canvas.axes.set_title('Convergence Analysis')
            self.convergence_canvas.axes.legend()
            
            # Add trend line
            trend_line = np.polyfit(range(len(convergence_progress)), convergence_progress, 1)
            trend_values = np.polyval(trend_line, range(len(convergence_progress)))
            self.convergence_canvas.axes.plot(trend_values, 'r--', label='Trend Line')
            
            self.convergence_canvas.draw()
            self.log("-- Convergence plot updated - Success")
            
        except Exception as e:
            self.convergence_canvas.axes.clear()
            self.convergence_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', transform=self.convergence_canvas.axes.transAxes)
        
        self.convergence_canvas.draw()
    
    def update_correlation_plot(self):
        """Update the correlation plot."""
        if not self.model.experiments or len(self.model.experiments) < 3:
            self.corr_canvas.axes.clear()
            self.corr_canvas.axes.text(0.5, 0.5, "Need at least 3 experiments for correlation analysis", 
                ha='center', va='center', transform=self.corr_canvas.axes.transAxes)
            self.corr_canvas.draw()
            return
            
        try:
            # Create DataFrame from experiments
            import pandas as pd
            
            data = []
            for exp in self.model.experiments:
                row = {}
                for param_name in self.model.parameters:
                    if param_name in exp['params']:
                        # Convert categorical parameters to numerical
                        if self.model.parameters[param_name].param_type == "categorical":
                            choices = self.model.parameters[param_name].choices
                            value = exp['params'][param_name]
                            row[param_name] = choices.index(value) if value in choices else 0
                        else:
                            row[param_name] = float(exp['params'][param_name])
                
                for obj in self.model.objectives:
                    if obj in exp['results']:
                        row[obj] = exp['results'][obj] * 100.0  # Convert to percentage
                
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Calculate correlation matrix
            corr = df.corr()
            
            # Plot correlation matrix
            self.corr_canvas.axes.clear()
            im = self.corr_canvas.axes.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=self.corr_canvas.axes)
            
            # Add labels
            self.corr_canvas.axes.set_xticks(np.arange(len(corr.columns)))
            self.corr_canvas.axes.set_yticks(np.arange(len(corr.columns)))
            self.corr_canvas.axes.set_xticklabels(corr.columns, rotation=45, ha="right")
            self.corr_canvas.axes.set_yticklabels(corr.columns)
            
            # Add correlation values
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    text = self.corr_canvas.axes.text(j, i, f"{corr.iloc[i, j]:.2f}",
                                                ha="center", va="center", 
                                                color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
                                     
            self.corr_canvas.axes.set_title("Parameter-Result Correlation Matrix")
            
        except Exception as e:
            self.corr_canvas.axes.clear()
            self.corr_canvas.axes.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', transform=self.corr_canvas.axes.transAxes)
        
        self.corr_canvas.draw()
    
    def add_result_for_selected(self):
        """Add a result for the selected experiment."""
        selected_items = self.experiment_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Select an experiment first.")
            return
            
        # Get experiment ID from the selection (item in column 1)
        row = selected_items[0].row()
        id_item = self.experiment_table.item(row, 1)
        
        if not id_item or not id_item.text().isdigit():
            QMessageBox.warning(self, "Warning", "Invalid experiment selection.")
            return
            
        exp_id = int(id_item.text()) - 1
        
        if exp_id < 0 or exp_id >= len(self.model.planned_experiments):
            QMessageBox.warning(self, "Warning", "Invalid experiment ID.")
            return
            
        # Check if this experiment already has results
        for i, exp in enumerate(self.model.experiments):
            # This is a simplified check, in a real app we would need more robust checking
            params_match = True
            for param_name, param_value in self.model.planned_experiments[exp_id].items():
                if param_name not in exp['params'] or exp['params'][param_name] != param_value:
                    params_match = False
                    break
                    
            if params_match:
                overwrite = QMessageBox.question(
                    self,
                    "Overwrite Result",
                    "This experiment already has results. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if overwrite == QMessageBox.Yes:
                    # Remove old result
                    self.model.experiments.pop(i)
                    break
                else:
                    return
        
        # Open result dialog
        dialog = ResultDialog(self, self.model, exp_id, self.model.planned_experiments[exp_id])
        
        if dialog.exec() == QDialog.Accepted and dialog.result:
            # Add result to model
            self.model.add_experiment_result(
                params=dialog.result["params"],
                results=dialog.result["results"]
            )
            
            # Update UI
            self.experiment_table.update_from_planned(self.model, self.round_start_indices)
            
            completed = len(self.model.experiments)
            total = len(self.model.planned_experiments)
            self.results_count_label.setText(f"{completed} / {total}")
            
            # Update best result label and other UI elements
            self.update_best_result_label()
            self.update_results_plot()
            self.update_best_results()
            
            # Update all results table
            self.all_results_table.update_from_model(self.model)
            
            self.log(f"-- Added result for experiment #{exp_id+1} - Success")
    
    def update_best_result_label(self):
        """Update the label showing the best result so far."""
        if not self.model.experiments:
            self.best_result_label.setText("N/A")
            return
            
        # Get best experiment by composite score
        best_score = 0.0
        for exp in self.model.experiments:
            if 'score' in exp and exp['score'] > best_score:
                best_score = exp['score']
                
        # Convert to percentage
        best_score *= 100.0
        
        self.best_result_label.setText(f"{best_score:.2f}%")
    
    def estimate_convergence(self):
        """Estimate number of rounds to convergence based on current progress."""
        if len(self.model.experiments) < 3:
            self.est_rounds_label.setText("N/A")
            return
            
        try:
            # Extract scores
            scores = [exp.get('score', 0.0) * 100.0 for exp in self.model.experiments]
            best_scores = np.maximum.accumulate(scores)
            
            # Fit simple convergence model
            from scipy.optimize import curve_fit
            
            def convergence_model(x, a, b, c):
                return a * (1 - np.exp(-b * x)) + c
                
            x = np.arange(1, len(best_scores) + 1)
            
            try:
                p0 = [max(100.0 - min(best_scores), 1.0), 0.1, min(best_scores[0], best_scores[-1])]
                popt, _ = curve_fit(convergence_model, x, best_scores, p0=p0, maxfev=10000)
                
                a, b, c = popt
                
                asymptotic_value = min(100.0, a + c)
                current = best_scores[-1]
                target = 0.95 * asymptotic_value
                
                if a > 0:
                    target_x = -np.log(1 - (target - c) / a) / b if a > 0 else float('inf')
                    remaining_rounds = max(0, int(np.ceil((target_x - len(best_scores)) / 5)))
                    
                    if remaining_rounds < 50:
                        self.est_rounds_label.setText(f"{remaining_rounds}")
                    else:
                        self.est_rounds_label.setText("Many")
                else:
                    self.est_rounds_label.setText("Unknown")
            except:
                self.est_rounds_label.setText("Unknown")
        except Exception as e:
            self.est_rounds_label.setText("Error")
            print(f"Error estimating convergence: {e}")
    
    def show_result_details(self):
        """Show details for a selected experiment result."""
        selected_items = self.all_results_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Select a result first.")
            return
            
        # Get experiment ID from selection
        row = selected_items[0].row()
        id_item = self.all_results_table.item(row, 1)
        
        if not id_item or not id_item.text().isdigit():
            QMessageBox.warning(self, "Warning", "Invalid result selection.")
            return
            
        exp_id = int(id_item.text()) - 1
        
        if exp_id < 0 or exp_id >= len(self.model.experiments):
            QMessageBox.warning(self, "Warning", "Invalid experiment ID.")
            return
            
        exp_data = self.model.experiments[exp_id]
        
        # Create detail dialog
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

    def on_param_button_clicked(self, param_name):
        """Handle click on a parameter button in the prior tab."""
        if not param_name or param_name not in self.model.parameters:
            return
            
        param = self.model.parameters[param_name]
        
        # Open the prior dialog to set/edit prior
        dialog = PriorDialog(self, self.model, param_name)
        if dialog.exec() == QDialog.Accepted and dialog.result:
            if param.param_type in ["continuous", "discrete"]:
                param.set_prior(
                    mean=dialog.result.get("mean"),
                    std=dialog.result.get("std")
                )
                self.log(f"-- Prior set for {param_name} (mean={dialog.result.get('mean')}, std={dialog.result.get('std')}) - Success")
            else:
                # Handle categorical priors (preferences)
                preferences = dialog.result.get("categorical_preferences", {})
                param.categorical_preferences = preferences
                self.log(f"-- Categorical prior set for {param_name} - Success")
                
            # Update prior table
            self.update_prior_table()
            # Update visualization if this parameter is selected
            if self.viz_param_combo.currentText() == param_name:
                self.update_prior_plot()

    def update_prior_param_buttons(self):
        """Update the parameter buttons in the prior tab."""
        # Clear existing buttons
        while self.param_buttons_layout.count():
            item = self.param_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # Add button for each parameter
        for name, param in self.model.parameters.items():
            button = QPushButton(name)
            
            # Add indicator if this parameter already has a prior
            has_prior = False
            if param.param_type in ["continuous", "discrete"] and param.prior_mean is not None:
                has_prior = True
                button.setText(f"{name} ✓")
            elif param.param_type == "categorical" and param.categorical_preferences:
                has_prior = True
                button.setText(f"{name} ✓")
                
            # Style button differently if it has a prior
            if has_prior:
                button.setStyleSheet("QPushButton { font-weight: bold; background-color: #e6f2ff; }")
                
            # Connect button to handler with parameter name
            button.clicked.connect(lambda checked, n=name: self.on_param_button_clicked(n))
            
            self.param_buttons_layout.addWidget(button)
            
        # Add help button
        help_btn = QPushButton("About Priors")
        help_btn.clicked.connect(self.show_prior_help)
        self.param_buttons_layout.addWidget(help_btn)
        
    def show_prior_help(self):
        """Show help information about priors."""
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