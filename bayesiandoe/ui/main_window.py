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
from ..registry import ChemicalRegistry

from .widgets import (
    LogDisplay, SplashScreen, ParameterTable, ExperimentTable,
    BestResultsTable, AllResultsTable
)
from .canvas import MplCanvas, Mpl3DCanvas
from .dialogs import (
    ParameterDialog, ResultDialog, TemplateSelector, PriorDialog,
    OptimizationSettingsDialog
)

from .tab_setup import (
    setup_setup_tab, setup_prior_tab, setup_experiment_tab, 
    setup_results_tab, setup_analysis_tab
)
from .ui_actions import (
    add_parameter, edit_parameter, remove_parameter,
    load_template, add_from_registry, add_to_registry, remove_from_registry,
    generate_initial_experiments, generate_next_experiments,
    add_result_for_selected, new_project, open_project, save_project,
    import_data, export_results, statistical_analysis, plan_parallel_experiments,
    open_structure_editor, show_optimization_settings, show_preferences,
    show_documentation, show_about, add_substrate_parameter
)
from .ui_visualization import (
    update_prior_plot, update_results_plot, update_model_plot,
    update_surface_plot, update_convergence_plot, update_correlation_plot
)
from .ui_utils import (
    log, update_ui_from_model, update_parameter_combos, update_prior_table,
    update_prior_ui, update_prior_param_buttons, update_best_result_label,
    update_rounding_settings
)
from .ui_callbacks import (
    update_objectives, show_registry_item_tooltip, refresh_registry,
    on_prior_selected, update_best_results, show_result_details,
    show_prior_help, on_param_button_clicked
)

class BayesianDOEApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = OptunaBayesianExperiment()
        self.current_round = 0
        self.round_start_indices = []
        self.working_directory = os.getcwd()
        
        self.registry_manager = ChemicalRegistry()
        self.registry_manager.initialize_registry()
        self.registry = self.registry_manager.get_full_registry()
        
        self.experiment_spin = QSpinBox()
        self.experiment_spin.setMinimum(1)
        self.experiment_spin.setMaximum(1000)
        self.experiment_spin.setValue(10)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Bayesian DOE - Chemical Reaction Optimizer")
        self.setMinimumSize(1200, 800)
        
        self.log_display = LogDisplay()
        
        # Add the n_initial_spin and n_next_spin attributes
        self.n_initial_spin = None
        self.n_next_spin = None
        
        self.create_menu_bar()
        self.create_status_bar()
        self.create_central_widget()
        
        self.log("-- Application started successfully")
        
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(QAction("New Project", self, triggered=lambda: new_project(self)))
        file_menu.addAction(QAction("Open Project", self, triggered=lambda: open_project(self)))
        file_menu.addAction(QAction("Save Project", self, triggered=lambda: save_project(self)))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Import Data", self, triggered=lambda: import_data(self)))
        file_menu.addAction(QAction("Export Results", self, triggered=lambda: export_results(self)))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))
        
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(QAction("Optimization Settings", self, triggered=lambda: show_optimization_settings(self)))
        edit_menu.addAction(QAction("Preferences", self, triggered=lambda: show_preferences(self)))
        
        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(QAction("Structure Editor", self, triggered=lambda: open_structure_editor(self)))
        tools_menu.addAction(QAction("Parallel Experiments", self, triggered=lambda: plan_parallel_experiments(self)))
        tools_menu.addAction(QAction("Statistical Analysis", self, triggered=lambda: statistical_analysis(self)))
        
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("Documentation", self, triggered=lambda: show_documentation(self)))
        help_menu.addAction(QAction("About", self, triggered=lambda: show_about(self)))
        
    def create_status_bar(self):
        from ..core import settings
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        rounding_section = QWidget()
        rounding_layout = QHBoxLayout(rounding_section)
        rounding_layout.setContentsMargins(0, 0, 0, 0)
        
        self.auto_round_check = QCheckBox("Auto Round")
        self.auto_round_check.setChecked(settings.auto_round)
        self.auto_round_check.setToolTip("Automatically round numerical values")
        self.auto_round_check.stateChanged.connect(lambda: update_rounding_settings(self))
        rounding_layout.addWidget(self.auto_round_check)
        
        rounding_layout.addWidget(QLabel("Precision:"))
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(0, 8)
        self.precision_spin.setValue(settings.rounding_precision)
        self.precision_spin.setToolTip("Number of decimal places for rounding")
        self.precision_spin.valueChanged.connect(lambda: update_rounding_settings(self))
        rounding_layout.addWidget(self.precision_spin)
        
        self.smart_round_check = QCheckBox("Smart")
        self.smart_round_check.setChecked(settings.smart_rounding)
        self.smart_round_check.setToolTip("Use smart rounding based on value magnitude")
        self.smart_round_check.stateChanged.connect(lambda: update_rounding_settings(self))
        rounding_layout.addWidget(self.smart_round_check)
        
        self.status_bar.addPermanentWidget(rounding_section)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setValue(0)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        self.tab_widget = QTabWidget()
        setup_setup_tab(self)
        setup_prior_tab(self)
        setup_experiment_tab(self)
        setup_results_tab(self)
        setup_analysis_tab(self)
        
        # Add a tab change handler to update the UI when switching tabs
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tab_widget)
        
        self.setCentralWidget(central_widget)
        
    def log(self, message):
        log(self, message)

    # Add this new method to handle tab changes
    def on_tab_changed(self, index):
        # Prior Knowledge tab is typically index 1
        if index == 1:  # Prior Knowledge tab
            update_prior_param_buttons(self)
            update_prior_table(self)
            update_parameter_combos(self)
            if hasattr(self, 'viz_param_combo') and self.viz_param_combo.count() > 0:
                update_prior_plot(self)
        # Results tab is typically index 3    
        elif index == 3:  # Results tab
            if self.model.experiments:
                update_results_plot(self)
                update_best_results(self)
        # Analysis tab is typically index 4
        elif index == 4:  # Analysis tab
            if self.model.experiments:
                if hasattr(self, 'convergence_canvas'):
                    update_convergence_plot(self)