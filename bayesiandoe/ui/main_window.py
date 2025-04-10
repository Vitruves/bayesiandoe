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
    show_prior_help, on_param_button_clicked, show_design_method_help
)

# Dictionary to store lazily loaded modules
_lazy_modules = {}

def lazy_import(module_name):
    """Lazily import modules only when needed"""
    if module_name not in _lazy_modules:
        _lazy_modules[module_name] = __import__(module_name, fromlist=[''])
    return _lazy_modules[module_name]

class BayesianDOEApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = OptunaBayesianExperiment()
        self.current_round = 1
        self.round_start_indices = []
        self.working_directory = os.getcwd()
        
        self.registry_manager = ChemicalRegistry()
        self.registry_manager.initialize_registry()
        self.registry = self.registry_manager.get_full_registry()
        
        self.experiment_spin = QSpinBox()
        self.experiment_spin.setMinimum(1)
        self.experiment_spin.setMaximum(1000)
        self.experiment_spin.setValue(10)
        
        # Initialize core components first
        self.log_display = LogDisplay()
        
        # Create UI skeleton
        self.setWindowTitle("Bayesian DOE - Chemical Reaction Optimizer")
        self.setMinimumSize(1200, 800)
        
        # Add the n_initial_spin and n_next_spin attributes
        self.n_initial_spin = None
        self.n_next_spin = None
        
        # Initialize suggestion time tracking
        self.model._suggestion_start_time = 0
        
        # Initialize planned_experiments to avoid None issues
        if not hasattr(self.model, 'planned_experiments'):
            self.model.planned_experiments = []
        
        # Create basic UI
        self.create_menu_bar()
        self.create_status_bar()
        
        # Initialize central widget but defer tab creation
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        self.setCentralWidget(central_widget)
        
        # Schedule detailed UI initialization for after showing the main window
        QTimer.singleShot(100, self.delayed_init_ui)
        
        self.log(" Application initialized")
    
    def delayed_init_ui(self):
        """Initialize UI components in a delayed manner"""
        # Initialize first tab immediately (most important for user)
        from .tab_setup import setup_setup_tab
        setup_setup_tab(self)
        
        # Schedule remaining tabs with delays between them
        QTimer.singleShot(200, lambda: self.init_tab_2())
        QTimer.singleShot(300, lambda: self.init_tab_3())
        QTimer.singleShot(400, lambda: self.init_tab_4())
        QTimer.singleShot(500, lambda: self.init_tab_5())
        
        # Add tab change handler to update the UI when switching tabs
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Initialize setup validation state
        self.setup_validated = False
        self.disable_tabs()
        self.on_tab_changed(0) # Ensure button state is correct initially
        
        self.log(" Application interface ready")
    
    def init_tab_2(self):
        """Initialize the second tab (Prior tab)"""
        from .tab_setup import setup_prior_tab
        setup_prior_tab(self)
    
    def init_tab_3(self):
        """Initialize the third tab (Experiment tab)"""
        from .tab_setup import setup_experiment_tab
        setup_experiment_tab(self)
    
    def init_tab_4(self):
        """Initialize the fourth tab (Results tab)"""
        from .tab_setup import setup_results_tab
        setup_results_tab(self)
    
    def init_tab_5(self):
        """Initialize the fifth tab (Analysis tab)"""
        from .tab_setup import setup_analysis_tab
        setup_analysis_tab(self)
        
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(QAction("New Project", self, triggered=lambda: self.on_menu_action("new_project")))
        file_menu.addAction(QAction("Open Project", self, triggered=lambda: self.on_menu_action("open_project")))
        file_menu.addAction(QAction("Save Project", self, triggered=lambda: self.on_menu_action("save_project")))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Import Data", self, triggered=lambda: self.on_menu_action("import_data")))
        file_menu.addAction(QAction("Export Results", self, triggered=lambda: self.on_menu_action("export_results")))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))
        
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(QAction("Optimization Settings", self, triggered=lambda: self.on_menu_action("optimization_settings")))
        edit_menu.addAction(QAction("Preferences", self, triggered=lambda: self.on_menu_action("preferences")))
        
        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(QAction("Structure Editor", self, triggered=lambda: self.on_menu_action("structure_editor")))
        tools_menu.addAction(QAction("Parallel Experiments", self, triggered=lambda: self.on_menu_action("parallel_experiments")))
        tools_menu.addAction(QAction("Statistical Analysis", self, triggered=lambda: self.on_menu_action("statistical_analysis")))
        
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("Documentation", self, triggered=lambda: self.on_menu_action("documentation")))
        help_menu.addAction(QAction("About", self, triggered=lambda: self.on_menu_action("about")))
    
    def on_menu_action(self, action):
        """Handle menu actions with lazy loading of required modules"""
        from .ui_actions import (
            new_project, open_project, save_project, import_data, export_results,
            statistical_analysis, plan_parallel_experiments, open_structure_editor,
            show_optimization_settings, show_preferences, show_documentation, show_about
        )
        from .ui_callbacks import show_design_method_help
        
        action_map = {
            "new_project": lambda: new_project(self),
            "open_project": lambda: open_project(self),
            "save_project": lambda: save_project(self),
            "import_data": lambda: import_data(self),
            "export_results": lambda: export_results(self),
            "optimization_settings": lambda: show_optimization_settings(self),
            "preferences": lambda: show_preferences(self),
            "structure_editor": lambda: open_structure_editor(self),
            "parallel_experiments": lambda: plan_parallel_experiments(self),
            "statistical_analysis": lambda: statistical_analysis(self),
            "documentation": lambda: show_documentation(self),
            "about": lambda: show_about(self),
            "algorithm_help": lambda: show_design_method_help(self)
        }
        
        if action in action_map:
            action_map[action]()
        
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
        self.auto_round_check.stateChanged.connect(lambda: self.update_rounding_settings())
        rounding_layout.addWidget(self.auto_round_check)
        
        rounding_layout.addWidget(QLabel("Precision:"))
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(0, 8)
        self.precision_spin.setValue(settings.rounding_precision)
        self.precision_spin.setToolTip("Number of decimal places for rounding")
        self.precision_spin.valueChanged.connect(lambda: self.update_rounding_settings())
        rounding_layout.addWidget(self.precision_spin)
        
        self.smart_round_check = QCheckBox("Smart")
        self.smart_round_check.setChecked(settings.smart_rounding)
        self.smart_round_check.setToolTip("Use smart rounding based on value magnitude")
        self.smart_round_check.stateChanged.connect(lambda: self.update_rounding_settings())
        rounding_layout.addWidget(self.smart_round_check)
        
        self.status_bar.addPermanentWidget(rounding_section)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setValue(0)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def update_rounding_settings(self):
        """Update rounding settings with lazy import"""
        from .ui_utils import update_rounding_settings
        update_rounding_settings(self)
        
    def log(self, message):
        """Log a message to the application log"""
        if hasattr(self, 'log_display'):
            self.log_display.log(message)
        print(message)

    def disable_tabs(self):
        """Disable tabs that require validated setup"""
        # Only enable setup tab
        for i in range(1, self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, False)

    def validate_setup(self):
        """Validate that experiment setup is complete before allowing access to other tabs"""
        # Check if parameters are defined
        if not self.model.parameters:
            QMessageBox.warning(self, "Validation Failed", 
                              "Please define at least one parameter before continuing.")
            return False
        
        # Check if objectives are defined
        if not self.model.objectives:
            QMessageBox.warning(self, "Validation Failed", 
                              "Please define at least one objective before continuing.")
            return False
        
        # All validation passed
        self.setup_validated = True
        
        # Enable all tabs
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, True)
        
        # Find the validate button and update its state
        validate_button = self.findChild(QPushButton, "ValidateButton")
        if validate_button:
            validate_button.setText("✓ Setup Validated")
            validate_button.setStyleSheet("""
                QPushButton#ValidateButton {
                    background-color: #2ecc71;
                    color: white;
                    border-radius: 5px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 14px;
                    margin-top: 10px;
                }
                QPushButton#ValidateButton:hover {
                    background-color: #27ae60;
                }
                QPushButton#ValidateButton:disabled {
                    background-color: #2ecc71; /* Keep green when disabled after validation */
                    color: white;
                }
            """)
            validate_button.setEnabled(False)
            validate_button.setVisible(False) # Hide after validation
        
        # Show success message
        self.log(" Experiment setup validated - Success")
        
        # Navigate to Prior Knowledge tab
        self.tab_widget.setCurrentIndex(1)
        
        return True

    def on_tab_changed(self, index):
        """Load tab contents when user switches to that tab, with auto-refresh"""
        # Show/hide validate button based on current tab
        validate_button = self.findChild(QPushButton, "ValidateButton")
        if validate_button:
            # Only show validation button on setup tab (index 0) if not validated
            if index == 0 and not self.setup_validated:
                validate_button.setVisible(True)
            else:
                validate_button.setVisible(False)
                
        # Prior Knowledge tab is typically index 1
        if index == 1:  # Prior Knowledge tab
            self.update_prior_tab()
        # Results tab is typically index 3    
        elif index == 3:  # Results tab
            self.update_results_tab(force_refresh=True)
        # Analysis tab is typically index 4
        elif index == 4:  # Analysis tab
            self.update_analysis_tab(force_refresh=True)
            
    def update_prior_tab(self):
        """Update prior tab contents with lazy imports"""
        # Import only when needed
        from .ui_utils import update_prior_param_buttons, update_prior_table, update_parameter_combos
        from .ui_visualization import update_prior_plot
        
        update_prior_param_buttons(self)
        update_prior_table(self)
        update_parameter_combos(self)
        if hasattr(self, 'viz_param_combo') and self.viz_param_combo.count() > 0:
            update_prior_plot(self)
    
    def update_results_tab(self, force_refresh=False):
        """Update results tab contents with lazy imports and auto-refresh on tab click"""
        # Import only when needed
        from .ui_visualization import update_results_plot
        from .ui_callbacks import update_best_results
        
        if self.model.experiments:
            # Always update tables
            if hasattr(self, 'best_table'):
                self.best_table.update_from_model(self.model, self.n_best_spin.value() if hasattr(self, 'n_best_spin') else 5)
                
            if hasattr(self, 'all_results_table'):
                self.all_results_table.update_from_model(self.model)
                
            # Force refresh plots when tab is clicked
            if force_refresh:
                update_results_plot(self)
                update_best_results(self)
                self.log(" Results visualizations refreshed - Success")
    
    def update_analysis_tab(self, force_refresh=False):
        """Update analysis tab contents with lazy imports and auto-refresh on tab click"""
        # Import only when needed
        from .ui_visualization import update_convergence_plot, update_model_plot, update_surface_plot
        
        if self.model.experiments:
            if force_refresh:
                if hasattr(self, 'convergence_canvas'):
                    update_convergence_plot(self)
                    
                if hasattr(self, 'model_canvas'):
                    update_model_plot(self)
                    
                # Skip surface plot automatic update as it's computationally expensive
                self.log(" Analysis visualizations refreshed - Success")

    def handle_critical_error(self, error_msg, details=None):
        """Handle critical errors with proper user feedback."""
        from PySide6.QtWidgets import QMessageBox
        
        # Log the error
        self.log(f" ERROR: {error_msg} - Failed")
        
        # Show error dialog with details
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(error_msg)
        
        if details:
            error_box.setDetailedText(details)
        
        # Add helpful action buttons
        retry_button = error_box.addButton("Retry", QMessageBox.ActionRole)
        continue_button = error_box.addButton("Continue Anyway", QMessageBox.ActionRole)
        cancel_button = error_box.addButton("Cancel", QMessageBox.RejectRole)
        
        error_box.exec_()
        
        # Handle user choice
        clicked = error_box.clickedButton()
        if clicked == retry_button:
            return "retry"
        elif clicked == continue_button:
            return "continue"
        else:
            return "cancel"

    def update_result_tables(self):
        """Update all result tables when a result is added or modified.
        This is called after direct table editing or through the add_result_for_selected method."""
        try:
            # Update the best results table
            if hasattr(self, 'best_table'):
                self.best_table.update_from_model(self.model, self.n_best_spin.value() if hasattr(self, 'n_best_spin') else 5)
                
            # Update the all results table
            if hasattr(self, 'all_results_table'):
                self.all_results_table.update_from_model(self.model)
                
            # Lazy import for updating best result label
            from .ui_utils import update_best_result_label
            update_best_result_label(self)
                
            # Update counts
            if hasattr(self, 'results_count_label'):
                completed_count = len(self.model.experiments)
                planned_count = len(self.model.planned_experiments)
                self.results_count_label.setText(f"{completed_count} / {planned_count}")
                
            # Update plots if we're on the results or analysis tab
            if hasattr(self, 'tab_widget'):
                current_tab = self.tab_widget.currentIndex()
                if current_tab == 3:  # Results tab
                    from .ui_visualization import update_results_plot
                    update_results_plot(self)
                elif current_tab == 4:  # Analysis tab
                    if hasattr(self, 'convergence_canvas'):
                        from .ui_visualization import update_convergence_plot
                        update_convergence_plot(self)
                    if hasattr(self, 'correlation_canvas'):
                        from .ui_visualization import update_correlation_plot
                        update_correlation_plot(self)
                    
            # Log success message (only if not already logged from the source)
            if not hasattr(self, '_result_updated_logged') or not self._result_updated_logged:
                self.log("-- Result tables updated - Success")
                self._result_updated_logged = True
                # Reset the flag after a short delay
                QTimer.singleShot(100, lambda: setattr(self, '_result_updated_logged', False))
                
        except Exception as e:
            import traceback
            print(f"Error updating result tables: {e}")
            print(traceback.format_exc())