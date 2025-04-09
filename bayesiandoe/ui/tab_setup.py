from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, 
    QGroupBox, QFormLayout, QGridLayout, QSpinBox, QDoubleSpinBox,
    QSlider, QScrollArea, QFrame, QHeaderView, QTableWidgetItem,
    QTabWidget, QCheckBox, QListWidget, QStackedWidget, QTableWidget,
    QListWidgetItem
)
from PySide6.QtGui import QFont, QColor

import matplotlib.pyplot as plt
import numpy as np

from .canvas import MplCanvas, Mpl3DCanvas
from .ui_callbacks import (
    update_objectives, refresh_registry, show_registry_item_tooltip,
    on_prior_selected, update_best_results, show_result_details,
    show_prior_help, on_param_button_clicked, show_design_method_help,
    show_parameter_links
)
from .ui_visualization import (
    update_prior_plot, update_results_plot, update_model_plot,
    update_surface_plot, update_convergence_plot, update_correlation_plot
)
from .ui_actions import (
    add_parameter, edit_parameter, remove_parameter,
    load_template, add_from_registry, add_to_registry, remove_from_registry,
    generate_initial_experiments, generate_next_experiments,
    add_result_for_selected, export_results
)
from .widgets import (
    ExperimentTable, BestResultsTable, AllResultsTable, ParameterTable
)

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
    
    self.add_param_button.clicked.connect(lambda: add_parameter(self))
    self.edit_param_button.clicked.connect(lambda: edit_parameter(self))
    self.remove_param_button.clicked.connect(lambda: remove_parameter(self))
    
    button_layout.addWidget(self.add_param_button)
    button_layout.addWidget(self.edit_param_button)
    button_layout.addWidget(self.remove_param_button)
    
    param_layout.addLayout(button_layout)
    
    left_panel.addWidget(param_group)
    
    templates_group = QGroupBox("Parameter Templates")
    templates_layout = QGridLayout(templates_group)
    
    # Create direct functions for each template button
    def load_reaction_conditions():
        load_template(self, "reaction_conditions")
        
    def load_catalyst():
        load_template(self, "catalyst")
        
    def load_solvent():
        load_template(self, "solvent")
        
    def load_cross_coupling():
        load_template(self, "cross_coupling")
        
    def load_oxidation():
        load_template(self, "oxidation")
        
    def load_reduction():
        load_template(self, "reduction")
        
    def load_amide_coupling():
        load_template(self, "amide_coupling")
        
    def load_organocatalysis():
        load_template(self, "organocatalysis")
    
    # Map template names to loading functions    
    templates = [
        ("Reaction Conditions", load_reaction_conditions),
        ("Catalyst Screening", load_catalyst),
        ("Solvent Screening", load_solvent),
        ("Cross-Coupling", load_cross_coupling),
        ("Oxidation", load_oxidation),
        ("Reduction", load_reduction),
        ("Amide Coupling", load_amide_coupling),
        ("Organocatalysis", load_organocatalysis)
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
            list_widget.itemEntered.connect(
                lambda item, rt=reg_type, cat=category: show_registry_item_tooltip(self, item, rt, cat)
            )
            
            cat_layout.addWidget(list_widget)
            
            category_tabs.addTab(cat_tab, category)
            category_widgets[category] = list_widget
        
        tab_layout.addWidget(category_tabs)
        
        btn_layout = QHBoxLayout()
        add_to_param_btn = QPushButton("Add to Parameters")
        add_to_reg_btn = QPushButton("Add New")
        remove_btn = QPushButton("Remove")
        
        rt = reg_type
        add_to_param_btn.clicked.connect(lambda checked, rt=rt: add_from_registry(self, rt))
        add_to_reg_btn.clicked.connect(lambda checked, rt=rt: add_to_registry(self, rt))
        remove_btn.clicked.connect(lambda checked, rt=rt: remove_from_registry(self, rt))
        
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
    
    refresh_registry(self)
    
    objectives_group = QGroupBox("Optimization Objectives")
    objectives_layout = QVBoxLayout(objectives_group)
    
    self.objectives_table = QTableWidget(3, 2)
    self.objectives_table.setHorizontalHeaderLabels(["Objective", "Weight"])
    self.objectives_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    self.objectives_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    
    self.objectives_table.setItem(0, 0, QTableWidgetItem("yield"))
    self.objectives_table.setItem(0, 1, QTableWidgetItem("1.0"))
    
    objectives_layout.addWidget(self.objectives_table)
    
    obj_buttons_layout = QHBoxLayout()
    
    apply_obj_btn = QPushButton("Apply Objectives")
    apply_obj_btn.clicked.connect(lambda: update_objectives(self))
    obj_buttons_layout.addWidget(apply_obj_btn)
    
    objectives_layout.addLayout(obj_buttons_layout)
    
    right_panel.addWidget(objectives_group)
    
    layout.addLayout(left_panel, 65)
    layout.addLayout(right_panel, 35)
    
    self.tab_widget.addTab(setup_tab, "Experiment Setup")

def setup_prior_tab(self):
    prior_tab = QWidget()
    layout = QHBoxLayout(prior_tab)
    
    left_panel = QVBoxLayout()
    
    prior_set_group = QGroupBox("Set Prior Knowledge")
    prior_set_layout = QVBoxLayout(prior_set_group)
    
    info_label = QLabel("Click on a parameter to set prior knowledge:")
    info_label.setWordWrap(True)
    prior_set_layout.addWidget(info_label)
    
    param_scroll = QScrollArea()
    param_scroll.setWidgetResizable(True)
    param_scroll.setMinimumHeight(300)
    
    param_content = QWidget()
    self.param_buttons_layout = QVBoxLayout(param_content)
    
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
    
    self.prior_table.itemSelectionChanged.connect(lambda: on_prior_selected(self))
    
    prior_table_layout.addWidget(self.prior_table)
    
    left_panel.addWidget(prior_table_group)
    
    # Parameter linking section
    link_group = QGroupBox("Parameter Linking")
    link_layout = QVBoxLayout(link_group)
    
    link_info = QLabel(
        "Parameter linking allows the model to learn relationships between parameters. "
        "When the model discovers that certain parameter combinations work well together, "
        "it will suggest similar combinations in future experiments."
    )
    link_info.setWordWrap(True)
    link_layout.addWidget(link_info)
    
    link_param_layout = QHBoxLayout()
    link_param_layout.addWidget(QLabel("Parameter:"))
    
    self.link_param_combo = QComboBox()
    link_param_layout.addWidget(self.link_param_combo, 1)
    
    self.show_links_btn = QPushButton("Show/Edit Links")
    self.show_links_btn.clicked.connect(lambda: show_parameter_links(self, self.link_param_combo.currentText()))
    link_param_layout.addWidget(self.show_links_btn)
    
    link_layout.addLayout(link_param_layout)
    
    left_panel.addWidget(link_group)
    
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
    self.update_viz_btn.clicked.connect(lambda: update_prior_plot(self))
    viz_control_layout.addWidget(self.update_viz_btn)
    
    viz_layout.addLayout(viz_control_layout)
    
    right_panel.addWidget(viz_group)
    
    layout.addLayout(left_panel, 1)
    layout.addLayout(right_panel, 1)
    
    self.tab_widget.addTab(prior_tab, "Prior Knowledge")

def setup_experiment_tab(self):
    """Set up the experiment tab interface."""
    experiment_tab = QWidget()
    self.tab_widget.addTab(experiment_tab, "Experiments")
    
    layout = QVBoxLayout(experiment_tab)
    
    # Top control panel
    control_panel = QHBoxLayout()
    
    # Initial experiments group
    initial_group = QGroupBox("Initial Experiments")
    initial_layout = QHBoxLayout(initial_group)
    
    initial_layout.addWidget(QLabel("Number of Experiments:"))
    self.n_initial_spin = QSpinBox()
    self.n_initial_spin.setRange(1, 100)
    self.n_initial_spin.setValue(10)
    initial_layout.addWidget(self.n_initial_spin)
    
    self.design_method_combo = QComboBox()
    self.design_method_combo.addItems(["Random", "Latin Hypercube", "Sobol", "BoTorch", "TPE", "GPEI"])
    self.design_method_combo.setCurrentText("Latin Hypercube")
    initial_layout.addWidget(self.design_method_combo)
    
    generate_btn = QPushButton("Generate Initial")
    generate_btn.clicked.connect(lambda: generate_initial_experiments(self))
    initial_layout.addWidget(generate_btn)
    
    control_panel.addWidget(initial_group)
    
    # Next round group
    next_group = QGroupBox("Next Round")
    next_layout = QHBoxLayout(next_group)
    
    next_layout.addWidget(QLabel("Number of Experiments:"))
    self.n_next_spin = QSpinBox()
    self.n_next_spin.setRange(1, 50)
    self.n_next_spin.setValue(5)
    next_layout.addWidget(self.n_next_spin)
    
    next_layout.addWidget(QLabel("Exploration:"))
    self.exploit_slider = QSlider(Qt.Horizontal)
    self.exploit_slider.setRange(0, 100)
    self.exploit_slider.setValue(70)
    self.exploit_slider.setToolTip("0 = pure exploration, 100 = pure exploitation")
    next_layout.addWidget(self.exploit_slider, 1)
    
    generate_next_btn = QPushButton("Generate Next")
    generate_next_btn.clicked.connect(lambda: generate_next_experiments(self))
    next_layout.addWidget(generate_next_btn)
    
    control_panel.addWidget(next_group)
    
    # Round indicator
    round_group = QGroupBox("Current Round")
    round_layout = QHBoxLayout(round_group)
    
    round_layout.addWidget(QLabel("Round:"))
    self.current_round_label = QLabel("1")
    self.current_round = 1
    
    # Make the round label stand out with larger font
    font = self.current_round_label.font()
    font.setPointSize(font.pointSize() + 4)
    font.setBold(True)
    self.current_round_label.setFont(font)
    
    round_layout.addWidget(self.current_round_label)
    control_panel.addWidget(round_group)
    
    layout.addLayout(control_panel)
    
    # Instructions for direct table editing
    instructions_box = QGroupBox("Result Entry")
    instructions_layout = QVBoxLayout(instructions_box)
    
    instructions_label = QLabel(
        "<b>Two ways to enter results:</b><br>"
        "1. <b>Direct editing:</b> Double-click on any result cell (columns with *) to enter values directly<br>"
        "2. <b>Dialog:</b> Select an experiment row and click 'Add Result' for a detailed entry form"
    )
    instructions_label.setWordWrap(True)
    instructions_layout.addWidget(instructions_label)
    
    # Quick tip
    tip_label = QLabel(
        "<i>Tip: You can enter values as 85.2 or 85.2% - both work!</i>"
    )
    tip_label.setStyleSheet("color: #0066cc;")
    instructions_layout.addWidget(tip_label)
    
    layout.addWidget(instructions_box)
    
    # Experiment table
    experiment_group = QGroupBox("Planned Experiments")
    experiment_layout = QVBoxLayout(experiment_group)
    
    self.experiment_table = ExperimentTable()
    experiment_layout.addWidget(self.experiment_table)
    
    button_layout = QHBoxLayout()
    
    add_result_btn = QPushButton("Add Result")
    add_result_btn.clicked.connect(lambda: add_result_for_selected(self))
    button_layout.addWidget(add_result_btn)
    
    experiment_layout.addLayout(button_layout)
    
    layout.addWidget(experiment_group, 1)  # Give it more vertical space
    
    # Update UI
    self.experiment_table.update_columns(self.model)

def setup_results_tab(self):
    results_tab = QWidget()
    layout = QVBoxLayout(results_tab)
    
    tabs = QTabWidget()
    
    viz_tab = QWidget()
    viz_layout = QVBoxLayout(viz_tab)
    
    control_layout = QHBoxLayout()
    
    control_layout.addWidget(QLabel("Plot Type:"))
    self.plot_type_combo = QComboBox()
    self.plot_type_combo.addItems([
        "Optimization History", 
        "Parameter Importance", 
        "Parameter Contour", 
        "Objective Correlation",
        "Parameter Links"
    ])
    control_layout.addWidget(self.plot_type_combo)
    
    control_layout.addWidget(QLabel("X-axis:"))
    self.x_param_combo = QComboBox()
    control_layout.addWidget(self.x_param_combo)
    
    control_layout.addWidget(QLabel("Y-axis:"))
    self.y_param_combo = QComboBox()
    control_layout.addWidget(self.y_param_combo)
    
    self.update_plot_btn = QPushButton("Update Plot")
    self.update_plot_btn.clicked.connect(lambda: update_results_plot(self))
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
    self.update_best_btn.clicked.connect(lambda: update_best_results(self))
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
    self.export_results_btn.clicked.connect(lambda: export_results(self))
    all_control.addWidget(self.export_results_btn)
    
    self.show_details_btn = QPushButton("Show Details")
    self.show_details_btn.clicked.connect(lambda: show_result_details(self))
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
    self.update_corr_btn.clicked.connect(lambda: update_correlation_plot(self))
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
    self.update_model_btn.clicked.connect(lambda: update_model_plot(self))
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
    self.update_surface_btn.clicked.connect(lambda: update_surface_plot(self))
    surface_control.addWidget(self.update_surface_btn)
    
    surface_layout.addLayout(surface_control)
    
    self.surface_canvas = Mpl3DCanvas(self, width=8, height=6)
    surface_layout.addWidget(self.surface_canvas)
    
    tabs.addTab(surface_tab, "Response Surface")
    
    convergence_tab = QWidget()
    convergence_layout = QVBoxLayout(convergence_tab)
    
    self.update_convergence_btn = QPushButton("Update Convergence Plot")
    self.update_convergence_btn.clicked.connect(lambda: update_convergence_plot(self))
    convergence_layout.addWidget(self.update_convergence_btn)
    
    self.convergence_canvas = MplCanvas(self, width=8, height=6)
    convergence_layout.addWidget(self.convergence_canvas)
    
    tabs.addTab(convergence_tab, "Convergence Analysis")
    
    layout.addWidget(tabs)
    
    self.tab_widget.addTab(analysis_tab, "Analysis") 