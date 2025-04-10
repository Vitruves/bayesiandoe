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
    self.registry_tabs.setMinimumHeight(400)  # Set a minimum height for the registry tabs
    
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
            list_widget.setMinimumHeight(300)  # Make each list widget taller
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
    
    # Replace with an enhanced objective input panel
    obj_input_panel = QHBoxLayout()
    
    # Add preset objectives dropdown with styling
    self.objective_input = QComboBox()
    self.objective_input.setEditable(True)
    self.objective_input.setPlaceholderText("Enter objective...")
    self.objective_input.addItems(["yield", "selectivity", "purity", "conversion", "cost", "time", "ee", "dr", "sustainability"])
    self.objective_input.setMinimumWidth(150)
    self.objective_input.setStyleSheet("""
        QComboBox {
            background-color: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QComboBox:focus {
            border: 1px solid #4dabf7;
        }
    """)
    obj_input_panel.addWidget(self.objective_input, 1)
    
    # Weight input with better styling
    weight_layout = QHBoxLayout()
    weight_layout.addWidget(QLabel("Weight:"))
    self.objective_weight = QDoubleSpinBox()
    self.objective_weight.setRange(0.1, 10.0)
    self.objective_weight.setValue(1.0)
    self.objective_weight.setSingleStep(0.1)
    self.objective_weight.setDecimals(1) 
    self.objective_weight.setStyleSheet("""
        QDoubleSpinBox {
            background-color: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 4px;
        }
        QDoubleSpinBox:focus {
            border: 1px solid #4dabf7;
        }
    """)
    weight_layout.addWidget(self.objective_weight)
    obj_input_panel.addLayout(weight_layout)
    
    # Add button with improved styling
    add_obj_btn = QPushButton("Add")
    add_obj_btn.setStyleSheet("""
        QPushButton {
            background-color: #4dabf7;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #339af0;
        }
        QPushButton:pressed {
            background-color: #228be6;
        }
    """)
    add_obj_btn.clicked.connect(lambda: self.add_objective())
    obj_input_panel.addWidget(add_obj_btn)
    
    objectives_layout.addLayout(obj_input_panel)
    
    # List of currently defined objectives with improved styling
    objectives_list_layout = QVBoxLayout()
    objectives_list_label = QLabel("Current Objectives:")
    objectives_list_label.setStyleSheet("font-weight: bold;")
    objectives_list_layout.addWidget(objectives_list_label)
    
    self.objectives_list = QListWidget()
    self.objectives_list.setMaximumHeight(80)
    self.objectives_list.setAlternatingRowColors(True)
    self.objectives_list.setStyleSheet("""
        QListWidget {
            background-color: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
        }
        QListWidget::item {
            padding: 4px;
        }
        QListWidget::item:selected {
            background-color: #4dabf7;
            color: white;
        }
        QListWidget::item:alternate {
            background-color: #e9ecef;
        }
    """)
    
    # Add initial objective
    self.objectives_list.addItem("yield [1.0]")
    objectives_list_layout.addWidget(self.objectives_list)
    
    # Action buttons
    action_buttons = QHBoxLayout()
    
    remove_obj_btn = QPushButton("Remove Selected")
    remove_obj_btn.setStyleSheet("""
        QPushButton {
            background-color: #f03e3e;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #e03131;
        }
        QPushButton:pressed {
            background-color: #c92a2a;
        }
    """)
    remove_obj_btn.clicked.connect(lambda: self.remove_objective())
    
    apply_obj_btn = QPushButton("Apply Objectives")
    apply_obj_btn.setStyleSheet("""
        QPushButton {
            background-color: #40c057;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #37b24d;
        }
        QPushButton:pressed {
            background-color: #2f9e44;
        }
    """)
    apply_obj_btn.clicked.connect(lambda: update_objectives(self))
    
    # Quick add common combinations
    quick_add_btn = QPushButton("Add Common Sets")
    quick_add_btn.setStyleSheet("""
        QPushButton {
            background-color: #7950f2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #6741d9;
        }
        QPushButton:pressed {
            background-color: #5f3dc4;
        }
    """)
    quick_add_btn.clicked.connect(lambda: self.show_quick_add_objectives())
    
    action_buttons.addWidget(remove_obj_btn)
    action_buttons.addWidget(quick_add_btn)
    action_buttons.addWidget(apply_obj_btn)
    
    objectives_list_layout.addLayout(action_buttons)
    objectives_layout.addLayout(objectives_list_layout)
    
    right_panel.addWidget(objectives_group)
    
    # Make registry taller by adjusting relative layout proportions
    layout.addLayout(left_panel, 65)
    layout.addLayout(right_panel, 35)
    
    # Add objective manipulation methods to the main app class
    def add_objective(self):
        obj_name = self.objective_input.currentText().strip().lower()
        weight = self.objective_weight.value()
        
        if obj_name:
            # Check if objective already exists
            for i in range(self.objectives_list.count()):
                text = self.objectives_list.item(i).text()
                if text.startswith(obj_name + " "):
                    return  # Already exists
                    
            # Add to list
            item = QListWidgetItem(f"{obj_name} [{weight:.1f}]")
            self.objectives_list.addItem(item)
            self.objectives_list.scrollToItem(item)
            
            # Clear input field for next entry
            self.objective_input.setCurrentText("")
            self.objective_input.setFocus()
    
    def remove_objective(self):
        selected_items = self.objectives_list.selectedItems()
        if selected_items and self.objectives_list.count() > 1:  # Keep at least one objective
            for item in selected_items:
                self.objectives_list.takeItem(self.objectives_list.row(item))
    
    def show_quick_add_objectives(self):
        """Show popup menu with common objective combinations"""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #4dabf7;
                color: white;
            }
        """)
        
        # Add common objective combinations
        action1 = menu.addAction("Yield Only")
        action2 = menu.addAction("Yield + Selectivity")
        action3 = menu.addAction("Yield + Purity")
        action4 = menu.addAction("Yield + ee (Enantioselectivity)")
        action5 = menu.addAction("Yield + Time + Cost")
        
        # Connect actions to handlers
        action1.triggered.connect(lambda: self.apply_objective_preset(["yield"]))
        action2.triggered.connect(lambda: self.apply_objective_preset(["yield", "selectivity"]))
        action3.triggered.connect(lambda: self.apply_objective_preset(["yield", "purity"]))
        action4.triggered.connect(lambda: self.apply_objective_preset(["yield", "ee"]))
        action5.triggered.connect(lambda: self.apply_objective_preset(["yield", "time", "cost"]))
        
        # Show the menu
        menu.exec(self.mapToGlobal(QPoint(
            self.objectives_list.x() + self.objectives_list.width()//2, 
            self.objectives_list.y() + self.objectives_list.height()//2
        )))
    
    def apply_objective_preset(self, objectives):
        """Apply a preset of objectives with default weights"""
        # Clear existing objectives
        self.objectives_list.clear()
        
        # Set weights based on number of objectives (simple normalization)
        weight = 1.0
        if len(objectives) > 1:
            # Primary objective has more weight
            weights = [2.0] + [1.0] * (len(objectives) - 1)
            # Normalize to sum to number of objectives
            total = sum(weights)
            weights = [w * len(objectives) / total for w in weights]
        else:
            weights = [1.0]
            
        # Add objectives with weights
        for obj, weight in zip(objectives, weights):
            self.objectives_list.addItem(f"{obj} [{weight:.1f}]")
            
        # Auto-apply objectives
        update_objectives(self)
    
    # Keyboard shortcut for adding objectives
    def handle_objective_return_key(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.add_objective()
            return True
        return False
        
    # Add key event filter to input field
    self.objective_input.keyPressEvent = lambda event: (
        handle_objective_return_key(self, event) or 
        type(self.objective_input).keyPressEvent(self.objective_input, event)
    )
    
    # Attach methods to the instance
    self.add_objective = add_objective.__get__(self)
    self.remove_objective = remove_objective.__get__(self)
    self.show_quick_add_objectives = show_quick_add_objectives.__get__(self)
    self.apply_objective_preset = apply_objective_preset.__get__(self)
    
    # Override the objectives table access for update_objectives
    def get_objectives_from_list(self):
        objectives = {}
        for i in range(self.objectives_list.count()):
            text = self.objectives_list.item(i).text()
            parts = text.split("[")
            if len(parts) == 2:
                obj_name = parts[0].strip()
                weight_str = parts[1].replace("]", "").strip()
                try:
                    weight = float(weight_str)
                    objectives[obj_name] = weight
                except ValueError:
                    pass
        return objectives
    
    # Attach to the instance
    self.get_objectives_from_list = get_objectives_from_list.__get__(self)
    
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
    
    # Add help button for algorithm selection
    help_btn = QPushButton("?")
    help_btn.setMaximumWidth(30)
    help_btn.setToolTip("Get help choosing the best algorithm")
    help_btn.clicked.connect(lambda: self.on_menu_action("algorithm_help"))
    initial_layout.addWidget(help_btn)
    
    generate_btn = QPushButton("Generate Initial")
    generate_btn.clicked.connect(lambda: generate_initial_experiments(self))
    initial_layout.addWidget(generate_btn)
    
    control_panel.addWidget(initial_group)
    
    # Next round group
    next_group = QGroupBox("Next Round")
    next_layout = QVBoxLayout(next_group)
    
    # Add experiment count in a horizontal layout
    exp_count_layout = QHBoxLayout()
    exp_count_layout.addWidget(QLabel("Number of Experiments:"))
    self.n_next_spin = QSpinBox()
    self.n_next_spin.setRange(1, 50)
    self.n_next_spin.setValue(5)
    exp_count_layout.addWidget(self.n_next_spin)
    next_layout.addLayout(exp_count_layout)
    
    # Add exploration/exploitation slider in its own row for larger size
    slider_layout = QVBoxLayout()
    slider_label_layout = QHBoxLayout()
    slider_label_layout.addWidget(QLabel("Exploration"))
    slider_label_layout.addStretch(1)
    slider_label_layout.addWidget(QLabel("Exploitation"))
    slider_layout.addLayout(slider_label_layout)
    
    self.exploit_slider = QSlider(Qt.Horizontal)
    self.exploit_slider.setRange(0, 100)
    self.exploit_slider.setValue(70)
    self.exploit_slider.setToolTip("0 = pure exploration (diverse experiments), 100 = pure exploitation (focus on best areas)")
    self.exploit_slider.setMinimumHeight(30)  # Make the slider taller
    self.exploit_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            height: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                        stop:0 #5DADE2, stop:1 #27AE60);
            margin: 0px;
        }
        QSlider::handle:horizontal {
            background: #2C3E50;
            border: 1px solid #1B2631;
            width: 18px;
            margin: -5px 0px;
            border-radius: 9px;
        }
    """)
    slider_layout.addWidget(self.exploit_slider)
    
    # Add labels below the slider to explain the meaning
    explanation_layout = QHBoxLayout()
    exploration_label = QLabel("Search new areas")
    exploration_label.setStyleSheet("color: #5DADE2;")
    explanation_layout.addWidget(exploration_label)
    explanation_layout.addStretch(1)
    exploitation_label = QLabel("Refine best results")
    exploitation_label.setStyleSheet("color: #27AE60;")
    explanation_layout.addWidget(exploitation_label)
    slider_layout.addLayout(explanation_layout)
    
    next_layout.addLayout(slider_layout)
    
    # Add the generate button
    generate_layout = QHBoxLayout()
    generate_next_btn = QPushButton("Generate Next")
    generate_next_btn.clicked.connect(lambda: generate_next_experiments(self))
    generate_layout.addStretch(1)
    generate_layout.addWidget(generate_next_btn)
    next_layout.addLayout(generate_layout)
    
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