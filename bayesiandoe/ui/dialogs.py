from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QRadioButton, QGroupBox, QComboBox, QCheckBox, QTextEdit,
    QMessageBox, QSlider, QFrame, QTableWidget, QTableWidgetItem,
    QScrollArea, QListWidget, QTabWidget, QWidget, QHeaderView
)
from ..visualizations import plot_parameter_importance
from .canvas import MplCanvas
from ..core import settings
import datetime

class ParameterDialog(QDialog):
    def __init__(self, parent=None, param=None, edit_mode=False):
        super().__init__(parent)
        self.param = param
        self.edit_mode = edit_mode
        self.result = None
        
        if edit_mode and param:
            self.setWindowTitle(f"Edit Parameter: {param.name}")
        else:
            self.setWindowTitle("Add Parameter")
            
        self.resize(500, 300)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        if self.edit_mode and self.param:
            self.name_edit.setText(self.param.name)
            self.name_edit.setEnabled(False)
        form_layout.addRow("Parameter Name:", self.name_edit)
        
        type_group = QGroupBox()
        type_group.setTitle("Parameter Type")
        type_layout = QHBoxLayout(type_group)
        
        self.continuous_radio = QRadioButton("Continuous")
        self.discrete_radio = QRadioButton("Discrete")
        self.categorical_radio = QRadioButton("Categorical")
        
        if self.edit_mode and self.param:
            if self.param.param_type == "continuous":
                self.continuous_radio.setChecked(True)
            elif self.param.param_type == "discrete":
                self.discrete_radio.setChecked(True)
            elif self.param.param_type == "categorical":
                self.categorical_radio.setChecked(True)
            
            self.continuous_radio.setEnabled(False)
            self.discrete_radio.setEnabled(False)
            self.categorical_radio.setEnabled(False)
        else:
            self.continuous_radio.setChecked(True)
        
        type_layout.addWidget(self.continuous_radio)
        type_layout.addWidget(self.discrete_radio)
        type_layout.addWidget(self.categorical_radio)
        
        form_layout.addRow(type_group)
        
        range_group = QGroupBox()
        range_group.setTitle("Value Range")
        range_group.setObjectName("ValueRange")
        range_layout = QFormLayout(range_group)
        
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setDecimals(settings.rounding_precision)
        
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setDecimals(settings.rounding_precision)
        self.max_spin.setValue(100.0)
        
        if self.edit_mode and self.param and self.param.param_type in ["continuous", "discrete"]:
            self.min_spin.setValue(self.param.low)
            self.max_spin.setValue(self.param.high)
        
        range_layout.addRow("Min Value:", self.min_spin)
        range_layout.addRow("Max Value:", self.max_spin)
        
        categorical_group = QGroupBox()
        categorical_group.setTitle("Categorical Values")
        categorical_group.setObjectName("CategoricalValues")
        categorical_layout = QVBoxLayout(categorical_group)
        
        categorical_label = QLabel("Enter values separated by commas:")
        self.categorical_edit = QLineEdit()
        
        if self.edit_mode and self.param and self.param.param_type == "categorical":
            self.categorical_edit.setText(", ".join(self.param.choices or []))
        
        categorical_layout.addWidget(categorical_label)
        categorical_layout.addWidget(self.categorical_edit)
        
        form_layout.addRow(range_group)
        form_layout.addRow(categorical_group)
        
        if not self.edit_mode:
            self.continuous_radio.toggled.connect(self.update_ui)
            self.discrete_radio.toggled.connect(self.update_ui)
            self.categorical_radio.toggled.connect(self.update_ui)
        
        self.units_edit = QLineEdit()
        if self.edit_mode and self.param:
            self.units_edit.setText(self.param.units or "")
        form_layout.addRow("Units (optional):", self.units_edit)
        
        layout.addLayout(form_layout)
        
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        
        layout.addLayout(button_box)
        
        self.update_ui()
        
    def update_ui(self):
        is_categorical = self.categorical_radio.isChecked()
        
        range_group = self.findChild(QGroupBox, "ValueRange")
        categorical_group = self.findChild(QGroupBox, "CategoricalValues")
        
        if range_group and categorical_group:
            if is_categorical:
                range_group.setVisible(False)
                categorical_group.setVisible(True)
            else:
                range_group.setVisible(True)
                categorical_group.setVisible(False)
            
    def accept(self):
        from ..parameters import ChemicalParameter
        
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Parameter name cannot be empty")
            return
            
        if self.continuous_radio.isChecked():
            param_type = "continuous"
            low = self.min_spin.value()
            high = self.max_spin.value()
            
            if low >= high:
                QMessageBox.warning(self, "Warning", "Min value must be less than max value")
                return
                
            self.result = ChemicalParameter(
                name=name,
                param_type=param_type,
                low=low,
                high=high,
                units=self.units_edit.text().strip() or None
            )
            
        elif self.discrete_radio.isChecked():
            param_type = "discrete"
            low = int(self.min_spin.value())
            high = int(self.max_spin.value())
            
            if low >= high:
                QMessageBox.warning(self, "Warning", "Min value must be less than max value")
                return
                
            self.result = ChemicalParameter(
                name=name,
                param_type=param_type,
                low=low,
                high=high,
                units=self.units_edit.text().strip() or None
            )
            
        elif self.categorical_radio.isChecked():
            param_type = "categorical"
            choices_text = self.categorical_edit.text().strip()
            
            if not choices_text:
                QMessageBox.warning(self, "Warning", "Categorical values cannot be empty")
                return
                
            choices = [c.strip() for c in choices_text.split(",")]
            
            self.result = ChemicalParameter(
                name=name,
                param_type=param_type,
                choices=choices,
                units=self.units_edit.text().strip() or None
            )
            
        super().accept()

class ResultDialog(QDialog):
    def __init__(self, parent=None, model=None, exp_id=None, params=None):
        super().__init__(parent)
        self.model = model
        self.exp_id = exp_id
        
        # Ensure params has the correct structure - make a deep copy to avoid modifying the original
        self.params = {}
        
        # Handle both new format (with 'params' dict) and old format
        if isinstance(params, dict):
            if 'params' in params:
                # New format - deep copy the params dict
                import copy
                self.params = copy.deepcopy(params['params'])
            else:
                # Old format - copy all parameter values
                for param_name in model.parameters:
                    if param_name in params:
                        self.params[param_name] = params[param_name]
        
        self.result = None
        
        self.setWindowTitle(f"Add Result for Experiment #{exp_id+1}")
        self.resize(500, 400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        param_group = QGroupBox("Experiment Parameters")
        param_layout = QGridLayout(param_group)
        
        row = 0
        col = 0
        max_cols = 2
        
        for name, value in self.params.items():
            if name in self.model.parameters:
                param = self.model.parameters[name]
                if isinstance(value, float):
                    value_str = settings.format_value(value)
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
        
        result_group = QGroupBox("Experimental Results")
        result_layout = QFormLayout(result_group)
        
        self.result_spins = {}
        
        for obj in self.model.objectives:
            spin = QDoubleSpinBox()
            # Use more reasonable range and better defaults
            spin.setRange(0.0, 100.0)  # Allow values from 0% to 100%
            spin.setSingleStep(1.0)     # Step by 1% when using arrows
            spin.setValue(0.0)          # Start with 0 value
            spin.setSuffix(" %")        # Show percentage symbol
            spin.setDecimals(2)         # Show 2 decimal places for precision
            spin.setFixedWidth(120)     # Make size consistent
            
            # Use Qt's built-in validation instead of manual validation
            spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
            spin.setAccelerated(True)
            
            # Connect to valueChanged to enable the OK button properly
            spin.valueChanged.connect(self.check_enable_button)
            
            result_layout.addRow(f"{obj.capitalize()}:", spin)
            self.result_spins[obj] = spin
        
        layout.addWidget(result_group)
        
        notes_group = QGroupBox("Notes (Optional)")
        notes_layout = QVBoxLayout(notes_group)
        
        self.notes_edit = QTextEdit()
        notes_layout.addWidget(self.notes_edit)
        
        layout.addWidget(notes_group)
        
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("Add Result")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        
        layout.addLayout(button_box)
        
        # Check initial state
        self.check_enable_button()
    
    def check_enable_button(self):
        """Enable the OK button only if at least one result is non-zero"""
        has_valid_result = False
        for obj, spin in self.result_spins.items():
            if spin.value() > 0:
                has_valid_result = True
                break
        
        self.ok_button.setEnabled(has_valid_result)
        
    def accept(self):
        try:
            results = {}
            
            for obj, spin in self.result_spins.items():
                # Get value and normalize to 0-1 range (from percentage)
                value = spin.value() / 100.0
                # Ensure it's within valid range
                value = max(0.0, min(1.0, value))
                results[obj] = value
                
            notes = self.notes_edit.toPlainText().strip()
            
            # Create a deep copy of params to avoid reference issues
            import copy
            params_copy = copy.deepcopy(self.params)
            
            if notes:
                params_copy["notes"] = notes
                
            # Create properly structured result dictionary
            self.result = {
                "params": params_copy,
                "results": results,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            super().accept()
        except Exception as e:
            import traceback
            print(f"Error accepting result: {e}")
            print(traceback.format_exc())
            
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to add result: {str(e)}\n\nMake sure all values are valid numbers."
            )

class TemplateSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Parameter Template")
        self.resize(400, 300)
        self.selected_template = None
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Select a parameter template:"))
        
        self.list_widget = QListWidget()
        self.list_widget.addItems([
            "Reaction Conditions",
            "Catalyst Screening",
            "Solvent Screening",
            "Cross-Coupling",
            "Oxidation",
            "Reduction",
            "Amide Coupling",
            "Organocatalysis"
        ])
        
        self.list_widget.setCurrentRow(0)
        layout.addWidget(self.list_widget)
        
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("Select")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        
        layout.addLayout(button_box)
        
    def accept(self):
        self.selected_template = self.list_widget.currentItem().text().lower().replace(" ", "_")
        super().accept()

class PriorDialog(QDialog):
    def __init__(self, parent=None, model=None, param_name=None):
        super().__init__(parent)
        self.model = model
        self.param_name = param_name
        self.result = None
        
        if not param_name or param_name not in model.parameters:
            self.reject()
            return
            
        self.param = model.parameters[param_name]
        
        self.setWindowTitle(f"Set Prior for {param_name}")
        self.resize(650, 550)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Prior knowledge helps guide optimization by incorporating your expertise about parameter values. "
            "For each parameter, specify your belief about its optimal value and your confidence in that belief."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        if self.param.param_type in ["continuous", "discrete"]:
            self.setup_numeric_ui(layout)
        else:
            self.setup_categorical_ui(layout)
            
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("Set Prior")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        
        layout.addLayout(button_box)
        
    def setup_numeric_ui(self, layout):
        form_layout = QFormLayout()
        
        self.mean_spin = QDoubleSpinBox()
        self.mean_spin.setRange(self.param.low, self.param.high)
        self.mean_spin.setDecimals(settings.rounding_precision)
        self.mean_spin.setValue((self.param.high + self.param.low) / 2)
        
        if self.param.prior_mean is not None:
            self.mean_spin.setValue(self.param.prior_mean)
            
        if self.param.units:
            self.mean_spin.setSuffix(f" {self.param.units}")
            
        form_layout.addRow("Expected Optimal Value:", self.mean_spin)
        
        self.confidence_combo = QComboBox()
        self.confidence_combo.addItems(["Very High", "High", "Medium", "Low", "Very Low"])
        self.confidence_combo.setCurrentText("Medium")
        self.confidence_combo.currentTextChanged.connect(self.update_std_from_confidence)
        
        form_layout.addRow("Confidence Level:", self.confidence_combo)
        
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(0.001, (self.param.high - self.param.low))
        self.std_spin.setDecimals(settings.rounding_precision)
        self.std_spin.setValue((self.param.high - self.param.low) / 4)
        
        if self.param.prior_std is not None:
            self.std_spin.setValue(self.param.prior_std)
            self.set_confidence_from_std()
            
        if self.param.units:
            self.std_spin.setSuffix(f" {self.param.units}")
            
        form_layout.addRow("Standard Deviation:", self.std_spin)
        
        layout.addLayout(form_layout)
        
        plot_group = QGroupBox("Prior Distribution Preview")
        plot_layout = QVBoxLayout(plot_group)
        
        self.canvas = MplCanvas(self, width=5, height=3)
        plot_layout.addWidget(self.canvas)
        
        self.mean_spin.valueChanged.connect(self.update_plot)
        self.std_spin.valueChanged.connect(self.update_plot)
        
        layout.addWidget(plot_group)
        
        self.update_plot()
        
    def setup_categorical_ui(self, layout):
        info_label = QLabel("Set preference weights for each category. Higher values indicate stronger preference.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.cat_sliders = {}
        
        for choice in self.param.choices:
            choice_layout = QHBoxLayout()
            choice_layout.addWidget(QLabel(choice))
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 10)
            slider.setValue(5)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            
            value_label = QLabel("5")
            slider.valueChanged.connect(lambda v, label=value_label: label.setText(str(v)))
            
            choice_layout.addWidget(slider)
            choice_layout.addWidget(value_label)
            
            layout.addLayout(choice_layout)
            self.cat_sliders[choice] = slider
            
    def update_std_from_confidence(self):
        confidence = self.confidence_combo.currentText()
        param_range = self.param.high - self.param.low
        
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
            
        if self.param.param_type == "discrete":
            std = max(1, int(std))
            
        self.std_spin.setValue(std)
        
    def set_confidence_from_std(self):
        param_range = self.param.high - self.param.low
        std_ratio = self.param.prior_std / param_range
        
        if std_ratio < 0.05:
            self.confidence_combo.setCurrentText("Very High")
        elif std_ratio < 0.1:
            self.confidence_combo.setCurrentText("High")
        elif std_ratio < 0.2:
            self.confidence_combo.setCurrentText("Medium")
        elif std_ratio < 0.3:
            self.confidence_combo.setCurrentText("Low")
        else:
            self.confidence_combo.setCurrentText("Very Low")
            
    def update_plot(self):
        import numpy as np
        from scipy import stats
        
        if self.param.param_type == "continuous":
            x = np.linspace(max(self.param.low - 2*self.std_spin.value(), self.param.low*0.8),
                          min(self.param.high + 2*self.std_spin.value(), self.param.high*1.2), 1000)
        else:
            x = np.arange(self.param.low, self.param.high + 1)
            
        pdf = stats.norm.pdf(x, loc=self.mean_spin.value(), scale=self.std_spin.value())
        
        self.canvas.axes.clear()
        self.canvas.axes.plot(x, pdf, 'b-', linewidth=2)
        self.canvas.axes.fill_between(x, pdf, color='blue', alpha=0.2)
        
        self.canvas.axes.axvline(self.mean_spin.value(), color='r', linestyle='-', alpha=0.7)
        self.canvas.axes.axvline(self.mean_spin.value() - self.std_spin.value(), color='g', linestyle='', alpha=0.7)
        self.canvas.axes.axvline(self.mean_spin.value() + self.std_spin.value(), color='g', linestyle='', alpha=0.7)
        
        self.canvas.axes.axvline(self.param.low, color='k', linestyle=':', alpha=0.5)
        self.canvas.axes.axvline(self.param.high, color='k', linestyle=':', alpha=0.5)
        
        self.canvas.axes.set_xlabel(f"{self.param_name} {f'({self.param.units})' if self.param.units else ''}")
        self.canvas.axes.set_ylabel('Probability Density')
        self.canvas.draw()
        
    def accept(self):
        if self.param.param_type in ["continuous", "discrete"]:
            mean = self.mean_spin.value()
            std = self.std_spin.value()
            
            if mean < self.param.low or mean > self.param.high:
                QMessageBox.warning(self, "Warning", 
                    f"Mean value {mean} is outside parameter range [{self.param.low}, {self.param.high}]")
                
            self.result = {
                "mean": mean,
                "std": std
            }
        else:
            preferences = {}
            for choice, slider in self.cat_sliders.items():
                preferences[choice] = slider.value()
                
            self.result = {
                "categorical_preferences": preferences
            }
            
        super().accept()

class OptimizationSettingsDialog(QDialog):
    def __init__(self, parent=None, model=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Optimization Settings")
        self.resize(600, 400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # Basic Settings Tab
        basic_tab = QWidget()
        basic_layout = QFormLayout(basic_tab)
        
        # Acquisition function
        self.acq_function_combo = QComboBox()
        self.acq_function_combo.addItems([
            "Expected Improvement (EI)", 
            "Probability of Improvement (PI)", 
            "Upper Confidence Bound (UCB)"
        ])
        
        # Set current selection based on model
        if self.model.acquisition_function == "ei":
            self.acq_function_combo.setCurrentIndex(0)
        elif self.model.acquisition_function == "pi":
            self.acq_function_combo.setCurrentIndex(1)
        elif self.model.acquisition_function == "ucb":
            self.acq_function_combo.setCurrentIndex(2)
            
        basic_layout.addRow("Acquisition Function:", self.acq_function_combo)
        
        # Exploitation weight slider
        exploit_group = QGroupBox("Exploration-Exploitation Balance")
        exploit_layout = QVBoxLayout(exploit_group)
        
        slider_layout = QHBoxLayout()
        self.exploit_slider = QSlider(Qt.Horizontal)
        self.exploit_slider.setRange(0, 100)
        self.exploit_slider.setValue(int(self.model.exploitation_weight * 100))
        self.exploit_slider.setTickPosition(QSlider.TicksBelow)
        self.exploit_slider.setTickInterval(10)
        
        slider_layout.addWidget(QLabel("Explore"))
        slider_layout.addWidget(self.exploit_slider)
        slider_layout.addWidget(QLabel("Exploit"))
        
        self.exploit_value_label = QLabel(f"{self.model.exploitation_weight:.2f}")
        self.exploit_slider.valueChanged.connect(
            lambda v: self.exploit_value_label.setText(f"{v/100:.2f}")
        )
        slider_layout.addWidget(self.exploit_value_label)
        
        exploit_layout.addLayout(slider_layout)
        
        # Help text
        exploit_help = QLabel(
            "Higher values favor exploitation (focusing on promising areas), "
            "while lower values favor exploration (trying diverse experiments)."
        )
        exploit_help.setWordWrap(True)
        exploit_layout.addWidget(exploit_help)
        
        basic_layout.addRow(exploit_group)
        
        # Advanced Settings Tab
        advanced_tab = QWidget()
        advanced_layout = QFormLayout(advanced_tab)
        
        # Thompson sampling
        self.thompson_check = QCheckBox("Use Thompson Sampling")
        self.thompson_check.setChecked(self.model.use_thompson_sampling)
        self.thompson_check.setToolTip(
            "Adds controlled randomness to the selection process to improve exploration"
        )
        advanced_layout.addRow("Sampling Method:", self.thompson_check)
        
        # Minimum points
        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(1, 20)
        self.min_points_spin.setValue(self.model.min_points_in_model)
        self.min_points_spin.setToolTip(
            "Minimum number of experiments required before using model-based suggestions"
        )
        advanced_layout.addRow("Min. Points for Model:", self.min_points_spin)
        
        # Exploration noise
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.001, 0.5)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setDecimals(3)
        self.noise_spin.setValue(self.model.exploration_noise)
        self.noise_spin.setToolTip(
            "Amount of noise added during Thompson sampling (higher = more exploration)"
        )
        advanced_layout.addRow("Exploration Noise:", self.noise_spin)
        
        # Add tabs
        tabs.addTab(basic_tab, "Basic Settings")
        tabs.addTab(advanced_tab, "Advanced Settings")
        
        layout.addWidget(tabs)
        
        # Help section
        help_group = QGroupBox("About Optimization Settings")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "These settings control how the Bayesian optimization algorithm searches for optimal conditions.\n\n"
            "Acquisition Functions:\n"
            "- Expected Improvement (EI): Balances exploration and exploitation, good default choice\n"
            "- Probability of Improvement (PI): More exploitative, focuses on promising areas\n"
            "- Upper Confidence Bound (UCB): More explorative, samples from uncertain regions\n\n"
            "Thompson Sampling adds controlled randomness to promote exploration and prevent getting stuck in local optima.\n\n"
            "Adjust these parameters if you observe that the optimization is converging too slowly or getting stuck."
        )
        help_layout.addWidget(help_text)
        
        layout.addWidget(help_group)
        
        # Buttons
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        
        layout.addLayout(button_box)
        
    def accept(self):
        # Update model with settings
        acq_index = self.acq_function_combo.currentIndex()
        if acq_index == 0:
            self.model.acquisition_function = "ei"
        elif acq_index == 1:
            self.model.acquisition_function = "pi"
        elif acq_index == 2:
            self.model.acquisition_function = "ucb"
            
        # Update exploitation weight
        self.model.exploitation_weight = self.exploit_slider.value() / 100.0
        
        # Update advanced settings
        self.model.use_thompson_sampling = self.thompson_check.isChecked()
        self.model.min_points_in_model = self.min_points_spin.value()
        self.model.exploration_noise = self.noise_spin.value()
        
        super().accept()

class ParameterLinkDialog(QDialog):
    def __init__(self, parent=None, model=None, param_name=None):
        super().__init__(parent)
        self.model = model
        self.param_name = param_name
        
        if not param_name or param_name not in model.parameters:
            self.reject()
            return
            
        self.param = model.parameters[param_name]
        
        self.setWindowTitle(f"Parameter Links for {param_name}")
        self.resize(700, 500)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Parameter linking allows the model to learn relationships between parameters. "
            "When a positive link exists, successful values of one parameter will influence suggestions for another. "
            "Negative links create inverse relationships between parameters."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Current links
        links_group = QGroupBox("Current Parameter Links")
        links_layout = QVBoxLayout(links_group)
        
        self.links_table = QTableWidget()
        self.links_table.setColumnCount(3)
        self.links_table.setHorizontalHeaderLabels(["Parameter", "Influence Type", "Strength"])
        self.links_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.links_table.setSelectionBehavior(QTableWidget.SelectRows)
        links_layout.addWidget(self.links_table)
        
        # Populate links table
        self.update_links_table()
        
        # Remove link button
        remove_btn = QPushButton("Remove Selected Link")
        remove_btn.clicked.connect(self.remove_link)
        links_layout.addWidget(remove_btn)
        
        layout.addWidget(links_group)
        
        # Add new link
        new_link_group = QGroupBox("Add New Parameter Link")
        new_link_layout = QFormLayout(new_link_group)
        
        self.link_param_combo = QComboBox()
        for name in self.model.parameters.keys():
            if name != self.param_name:
                self.link_param_combo.addItem(name)
                
        self.link_type_combo = QComboBox()
        self.link_type_combo.addItems(["Positive", "Negative"])
        
        self.link_strength_spin = QDoubleSpinBox()
        self.link_strength_spin.setRange(0.1, 1.0)
        self.link_strength_spin.setSingleStep(0.1)
        self.link_strength_spin.setValue(0.5)
        self.link_strength_spin.setDecimals(1)
        
        new_link_layout.addRow("Linked Parameter:", self.link_param_combo)
        new_link_layout.addRow("Influence Type:", self.link_type_combo)
        new_link_layout.addRow("Influence Strength:", self.link_strength_spin)
        
        add_btn = QPushButton("Add Link")
        add_btn.clicked.connect(self.add_link)
        new_link_layout.addRow("", add_btn)
        
        layout.addWidget(new_link_group)
        
        # Explanation of influence types
        explanation_group = QGroupBox("How Parameter Linking Works")
        explanation_layout = QVBoxLayout(explanation_group)
        
        explanation_text = QTextEdit()
        explanation_text.setReadOnly(True)
        explanation_text.setPlainText(
            "Positive Link: When parameter A performs well at high values, parameter B will be suggested with higher values.\n\n"
            "Negative Link: When parameter A performs well at high values, parameter B will be suggested with lower values.\n\n"
            "Strength: Controls how much influence the linked parameter has on suggestions (0.1 = weak, 1.0 = strong).\n\n"
            "Example: If high Temperature (150°C) with DMSO solvent gave good results, a positive link will suggest high "
            "Temperature when DMSO is selected in future experiments."
        )
        explanation_layout.addWidget(explanation_text)
        
        layout.addWidget(explanation_group)
        
        # Visualization of links
        vis_group = QGroupBox("Link Visualization")
        vis_layout = QVBoxLayout(vis_group)
        
        self.vis_canvas = MplCanvas(self, width=5, height=3)
        vis_layout.addWidget(self.vis_canvas)
        
        layout.addWidget(vis_group)
        
        # Button box
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("Close")
        
        self.ok_button.clicked.connect(self.accept)
        
        button_box.addWidget(self.ok_button)
        
        layout.addLayout(button_box)
        
        # Draw links visualization
        self.update_links_visualization()
        
    def update_links_table(self):
        self.links_table.setRowCount(0)
        
        if not hasattr(self.param, 'linked_parameters'):
            return
            
        for linked_param, link_info in self.param.linked_parameters.items():
            row = self.links_table.rowCount()
            self.links_table.insertRow(row)
            
            # Parameter name
            name_item = QTableWidgetItem(linked_param)
            self.links_table.setItem(row, 0, name_item)
            
            # Influence type
            type_item = QTableWidgetItem(link_info['type'].capitalize())
            type_item.setTextAlignment(Qt.AlignCenter)
            if link_info['type'] == 'positive':
                type_item.setBackground(QColor(224, 255, 224))
            else:
                type_item.setBackground(QColor(255, 224, 224))
            self.links_table.setItem(row, 1, type_item)
            
            # Strength
            strength_item = QTableWidgetItem(f"{link_info['strength']:.1f}")
            strength_item.setTextAlignment(Qt.AlignCenter)
            self.links_table.setItem(row, 2, strength_item)
            
    def update_links_visualization(self):
        self.vis_canvas.axes.clear()
        
        if not hasattr(self.param, 'linked_parameters') or not self.param.linked_parameters:
            self.vis_canvas.axes.text(0.5, 0.5, "No parameter links defined", 
                ha='center', va='center', transform=self.vis_canvas.axes.transAxes)
            self.vis_canvas.draw()
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get all parameters involved in links
        all_params = set([self.param_name])
        for linked_param in self.param.linked_parameters.keys():
            all_params.add(linked_param)
            
        # Convert to list for indexing
        params_list = list(all_params)
        n_params = len(params_list)
        
        # Create adjacency matrix for the link graph
        matrix = np.zeros((n_params, n_params))
        colors = []
        
        center_idx = params_list.index(self.param_name)
        
        for linked_param, link_info in self.param.linked_parameters.items():
            linked_idx = params_list.index(linked_param)
            
            # Set the edge weight
            influence = link_info['strength']
            
            # Set color based on influence type
            if link_info['type'] == 'positive':
                color = 'green'
            else:
                color = 'red'
                
            # Only draw edges from center to linked params
            matrix[center_idx, linked_idx] = influence
            colors.append(color)
            
        # Create positions for nodes in a circle
        theta = np.linspace(0, 2 * np.pi, n_params, endpoint=False)
        pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}
        
        # Draw the nodes
        for i, param in enumerate(params_list):
            x, y = pos[i]
            circle = plt.Circle((x, y), 0.1, color='#4285f4' if i == center_idx else '#ea4335')
            self.vis_canvas.axes.add_patch(circle)
            self.vis_canvas.axes.text(x, y, param, ha='center', va='center', fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
        # Draw the edges
        color_idx = 0
        for i in range(n_params):
            for j in range(n_params):
                if matrix[i, j] > 0:
                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    
                    # Calculate arrow points with appropriate scaling
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Adjust endpoints to hit circles
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    
                    x1_adj = x1 + 0.15 * dx_norm
                    y1_adj = y1 + 0.15 * dy_norm
                    x2_adj = x2 - 0.15 * dx_norm
                    y2_adj = y2 - 0.15 * dy_norm
                    
                    # Draw arrow
                    self.vis_canvas.axes.arrow(x1_adj, y1_adj, x2_adj-x1_adj, y2_adj-y1_adj,
                                     color=colors[color_idx], width=0.02*matrix[i, j],
                                     head_width=0.07, head_length=0.1, length_includes_head=True,
                                     alpha=0.8)
                    color_idx += 1
        
        # Set equal aspect and remove axis
        self.vis_canvas.axes.set_aspect('equal')
        self.vis_canvas.axes.axis('off')
        
        # Set limits
        margin = 0.2
        self.vis_canvas.axes.set_xlim(-1.1 - margin, 1.1 + margin)
        self.vis_canvas.axes.set_ylim(-1.1 - margin, 1.1 + margin)
        
        self.vis_canvas.draw()
        
    def add_link(self):
        linked_param = self.link_param_combo.currentText()
        if not linked_param or linked_param == self.param_name:
            return
            
        link_type = self.link_type_combo.currentText().lower()
        link_strength = self.link_strength_spin.value()
        
        # Add the link
        if hasattr(self.param, 'add_linked_parameter'):
            self.param.add_linked_parameter(linked_param, link_strength, link_type)
            
            # Update the UI
            self.update_links_table()
            self.update_links_visualization()
            
    def remove_link(self):
        selected_items = self.links_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        linked_param = self.links_table.item(row, 0).text()
        
        # Remove the link
        if hasattr(self.param, 'remove_linked_parameter'):
            self.param.remove_linked_parameter(linked_param)
            
            # Update the UI
            self.update_links_table()
            self.update_links_visualization()