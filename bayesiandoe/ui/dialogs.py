from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QRadioButton, QGroupBox, QComboBox, QCheckBox, QTextEdit,
    QMessageBox, QSlider, QFrame, QTableWidget, QTableWidgetItem,
    QScrollArea, QListWidget, QTabWidget
)
from ..visualizations import plot_parameter_importance
from ..ui.canvas import MplCanvas
from ..core import settings

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
        self.params = params
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
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setDecimals(settings.rounding_precision)
            
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
        
    def accept(self):
        results = {}
        
        for obj, spin in self.result_spins.items():
            value = spin.value()
            results[obj] = value / 100.0
            
        notes = self.notes_edit.toPlainText().strip()
        if notes:
            self.params["notes"] = notes
            
        self.result = {
            "params": self.params,
            "results": results
        }
        
        super().accept()

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
        self.resize(450, 350)
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
        self.canvas.axes.axvline(self.mean_spin.value() - self.std_spin.value(), color='g', linestyle='--', alpha=0.7)
        self.canvas.axes.axvline(self.mean_spin.value() + self.std_spin.value(), color='g', linestyle='--', alpha=0.7)
        
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