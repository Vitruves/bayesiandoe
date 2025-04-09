from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QTableWidgetItem, QPushButton, QListWidgetItem,
    QLabel, QHBoxLayout, QSlider, QWidget
)
from PySide6.QtGui import QColor

def log(self, message):
    self.log_display.log(message)
    self.status_label.setText(message.strip())
    QApplication.processEvents()

def update_ui_from_model(self):
    self.param_table.update_from_model(self.model)
    update_prior_table(self)
    update_parameter_combos(self)
    update_prior_param_buttons(self)
    self.experiment_table.update_columns(self.model)
    self.experiment_table.update_from_planned(self.model, self.round_start_indices)
    self.best_table.update_columns(self.model)
    self.best_table.update_from_model(self.model, self.n_best_spin.value())
    self.all_results_table.update_from_model(self.model)
    
    self.current_round_label.setText(str(self.current_round))
    if self.model.experiments and self.model.planned_experiments:
        self.results_count_label.setText(f"{len(self.model.experiments)} / {len(self.model.planned_experiments)}")
        self.total_exp_label.setText(str(len(self.model.planned_experiments)))
        
    completed = len(self.model.experiments)
    total = len(self.model.planned_experiments) if self.model.planned_experiments else 0
    if total > 0:
        progress = int(100 * completed / total)
        self.progress_bar.setValue(progress)
    else:
        self.progress_bar.setValue(0)

def update_parameter_combos(self):
    prior_param = self.prior_param_combo.currentText() if hasattr(self, 'prior_param_combo') and self.prior_param_combo.count() > 0 else ""
    viz_param = self.viz_param_combo.currentText() if self.viz_param_combo.count() > 0 else ""
    x_param = self.x_param_combo.currentText() if self.x_param_combo.count() > 0 else ""
    y_param = self.y_param_combo.currentText() if self.y_param_combo.count() > 0 else ""
    
    if hasattr(self, 'prior_param_combo'):
        self.prior_param_combo.clear()
    self.viz_param_combo.clear()
    self.x_param_combo.clear()
    self.y_param_combo.clear()
    
    param_names = list(self.model.parameters.keys())
    
    if hasattr(self, 'prior_param_combo'):
        self.prior_param_combo.addItems(param_names)
    self.viz_param_combo.addItems(param_names)
    self.x_param_combo.addItems(param_names)
    self.y_param_combo.addItems(param_names)
    
    if prior_param in param_names and hasattr(self, 'prior_param_combo'):
        self.prior_param_combo.setCurrentText(prior_param)
        
    if viz_param in param_names:
        self.viz_param_combo.setCurrentText(viz_param)
        
    if x_param in param_names:
        self.x_param_combo.setCurrentText(x_param)
        
    if y_param in param_names:
        self.y_param_combo.setCurrentText(y_param)
        
    if hasattr(self, 'prior_param_combo') and self.prior_param_combo.count() > 0:
        update_prior_ui(self)

def update_prior_table(self):
    self.prior_table.setRowCount(0)
    
    for name, param in self.model.parameters.items():
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

def update_prior_ui(self):
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
                for j in reversed(range(item.layout().count())):
                    sub_item = item.layout().itemAt(j)
                    if sub_item and sub_item.widget():
                        sub_item.widget().deleteLater()
                self.categorical_layout.removeItem(item)
    
    if param.param_type in ["continuous", "discrete"]:
        self.prior_ui_stack.setCurrentIndex(0)
        
        self.prior_mean_spin.setRange(param.low, param.high)
        
        if param.prior_mean is not None and param.prior_std is not None:
            self.prior_mean_spin.setValue(param.prior_mean)
            self.prior_std_spin.setValue(param.prior_std)
            
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
            self.prior_mean_spin.setValue((param.high + param.low) / 2)
            self.prior_confidence_combo.setCurrentText("Medium")
            update_std_from_confidence(self, "Medium")
            
    else:
        self.prior_ui_stack.setCurrentIndex(1)
        
        self.categorical_layout.addWidget(QLabel("Set preference weights for each category:"))
        
        from PySide6.QtWidgets import QHBoxLayout, QSlider, QLabel
        
        for i, choice in enumerate(param.choices):
            row_layout = QHBoxLayout()
            
            label = QLabel(choice)
            label.setMinimumWidth(100)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 10)
            
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

def update_prior_param_buttons(self):
    from PySide6.QtWidgets import QPushButton
    
    while self.param_buttons_layout.count():
        item = self.param_buttons_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
            
    for name, param in self.model.parameters.items():
        button = QPushButton(name)
        
        has_prior = False
        if param.param_type in ["continuous", "discrete"] and param.prior_mean is not None:
            has_prior = True
            button.setText(f"{name} ✓")
        elif param.param_type == "categorical" and hasattr(param, 'categorical_preferences') and param.categorical_preferences:
            has_prior = True
            button.setText(f"{name} ✓")
            
        if has_prior:
            button.setStyleSheet("QPushButton { font-weight: bold; background-color: #e6f2ff; }")
        
        from .ui_callbacks import on_param_button_clicked
        button.clicked.connect(lambda checked, n=name: on_param_button_clicked(self, n))
        
        self.param_buttons_layout.addWidget(button)
        
    help_btn = QPushButton("About Priors")
    
    from .ui_callbacks import show_prior_help
    help_btn.clicked.connect(lambda: show_prior_help(self))
    self.param_buttons_layout.addWidget(help_btn)

def update_best_result_label(self):
    if not self.model.experiments:
        self.best_result_label.setText("N/A")
        return
        
    best_score = 0.0
    for exp in self.model.experiments:
        if 'score' in exp and exp['score'] > best_score:
            best_score = exp['score']
            
    best_score *= 100.0
    
    self.best_result_label.setText(f"{best_score:.2f}%")

def update_rounding_settings(self):
    from ..core import settings
    
    settings.auto_round = self.auto_round_check.isChecked()
    settings.rounding_precision = self.precision_spin.value()
    settings.smart_rounding = self.smart_round_check.isChecked()
    
    update_ui_from_model(self)
    
    auto_status = "enabled" if settings.auto_round else "disabled"
    smart_status = "enabled" if settings.smart_rounding else "disabled"
    log(self, f"-- Rounding settings updated: Auto-round {auto_status}, Precision {settings.rounding_precision}, Smart rounding {smart_status}")

def update_std_from_confidence(self, confidence):
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