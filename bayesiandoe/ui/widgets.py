import sys
import os
import time
import random
import logging
import numpy as np
import concurrent.futures
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, Signal, QMetaObject
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor, QPainter, QBrush, QPen, QLinearGradient, QImage
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QSplashScreen, QFrame, QHeaderView,
    QLabel, QMessageBox, QComboBox, QHBoxLayout, QWidget, QLayout, QVBoxLayout, QPushButton
)
from ..core import _calculate_parameter_distance, settings
import matplotlib.pyplot as plt

class LogDisplay:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger('BayesianDOE')
        
    def log(self, message):
        self.logger.info(message)

class SplashScreen(QSplashScreen):
    def __init__(self):
        self.splash_pix = QPixmap(700, 500)
        self.splash_pix.fill(Qt.transparent)
        
        super().__init__(self.splash_pix, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMask(self.splash_pix.mask())
        
        self.progress = 0
        self.counter = 0
        self.duration_ms = 5000  # Reduced minimum display time
        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(20)  # More frequent updates for smoother animation
        
        # Initialize system details
        self.system_details = "Detecting system..."
        
        # Define essential and non-essential packages
        self.essential_packages = [
            ("numpy", "Loading numerical routines"),
            ("pandas", "Loading data structures"),
            ("PySide6", "Initializing UI components")
        ]
        
        self.nonessential_packages = [
            ("matplotlib", "Loading visualization libraries"),
            ("scipy", "Loading scientific computing libraries"),
            ("optuna", "Loading optimization framework"),
            ("sklearn", "Loading machine learning libraries")
        ]
        
        # Package loading state
        self.current_package_index = 0
        self.package_loading_complete = False
        self.essential_loading_complete = False
        self.loaded_packages = 0
        self.missing_packages = []
        
        # Pre-render background to improve performance
        self.cached_background = None
        self.cached_molecular = None
        
        # Start package loading threads with larger thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.load_essential_future = self.executor.submit(self.load_essential_packages)
        
        # Draw initial splash screen immediately
        self.draw_splash()
    
    def load_essential_packages(self):
        """Load essential packages first in parallel"""
        try:
            import platform
            system_info = f"{platform.system()} {platform.release()}"
            python_version = platform.python_version()
            self.system_details = f"{system_info} | Python {python_version}"
        except:
            self.system_details = "System info unavailable"
            
        # Load essential packages
        loaded_count = 0
        missing_packages = []
        
        for i, (package, message) in enumerate(self.essential_packages):
            try:
                self.current_package_index = i
                
                # Import the package
                __import__(package)
                
                loaded_count += 1
            except ImportError:
                missing_packages.append(package)
        
        # Update the loaded package count
        self.loaded_packages = loaded_count
        self.missing_packages.extend(missing_packages)
        
        self.essential_loading_complete = True
        
        # Start loading non-essential packages
        self.nonessential_future = self.executor.submit(self.load_nonessential_packages)
        
        return loaded_count, missing_packages
    
    def load_nonessential_packages(self):
        """Load non-essential packages after essential ones are loaded"""
        loaded_count = 0
        missing_packages = []
        
        offset = len(self.essential_packages)
        for i, (package, message) in enumerate(self.nonessential_packages):
            try:
                self.current_package_index = offset + i
                
                # Import the package
                __import__(package)
                
                loaded_count += 1
            except ImportError:
                missing_packages.append(package)
        
        # Update total package count
        self.loaded_packages += loaded_count
        self.missing_packages.extend(missing_packages)
        
        self.package_loading_complete = True
        return loaded_count, missing_packages
    
    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        
    def update_progress(self):
        elapsed_time = (time.time() - self.start_time) * 1000
        
        # Calculate progress based on package loading and elapsed time
        if self.package_loading_complete:
            # If all packages are loaded, complete the progress
            time_progress = min(int(elapsed_time / self.duration_ms * 100), 100)
            self.progress = max(self.progress, time_progress)  # Only increase, never decrease
        elif self.essential_loading_complete:
            # If essential packages loaded but non-essential still loading
            essential_weight = 0.6  # Essential packages are 60% of progress
            nonessential_progress = int((self.current_package_index - len(self.essential_packages) + 1) / 
                                       len(self.nonessential_packages) * (100 - essential_weight * 100))
            self.progress = int(essential_weight * 100) + nonessential_progress
        else:
            # During essential package loading
            self.progress = int((self.current_package_index + 1) / 
                              len(self.essential_packages) * 60)  # Cap at 60% until essential complete
        
        self.counter += 1
        
        # Determine if we should continue showing the splash screen
        should_continue = ((elapsed_time < self.duration_ms) or 
                          not self.essential_loading_complete or
                          (not self.package_loading_complete and elapsed_time < 4000))  # 4s max wait time
        
        if should_continue:
            self.draw_splash()
            
            if self.counter % 10 == 0:
                self.raise_()
        else:
            # Ensure all threads are properly cleaned up
            self.timer.stop()
            if not self.executor.shutdown(wait=False):
                self.executor.shutdown(wait=False)
            self.close()
        
    def draw_splash(self):
        painter = QPainter(self.splash_pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        rect = self.splash_pix.rect()
        
        # Fill background
        painter.fillRect(rect, QColor(15, 20, 25))
        
        # Draw molecular background (with caching for performance)
        self.draw_molecular_background(painter, rect)
     
        # Draw main title
        font = QFont("Helvetica", 48, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        
        painter.setPen(QColor(240, 250, 250))
        painter.drawText(rect.adjusted(0, 0, 0, -rect.height()/3), Qt.AlignHCenter | Qt.AlignBottom, "BAYESIAN DOE")
        
        # Draw progress bar
        self.draw_cyber_progress(painter, rect)
        
        # Draw current loading message
        packages = self.essential_packages + self.nonessential_packages
        if self.current_package_index < len(packages):
            _, message = packages[self.current_package_index]
        else:
            message = "Finalizing initialization"
            
        font = QFont("Helvetica", 14)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        painter.setPen(QColor(0, 220, 200))
        painter.drawText(rect.adjusted(0, rect.height()*0.78 + 30, 0, 0), Qt.AlignHCenter | Qt.AlignTop, message)
        
        # Draw version and copyright
        font = QFont("Helvetica", 16)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        painter.setPen(QColor(150, 170, 170))
        
        painter.drawText(rect.adjusted(20, 0, 0, -15), Qt.AlignLeft | Qt.AlignBottom, "v1.1.0")
        painter.drawText(rect.adjusted(0, 0, -20, -15), Qt.AlignRight | Qt.AlignBottom, "© 2025 Johan H.G. Natter")
        
        # Show system details
        font = QFont("Helvetica", 12)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        painter.setPen(QColor(120, 140, 140))
        painter.drawText(rect.adjusted(0, 0, 0, -40), Qt.AlignHCenter | Qt.AlignBottom, self.system_details)
        
        # Only show extra stats when complete (reduces drawing operations)
        if self.package_loading_complete:
            stats_text = f"Loaded {self.loaded_packages}/{len(self.essential_packages) + len(self.nonessential_packages)} packages"
            painter.drawText(rect.adjusted(0, 0, 0, -60), Qt.AlignHCenter | Qt.AlignBottom, stats_text)
        
        painter.end()
        self.setPixmap(self.splash_pix)
        
    def draw_molecular_background(self, painter, rect):
        # Use static seed for consistent appearance
        np.random.seed(42)
        
        # Only redraw full molecular background every few frames
        if self.cached_molecular is None or self.counter % 5 == 0:
            # Create a new pixmap for molecular background
            if self.cached_molecular is None:
                self.cached_molecular = QPixmap(rect.width(), rect.height())
                self.cached_molecular.fill(Qt.transparent)
            
            mol_painter = QPainter(self.cached_molecular)
            mol_painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Clear background
            mol_painter.fillRect(rect, Qt.transparent)
            
            # Draw background grid (static)
            grid_color = QColor(40, 60, 80, 40) if not self.package_loading_complete else QColor(40, 70, 60, 40)
            mol_painter.setPen(QPen(grid_color, 0.8))
            step = 30
            for x in range(0, rect.width(), step):
                mol_painter.drawLine(x, 0, x, rect.height())
            for y in range(0, rect.height(), step):
                mol_painter.drawLine(0, y, rect.width(), y)
            
            # Generate nodes (atoms) positions once and reuse
            if not hasattr(self, 'nodes_positions'):
                self.nodes_positions = []
                for i in range(15):
                    x = np.random.randint(50, rect.width() - 50)
                    y = np.random.randint(50, rect.height() - 50)
                    r = np.random.randint(5, 12)
                    self.nodes_positions.append((x, y, r))
            
            # Generate bonds once
            if not hasattr(self, 'bonds'):
                self.bonds = []
                for i in range(len(self.nodes_positions)):
                    for j in range(i+1, len(self.nodes_positions)):
                        bond_seed = i * 1000 + j
                        np.random.seed(bond_seed)
                        if np.random.random() < 0.4:
                            self.bonds.append((i, j, bond_seed))
            
            # Draw bonds between atoms (semitransparent)
            edge_color = QColor(70, 100, 180, 70) if not self.package_loading_complete else QColor(70, 180, 160, 70)
            
            for i, j, bond_seed in self.bonds:
                x1, y1, _ = self.nodes_positions[i]
                x2, y2, _ = self.nodes_positions[j]
                
                # Add a subtle pulse effect to bonds
                phase = (self.counter * 2 + bond_seed) % 360
                pulse_alpha = int(40 + 30 * np.sin(np.radians(phase)))
                edge_color.setAlpha(pulse_alpha)
                mol_painter.setPen(QPen(edge_color, 1.2))
                mol_painter.drawLine(x1, y1, x2, y2)
                
            mol_painter.end()
        
        # Draw the cached molecular background
        painter.drawPixmap(0, 0, self.cached_molecular)
        
        # Draw animated elements on top (nodes with pulsing effect)
        for i, (x, y, r) in enumerate(self.nodes_positions):
            # Add a subtle oscillation effect
            phase = (self.counter + i * 15) % 360
            pulse = 0.2 * np.sin(np.radians(phase))
            r_animated = r * (1.0 + pulse)
            
            # Use different colors based on package loading progress
            if not self.package_loading_complete:
                # Blue-cyan scheme during loading
                color = QColor(
                    np.random.randint(30, 60),
                    np.random.randint(100, 180),
                    np.random.randint(180, 220),
                    100 + int(20 * pulse)  # Make opacity pulse too
                )
            else:
                # Green-cyan scheme when complete
                color = QColor(
                    np.random.randint(30, 70),
                    np.random.randint(160, 220),
                    np.random.randint(180, 210),
                    100 + int(20 * pulse)
                )
                
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(x - r_animated, y - r_animated, r_animated*2, r_animated*2)
            
    def draw_cyber_progress(self, painter, rect):
        bar_width = rect.width() * 0.6
        bar_height = 14
        
        x = (rect.width() - bar_width) / 2
        y = rect.height() * 0.78
        
        segments = 20
        seg_width = bar_width / segments
        
        # Draw background segments
        for i in range(segments):
            color = QColor(25, 35, 45) if i % 2 == 0 else QColor(30, 40, 50)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            
            seg_x = x + (i * seg_width)
            painter.drawRect(seg_x, y, seg_width - 1, bar_height)
            
        # Calculate filled segments
        filled_segments = int((self.progress / 100) * segments)
        
        # Determine color based on loading state
        if self.package_loading_complete:
            color1 = QColor(0, 220, 200)  # Success green-cyan
            color2 = QColor(0, 200, 180)
        else:
            color1 = QColor(0, 180, 255)  # Loading blue
            color2 = QColor(0, 150, 220)
            
        # Draw filled segments
        for i in range(filled_segments):
            color = color1 if i % 2 == 0 else color2
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            
            seg_x = x + (i * seg_width)
            painter.drawRect(seg_x, y, seg_width - 1, bar_height)
            
            # Add lighting effect
            painter.setPen(QPen(QColor(220, 255, 255, 150), 1))
            painter.drawLine(seg_x, y, seg_x + seg_width - 1, y)
            
        # Draw progress text
        font = QFont("Helvetica", 13)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        painter.setFont(font)
        
        # Shadow for text
        painter.setPen(QColor(0, 0, 0, 100))
        progress_text = f"INITIALIZING SYSTEM {self.progress}%"
        painter.drawText(QRect(x + 2, y - 30 + 2, bar_width, 25), Qt.AlignCenter, progress_text)
        
        # Main text
        text_color = QColor(0, 220, 200) if self.package_loading_complete else QColor(0, 180, 255)
        painter.setPen(text_color)
        painter.drawText(QRect(x, y - 30, bar_width, 25), Qt.AlignCenter, progress_text)

class ParameterTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Parameter", "Type", "Range/Choices", "Unit"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
    def update_from_model(self, model):
        self.setRowCount(0)
        for name, param in model.parameters.items():
            row = self.rowCount()
            self.insertRow(row)
            
            self.setItem(row, 0, QTableWidgetItem(name))
            self.setItem(row, 1, QTableWidgetItem(param.param_type))
            
            if param.param_type in ["continuous", "discrete"]:
                low_val = settings.format_value(param.low)
                high_val = settings.format_value(param.high)
                range_str = f"{low_val} - {high_val}"
            else:
                range_str = ", ".join(str(c) for c in param.choices)
                
            self.setItem(row, 2, QTableWidgetItem(range_str))
            self.setItem(row, 3, QTableWidgetItem(param.units or ""))

class ExperimentTable(QTableWidget):
    # Define the missing signal
    clear_table_signal = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(["Round", "ID", "Status", "Parameters", "Results", "Score", "Notes"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.cellChanged.connect(self.handle_cell_change)
        
        # Connect the signal to clear method
        self.clear_table_signal.connect(self.clearContents)
        
        # Store main window reference for later use
        self.main_window = self.get_main_window()
        
        # Initialize round selector
        self.round_selector = None
        self.round_select_widget = None
        
        # Add a direct reference to store the model
        self.model = None
        
        # Initialize thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._prediction_start_time = 0
        self._prediction_future = None
        
    def get_main_window(self):
        """Find the main window instance by traversing up the widget tree"""
        from ..ui.main_window import BayesianDOEApp
        parent = self.parent()
        
        # Traverse up the parent hierarchy until we find the main window
        while parent and not isinstance(parent, BayesianDOEApp):
            parent = parent.parent()
        
        return parent
    
    def update_columns(self, model):
        # Store reference to the model
        self.model = model
        
        columns = ["Round", "ID"] + list(model.parameters.keys()) + model.objectives + ["Predicted"]
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        # Set up parameters columns to stretch
        for i in range(2, 2 + len(model.parameters)):
            self.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        
        # Set up objectives columns to be resizable to content
        for i in range(2 + len(model.parameters), len(columns)-1):
            self.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
            
            # Add visual cue for editable objective columns
            obj_header = self.horizontalHeaderItem(i)
            if obj_header:
                obj_name = obj_header.text()
                obj_header.setText(f"{obj_name}*")
                obj_header.setToolTip(f"Double-click to edit {obj_name} value")
                obj_header.setBackground(QColor(240, 255, 240))  # Light green background
        
        # Set up prediction column
        predict_col = len(columns) - 1
        self.horizontalHeader().setSectionResizeMode(predict_col, QHeaderView.ResizeToContents)
        
        # Store column indices for faster access
        self.param_columns_start = 2
        self.param_columns_end = 2 + len(model.parameters) - 1
        self.objective_columns_start = 2 + len(model.parameters)
        self.objective_columns_end = self.objective_columns_start + len(model.objectives) - 1
        
        # Add a tooltip to the table view
        self.setToolTip("Double-click on result cells to enter values directly")
        
        # Style the table to indicate editable columns
        self.setStyleSheet("""
            QTableWidget::item:hover {
                background-color: rgba(240, 255, 240, 100);
                border: 1px solid #4CAF50;
            }
        """)

    def handle_cell_change(self, row, column):
        """Handle cell value changes - specifically for entering results"""
        try:
            # Only process if we have a valid model
            if not self.model:
                return
                
            # Only handle changes in objective columns
            if column < self.objective_columns_start or column > self.objective_columns_end:
                return
            
            # Check if this is a previous round (read-only)
            main_window = self.get_main_window()
            if main_window and hasattr(main_window, 'view_round') and hasattr(main_window, 'current_round'):
                if main_window.view_round < main_window.current_round:
                    # Don't allow edits to previous rounds
                    QMessageBox.warning(self, "Read-Only", 
                                     "Previous rounds are read-only. Please go to the current round to add new results.")
                    # Restore the table (discard changes)
                    self.update_from_planned(self.model, getattr(main_window, 'round_start_indices', []))
                    return
                
            # Identify which experiment this is
            id_item = self.item(row, 1)
            if not id_item or not id_item.text().isdigit():
                return
                
            exp_id = int(id_item.text()) - 1
            if exp_id < 0 or exp_id >= len(self.model.planned_experiments):
                return
                
            # Get value that was entered
            result_item = self.item(row, column)
            if not result_item:
                return
                
            result_text = result_item.text().strip()
            if not result_text:
                return
                
            # Parse the value (handle percentage signs)
            try:
                has_percent = '%' in result_text
                result_value = float(result_text.replace('%', '').strip())
                
                # Convert to 0-1 scale if needed
                if has_percent:
                    # Value is explicitly marked as percentage (e.g., "78.5%"), convert to 0-1
                    result_value = result_value / 100.0
                elif result_value > 1.0:
                    # Value > 1 without % sign, assume it's still a percentage (e.g., "78.5")
                    result_value = result_value / 100.0
                # else: value is already in 0-1 scale, keep as is
                
                # Validate the final value is in 0-1 range
                if result_value < 0.0 or result_value > 1.0:
                    QMessageBox.warning(self, "Invalid Value", 
                                     "Please enter a value between 0-100% or 0-1")
                    if result_item:
                        self.blockSignals(True)
                        result_item.setText("")
                        self.blockSignals(False)
                    return
                    
            except ValueError:
                # Restore previous value or clear cell
                QMessageBox.warning(self, "Invalid Value", 
                                  "Please enter a valid number (e.g., 78.5 or 78.5%)")
                if result_item:
                    self.blockSignals(True)
                    result_item.setText("")
                    self.blockSignals(False)
                return
                
            # Get which objective this is
            column_idx = column - self.objective_columns_start
            if column_idx < 0 or column_idx >= len(self.model.objectives):
                return
                
            objective_name = self.model.objectives[column_idx]
            
            # Either update existing experiment or create new one
            exp_params = self.model.planned_experiments[exp_id]
            
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
                    break
            
            if existing_exp:
                # Update existing result
                existing_exp['results'][objective_name] = result_value
                print(f"Updated existing result for experiment #{exp_id+1}, {objective_name}={result_value:.4f}")
                
                # Recalculate score
                score = self.model._calculate_composite_score(existing_exp['results'])
                existing_exp['score'] = score
                
                # Format cell with proper percentage
                self.blockSignals(True)
                result_item.setText(f"{result_value*100:.2f}%")
                result_item.setBackground(QColor(224, 255, 224))
                self.blockSignals(False)
                
                # Update other cells in the row with any other objective values
                for i, obj in enumerate(self.model.objectives):
                    if obj != objective_name and obj in existing_exp['results']:
                        obj_col = self.objective_columns_start + i
                        obj_item = self.item(row, obj_col)
                        if obj_item:
                            self.blockSignals(True)
                            obj_item.setText(f"{existing_exp['results'][obj]*100:.2f}%")
                            obj_item.setBackground(QColor(224, 255, 224))
                            self.blockSignals(False)
            else:
                # Create new result
                from datetime import datetime
                
                results = {objective_name: result_value}
                
                new_exp = {
                    'params': exp_params,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Calculate composite score
                score = self.model._calculate_composite_score(results)
                new_exp['score'] = score
                
                # Add to experiments
                self.model.experiments.append(new_exp)
                print(f"Added new result for experiment #{exp_id+1}, {objective_name}={result_value:.4f}")
                
                # Format cell with proper percentage
                self.blockSignals(True)
                result_item.setText(f"{result_value*100:.2f}%")
                result_item.setBackground(QColor(224, 255, 224))
                self.blockSignals(False)
                
                # Highlight the entire row
                for col in range(self.columnCount()):
                    item = self.item(row, col)
                    if item:
                        item.setBackground(QColor(224, 255, 224))
            
            # Notify parent of the change so it can update other tables
            parent = self.parent()
            while parent and not hasattr(parent, 'update_result_tables'):
                parent = parent.parent()
                
            if parent and hasattr(parent, 'update_result_tables'):
                parent.update_result_tables()
                
            # Log the update
            self.log(f" Added result for experiment #{exp_id+1}: {objective_name}={result_value*100:.2f}% - Success")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error handling cell change: {e}")
            print(error_details)
    
    def update_from_planned(self, model, round_start_indices):
        """Update experiment table with planned experiments, filtered by round"""
        try:
            self.clear_table_signal.emit()
            self.setRowCount(0)
            
            if not hasattr(model, 'planned_experiments') or not model.planned_experiments:
                print("No planned experiments found.")
                return
            
            # Get main window to access current_round
            main_window = self.get_main_window()
            if not main_window:
                print("Warning: Could not find main window")
                return
                
            # Get filter round (if viewing a specific round)
            filter_round = getattr(self, 'filter_by_round', None)
            current_round = filter_round if filter_round is not None else main_window.current_round
            
            # Setup the round selector
            self.setup_round_selector(main_window, current_round)
            
            # Calculate indices for current round
            start_idx = 0
            end_idx = len(model.planned_experiments)
            
            if round_start_indices:
                # Set start index for this round
                if current_round > 1 and len(round_start_indices) >= current_round-1:
                    start_idx = round_start_indices[current_round-2]
                
                # Set end index for this round
                if current_round <= len(round_start_indices):
                    end_idx = round_start_indices[current_round-1]
            
            # Get experiments for just this round
            filtered_experiments = model.planned_experiments[start_idx:end_idx]
            print(f"Round {current_round}: Showing {len(filtered_experiments)} experiments from {start_idx} to {end_idx}")
            
            # Get completed experiments
            completed_indices = []
            for i, exp in enumerate(model.experiments):
                if i < len(model.experiments):
                    completed_indices.append(i)
                    
            # Add rows to table for each experiment
            for i, exp_data in enumerate(filtered_experiments):
                self.insertRow(i)
                
                # Global experiment ID
                exp_id = start_idx + i + 1
                
                # Add round number 
                round_item = QTableWidgetItem(str(current_round))
                round_item.setFlags(round_item.flags() & ~Qt.ItemIsEditable)
                self.setItem(i, 0, round_item)
                
                # Add experiment ID
                id_item = QTableWidgetItem(str(exp_id))
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
                self.setItem(i, 1, id_item)
                
                # Get params from planned experiment - ensure correct structure
                exp_params = {}
                if 'params' in exp_data:
                    # New format with 'params' key
                    exp_params = exp_data['params']
                else:
                    # Old format (parameters directly in experiment dict)
                    for param_name in model.parameters.keys():
                        if param_name in exp_data:
                            exp_params[param_name] = exp_data[param_name]
                    # Update with structured format to ensure consistency
                    exp_data['params'] = exp_params
                
                # Add parameters
                for j, param_name in enumerate(model.parameters.keys()):
                    param_col = j + 2  # parameters start at column 2
                    param_value = exp_params.get(param_name, "--")
                    
                    # Format value appropriately
                    if isinstance(param_value, float):
                        param_value_str = f"{param_value:.4g}"
                    else:
                        param_value_str = str(param_value)
                    
                    param_item = QTableWidgetItem(param_value_str)
                    param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)
                    self.setItem(i, param_col, param_item)
                
                # Add result placeholder for each objective
                result_col_start = 2 + len(model.parameters)
                for j, obj in enumerate(model.objectives):
                    result_col = result_col_start + j
                    
                    # Check if this experiment already has results
                    matching_exp_idx = self._find_matching_planned_experiment(model, exp_data, completed_indices)
                    
                    if matching_exp_idx is not None:
                        # This experiment already has results, show them
                        result_value = model.experiments[matching_exp_idx].get('results', {}).get(obj)
                        if result_value is not None:
                            result_str = f"{result_value*100:.2f}%"
                            result_item = QTableWidgetItem(result_str)
                            result_item.setBackground(QColor(224, 255, 224))  # Light green
                            result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
                        else:
                            result_item = QTableWidgetItem("--")
                            result_item.setToolTip("Double-click to enter result")
                    else:
                        # No results yet
                        result_item = QTableWidgetItem("--")
                        result_item.setToolTip("Double-click to enter result")
                    
                    self.setItem(i, result_col, result_item)
                
                # Add prediction placeholder
                predict_col = result_col_start + len(model.objectives)
                predict_item = QTableWidgetItem("--")
                predict_item.setFlags(predict_item.flags() & ~Qt.ItemIsEditable)
                self.setItem(i, predict_col, predict_item)
            
            # Enable prediction calculation if we have completed experiments
            if completed_indices:
                exp_id_to_row = {start_idx + i + 1: i for i in range(len(filtered_experiments))}
                self._async_update_predictions(model, completed_indices, exp_id_to_row)
            
            self.resizeColumnsToContents()
            
        except Exception as e:
            import traceback
            print(f"Error in update_from_planned: {e}")
            print(traceback.format_exc())
    
    def setup_round_selector(self, main_window, current_round):
        """Set up the round selector widget properly attached to main window"""
        max_round = main_window.current_round
        
        # Find appropriate experiment tab parent
        experiment_tab = None
        if hasattr(main_window, 'tab_widget'):
            for i in range(main_window.tab_widget.count()):
                if main_window.tab_widget.tabText(i) == "Experiments":
                    experiment_tab = main_window.tab_widget.widget(i)
                    break
        
        # If we already have a round selector widget, just update it
        if hasattr(self, 'round_selector') and self.round_selector:
            # Update without triggering signals
            self.round_selector.blockSignals(True)
            self.round_selector.clear()
            for r in range(1, max_round+1):
                self.round_selector.addItem(f"Round {r}")
            
            # Make sure we select the correct round
            if current_round >= 1 and current_round <= max_round:
                self.round_selector.setCurrentIndex(current_round-1)
            self.round_selector.blockSignals(False)
            
            # Make sure it's visible
            if hasattr(self, 'round_select_widget'):
                self.round_select_widget.setVisible(max_round > 1)
            return
            
        # Create new round selector
        if experiment_tab:
            # Create a more prominent widget with better styling
            self.round_select_widget = QWidget(experiment_tab)
            self.round_select_widget.setObjectName("RoundSelectorWidget")
            round_select_layout = QHBoxLayout(self.round_select_widget)
            round_select_layout.setContentsMargins(5, 5, 5, 5)
            round_select_layout.setSpacing(10)
            
            # Add a more descriptive label
            round_label = QLabel("<b>View Experiment Round:</b>")
            round_select_layout.addWidget(round_label)
            
            # Create the combo box with improved styling
            self.round_selector = QComboBox()
            self.round_selector.setMinimumWidth(150)
            self.round_selector.setObjectName("RoundSelector")
            self.round_selector.setStyleSheet("""
                QComboBox {
                    padding: 4px 8px;
                    border: 1px solid #999;
                    border-radius: 4px;
                    background-color: #f8f8f8;
                }
                QComboBox:hover {
                    border-color: #4CAF50;
                    background-color: #f0f8f0;
                }
                QComboBox::drop-down {
                    width: 20px;
                }
            """)
            
            # Add rounds
            for r in range(1, max_round+1):
                self.round_selector.addItem(f"Round {r}")
            
            # Make sure we select the correct round
            if current_round >= 1 and current_round <= max_round:
                self.round_selector.setCurrentIndex(current_round-1)
            
            # Connect directly to main window with a more robust connection
            def on_round_selected(index):
                selected_round = index + 1
                # Get a fresh reference to the main window each time
                main_win = self.get_main_window()
                if main_win and hasattr(main_win, 'view_selected_round'):
                    main_win.view_selected_round(selected_round)
                else:
                    print("Warning: Could not find main window or view_selected_round method")
            
            # Disconnect any existing connections first
            try:
                self.round_selector.currentIndexChanged.disconnect()
            except:
                pass  # No existing connections
                
            self.round_selector.currentIndexChanged.connect(on_round_selected)
            round_select_layout.addWidget(self.round_selector)
            
            # Add a help button to explain round navigation
            help_btn = QPushButton("?")
            help_btn.setMaximumWidth(30)
            help_btn.setToolTip("Click to learn about experiment rounds")
            help_btn.clicked.connect(lambda: QMessageBox.information(self, 
                "Experiment Rounds", 
                "You can navigate between different rounds of experiments.\n\n"
                "• Previous rounds are locked (read-only)\n"
                "• Current round is editable\n"
                "• All rounds are included in analysis and predictions"))
            round_select_layout.addWidget(help_btn)
            
            # Only show the selector if we have multiple rounds
            self.round_select_widget.setVisible(max_round > 1)
            
            # Find experiment content layout and place at the top
            content_layout = None
            for layout in experiment_tab.findChildren(QVBoxLayout):
                if layout.objectName() == "experiment_content_layout":
                    content_layout = layout
                    break
            
            # If we found the layout, insert at the beginning
            if content_layout:
                if content_layout.count() > 0:
                    content_layout.insertWidget(0, self.round_select_widget)
                else:
                    content_layout.addWidget(self.round_select_widget)
            else:
                # Fallback - find a layout that contains this table
                parent_widget = self.parent()
                while parent_widget and not isinstance(parent_widget, QVBoxLayout):
                    if hasattr(parent_widget, 'layout') and isinstance(parent_widget.layout(), QVBoxLayout):
                        parent_widget.layout().insertWidget(0, self.round_select_widget)
                        break
                    parent_widget = parent_widget.parent()

    def _async_update_predictions(self, model, completed_indices, exp_id_to_row):
        """Start async calculation of predictions with proper error handling"""
        try:
            self._prediction_start_time = time.time()
            if hasattr(self, '_prediction_future') and self._prediction_future and not self._prediction_future.done():
                self._prediction_future.cancel()
                
            # Make sure executor exists
            if not hasattr(self, 'executor') or self.executor is None or self.executor._shutdown:
                import concurrent.futures
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                
            # Submit the task
            self._prediction_future = self.executor.submit(
                self._calculate_predictions, model, completed_indices, exp_id_to_row
            )
            # Add timeout watchdog timer
            QTimer.singleShot(5000, lambda: self._check_prediction_timeout())
        except Exception as e:
            print(f"Error starting prediction calculation: {e}")
            import traceback
            print(traceback.format_exc())

    def _check_prediction_timeout(self):
        """Check if prediction calculation has timed out and cancel if needed"""
        try:
            if hasattr(self, '_prediction_future') and self._prediction_future and not self._prediction_future.done():
                if time.time() - self._prediction_start_time > 5.0:
                    self._prediction_future.cancel()
                    print("Warning: Prediction calculation timed out - using simple interpolation")
                    # Fall back to simpler method if needed
            
            # Clean up completed futures to prevent memory leaks
            if hasattr(self, '_prediction_future') and self._prediction_future and self._prediction_future.done():
                try:
                    # Extract any exceptions without raising them
                    exception = self._prediction_future.exception(timeout=0)
                    if exception:
                        print(f"Prediction calculation failed: {exception}")
                except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
                    pass
                
                # Clear the reference
                self._prediction_future = None
        except Exception as e:
            print(f"Error in prediction timeout check: {e}")

    def optimize_update_from_planned(self):
        """Optimize table update for large experiments."""
        if len(self.model.planned_experiments) > 100:
            # For large datasets, update in batches
            self.setUpdatesEnabled(False)
            
            # Only show latest round and top 20 rows by default
            current_round = self.current_round
            start_idx = self.round_start_indices[current_round-1] if current_round > 0 and len(self.round_start_indices) > current_round-1 else 0
            
            # Calculate visible range
            visible_start = max(0, start_idx - 5)  # Show 5 experiments from previous round if available
            visible_end = min(len(self.model.planned_experiments), start_idx + 20)
            
            # Only update visible rows initially
            self._update_table_range(visible_start, visible_end)
            
            # Schedule background update for remaining rows
            QTimer.singleShot(100, self._update_remaining_rows)
            
            self.setUpdatesEnabled(True)
        else:
            # For smaller datasets, update all at once
            self._update_table_range(0, len(self.model.planned_experiments))

    def debug_table_state(self):
        """Print debug information about the table state."""
        try:
            model = getattr(self, 'model', None)
            if not model:
                print("No model attached to table")
                return
            
            print(f"Table has {self.rowCount()} rows, {self.columnCount()} columns")
            print(f"Model has {len(model.planned_experiments)} planned experiments")
            print(f"Model has {len(model.experiments)} completed experiments")
            
            # Check column headers
            headers = []
            for i in range(self.columnCount()):
                header_item = self.horizontalHeaderItem(i)
                headers.append(header_item.text() if header_item else f"Col {i}")
            print(f"Column headers: {headers}")
            
            # Check for empty cells in key columns
            empty_cells = 0
            for row in range(self.rowCount()):
                for col in range(min(4, self.columnCount())):
                    if not self.item(row, col) or not self.item(row, col).text():
                        empty_cells += 1
            if empty_cells > 0:
                print(f"WARNING: {empty_cells} empty cells found in key columns")
        except Exception as e:
            print(f"Error in debug_table_state: {e}")

    def log(self, message):
        """Internal logger - forwards to parent log if available."""
        print(message)
        
        # Try to find main window to log message
        parent = self.parent()
        while parent and not hasattr(parent, 'log'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'log'):
            parent.log(message)

    def _find_matching_planned_experiment(self, model, exp_data, completed_indices):
        """Find a matching completed experiment for a planned experiment."""
        if not model.experiments or not completed_indices:
            return None
        
        # Get parameter values from this planned experiment
        planned_params = exp_data.get('params', {})
        if not planned_params:
            # Handle old format where parameters are at top level
            for param_name in model.parameters.keys():
                if param_name in exp_data:
                    planned_params[param_name] = exp_data[param_name]
        
        if not planned_params:
            return None
        
        # For each completed experiment, check if parameters match
        for idx in completed_indices:
            if idx >= len(model.experiments):
                continue
            
            completed_exp = model.experiments[idx]
            completed_params = completed_exp.get('params', {})
            
            # Handle legacy format where parameters are at top level
            if not completed_params:
                completed_params = {}
                for param_name in model.parameters.keys():
                    if param_name in completed_exp:
                        completed_params[param_name] = completed_exp[param_name]
            
            # Check if all parameter values match
            all_match = True
            for param_name, planned_value in planned_params.items():
                if param_name not in completed_params:
                    all_match = False
                    break
                
                completed_value = completed_params[param_name]
                
                # Compare values with tolerance for floating point
                if isinstance(planned_value, float) and isinstance(completed_value, float):
                    if abs(planned_value - completed_value) > 1e-6:
                        all_match = False
                        break
                elif planned_value != completed_value:
                    all_match = False
                    break
                
            if all_match:
                return idx
            
        return None

    def _calculate_predictions(self, model, completed_indices, exp_id_to_row):
        """Calculate predictions for experiments in background thread."""
        try:
            # Skip if no completed experiments
            if not completed_indices:
                return
            
            # Prepare data for training surrogate model
            X_train = []
            y_train = []
            
            for idx in completed_indices:
                if idx >= len(model.experiments):
                    continue
                
                exp = model.experiments[idx]
                if 'params' not in exp or 'results' not in exp:
                    continue
                
                # Extract parameter values
                x_values = []
                for param_name in model.parameters.keys():
                    if param_name in exp['params']:
                        param = model.parameters[param_name]
                        value = exp['params'][param_name]
                        
                        # Normalize categorical values
                        if param.param_type == 'categorical':
                            if value in param.choices:
                                value = param.choices.index(value) / (len(param.choices) - 1) if len(param.choices) > 1 else 0.5
                            else:
                                value = 0.5
                        # Normalize continuous/discrete values
                        elif param.param_type in ['continuous', 'discrete']:
                            value = (value - param.low) / (param.high - param.low) if param.high > param.low else 0.5
                        
                        x_values.append(value)
                    else:
                        x_values.append(0.5)  # Default value if missing
                
                # Calculate composite score
                score = exp.get('score')
                if score is None:
                    score = model._calculate_composite_score(exp.get('results', {}))
                
                if score is not None:
                    X_train.append(x_values)
                    y_train.append(score)
            
            # Train a simple model
            if len(X_train) < 3 or len(y_train) < 3:
                return
            
            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train a simple linear model (we could use more sophisticated models later)
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            
            # Predict for each experiment in the current view
            for exp_id, row in exp_id_to_row.items():
                if exp_id <= 0 or exp_id > len(model.planned_experiments):
                    continue
                
                exp = model.planned_experiments[exp_id - 1]
                if 'params' not in exp:
                    continue
                
                # Extract parameter values
                x_values = []
                for param_name in model.parameters.keys():
                    if param_name in exp['params']:
                        param = model.parameters[param_name]
                        value = exp['params'][param_name]
                        
                        # Normalize categorical values
                        if param.param_type == 'categorical':
                            if value in param.choices:
                                value = param.choices.index(value) / (len(param.choices) - 1) if len(param.choices) > 1 else 0.5
                            else:
                                value = 0.5
                        # Normalize continuous/discrete values
                        elif param.param_type in ['continuous', 'discrete']:
                            value = (value - param.low) / (param.high - param.low) if param.high > param.low else 0.5
                        
                        x_values.append(value)
                    else:
                        x_values.append(0.5)  # Default value if missing
                
                # Make prediction
                x_pred = np.array([x_values])
                prediction = reg.predict(x_pred)[0]
                
                # Store prediction in a closure that can be called from main thread
                prediction_row = row
                prediction_value = prediction
                
                # Update UI with prediction (in main thread)
                def update_prediction():
                    if prediction_row < self.rowCount():
                        predict_col = self.param_columns_end + len(model.objectives) + 1
                        predict_item = self.item(prediction_row, predict_col)
                        if predict_item:
                            predict_item.setText(f"{prediction_value*100:.1f}%")
                            if prediction_value > 0.7:
                                predict_item.setBackground(QColor(200, 255, 200))  # Green for good predictions
                            elif prediction_value > 0.5:
                                predict_item.setBackground(QColor(255, 255, 200))  # Yellow for medium
                            else:
                                predict_item.setBackground(QColor(255, 200, 200))  # Red for poor
                
                # Store the update function and row value as an attribute to avoid it being garbage collected
                prediction_id = f"prediction_{exp_id}"
                setattr(self, prediction_id, (update_prediction, prediction_row, prediction_value))
                
                # Execute in main thread - using the correct signature for invokeMethod
                from PySide6.QtCore import QMetaObject, Qt
                QMetaObject.invokeMethod(self, "_apply_prediction", Qt.QueuedConnection,
                                        Qt.Q_ARG(str, prediction_id))
        
        except Exception as e:
            import traceback
            print(f"Error calculating predictions: {e}")
            print(traceback.format_exc())

    def _apply_prediction(self, prediction_id):
        """Helper method to be called via invokeMethod that applies a prediction update"""
        if hasattr(self, prediction_id):
            update_func, row, value = getattr(self, prediction_id)
            update_func()
            # Clean up after applying
            delattr(self, prediction_id)

    def commit_pending_changes(self):
        """Commit any pending changes to the model when switching between rounds"""
        try:
            if not self.model:
                return
                
            # Force end of any current edits
            self.clearFocus()
            
            # Forcibly apply any active editor
            if self.state() == QTableWidget.EditingState:
                self.commitData(self.currentEditor())
                self.closeEditor(self.currentEditor(), QTableWidget.EndEditHint.SubmitModelCache)
            
            # Check for any highlighted cells that might indicate unsaved changes
            highlighted_cells = []
            for row in range(self.rowCount()):
                for col in range(self.objective_columns_start, self.objective_columns_end + 1):
                    item = self.item(row, col)
                    if item and item.background().color().name() == QColor(224, 255, 224).name():
                        highlighted_cells.append((row, col))
            
            # Process any highlighted cells to ensure their data is in the model
            for row, col in highlighted_cells:
                item = self.item(row, col)
                if item and item.text():
                    # Check if this data is already in the model
                    id_item = self.item(row, 1)
                    if id_item and id_item.text().isdigit():
                        exp_id = int(id_item.text()) - 1
                        if exp_id >= 0 and exp_id < len(self.model.planned_experiments):
                            # Get the corresponding objective
                            col_idx = col - self.objective_columns_start
                            if col_idx < 0 or col_idx >= len(self.model.objectives):
                                continue
                                
                            objective_name = self.model.objectives[col_idx]
                            
                            # Check if we need to update the model
                            text_value = item.text().strip()
                            if text_value and '%' in text_value:
                                # Already has a percentage format - should be in the model
                                continue
                                
                            # Trigger a cell change event to update the model
                            item.setText(item.text())  # Force refresh
            
            # Clear any temporary highlights
            self.clearSelection()
            
            # Return True to indicate success
            return True
            
        except Exception as e:
            import traceback
            print(f"Error committing pending changes: {e}")
            print(traceback.format_exc())
            return False

class BestResultsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
    def update_columns(self, model):
        # Store reference to model
        self.model = model
        
        columns = ["Rank"] + list(model.parameters.keys()) + model.objectives
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, len(columns)):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
    def update_from_model(self, model, n_best=5):
        from ..core import settings
        
        self.setRowCount(0)
        
        if not model.experiments:
            return
            
        best_exps = model.get_best_experiments(n=n_best)
        
        for i, exp_data in enumerate(best_exps):
            row = self.rowCount()
            self.insertRow(row)
            
            self.setItem(row, 0, QTableWidgetItem(str(i+1)))
            
            col = 1
            exp_params = exp_data.get('params', {})
            for param_name in model.parameters.keys():
                if param_name in exp_params:
                    value = exp_params[param_name]
                    if isinstance(value, float):
                        value_str = settings.format_value(value)
                    else:
                        value_str = str(value)
                    self.setItem(row, col, QTableWidgetItem(value_str))
                else:
                    self.setItem(row, col, QTableWidgetItem(""))
                col += 1
                
            exp_results = exp_data.get('results', {})
            for obj in model.objectives:
                if obj in exp_results and exp_results[obj] is not None:
                    value = exp_results[obj] * 100.0
                    self.setItem(row, col, QTableWidgetItem(f"{settings.format_value(value)}%"))
                else:
                    self.setItem(row, col, QTableWidgetItem(""))
                col += 1

class AllResultsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
    def update_from_model(self, model):
        from ..core import settings
        
        self.setRowCount(0)
        
        if not model.experiments:
            return
            
        columns = ["Round", "ID", "Date"] + model.objectives
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        for i, exp_data in enumerate(model.experiments):
            row = self.rowCount()
            self.insertRow(row)
            
            exp_round = i // 5 + 1
            
            if "timestamp" in exp_data:
                date_str = exp_data["timestamp"].split("T")[0]
            else:
                date_str = "-"
                
            self.setItem(row, 0, QTableWidgetItem(str(exp_round)))
            self.setItem(row, 1, QTableWidgetItem(str(i+1)))
            self.setItem(row, 2, QTableWidgetItem(date_str))
            
            exp_results = exp_data.get('results', {})
            for col, obj in enumerate(model.objectives, 3):
                if obj in exp_results and exp_results[obj] is not None:
                    value = exp_results[obj] * 100.0
                    self.setItem(row, col, QTableWidgetItem(f"{settings.format_value(value)}%"))
                else:
                    self.setItem(row, col, QTableWidgetItem("-"))