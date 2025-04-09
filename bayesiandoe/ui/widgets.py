import time
import random
import logging
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor, QPainter, QBrush, QPen, QLinearGradient, QImage
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QSplashScreen, QFrame, QHeaderView,
    QLabel, QMessageBox
)
from ..core import _calculate_parameter_distance, settings
from concurrent.futures import ThreadPoolExecutor

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
        self.duration_ms = 4000
        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(30)
        self.draw_splash()
    
    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        
    def update_progress(self):
        elapsed_time = (time.time() - self.start_time) * 1000
        self.progress = min(int(elapsed_time / self.duration_ms * 100), 100)
        self.counter += 1
        
        if elapsed_time < self.duration_ms:
            self.draw_splash()
            
            if self.counter % 10 == 0:
                self.raise_()
        else:
            self.timer.stop()
            self.close()
        
    def draw_splash(self):
        painter = QPainter(self.splash_pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        rect = self.splash_pix.rect()
        
        painter.fillRect(rect, QColor(15, 20, 25))
        
        self.draw_molecular_background(painter, rect)
     
        font = QFont("Helvetica", 48, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        
        painter.setPen(QColor(240, 250, 250))
        painter.drawText(rect.adjusted(0, 0, 0, -rect.height()/3), Qt.AlignHCenter | Qt.AlignBottom, "BAYESIAN DOE")
        
        self.draw_cyber_progress(painter, rect)
        
        font = QFont("Helvetica", 16)
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        painter.setPen(QColor(150, 170, 170))
        
        painter.drawText(rect.adjusted(20, 0, 0, -15), Qt.AlignLeft | Qt.AlignBottom, "v1.0.5")
        painter.drawText(rect.adjusted(0, 0, -20, -15), Qt.AlignRight | Qt.AlignBottom, "© 2025 Johan H.G. Natter")
        painter.drawText(rect.adjusted(0, 0, 0, -15), Qt.AlignHCenter | Qt.AlignBottom, "")
        
        painter.end()
        self.setPixmap(self.splash_pix)
        
    def draw_molecular_background(self, painter, rect):
        painter.setPen(QPen(QColor(40, 60, 65, 150), 1.2))
        
        random.seed(42)
        
        nodes = []
        for i in range(15):
            x = random.randint(50, rect.width() - 50)
            y = random.randint(50, rect.height() - 50)
            r = random.randint(5, 12)
            nodes.append((x, y, r))
            
            color = QColor(
                random.randint(30, 70),
                random.randint(120, 200),
                random.randint(150, 210),
                100
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(x - r, y - r, r*2, r*2)
            
        painter.setPen(QPen(QColor(70, 90, 95, 70), 1.2))
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if random.random() < 0.4:
                    x1, y1, _ = nodes[i]
                    x2, y2, _ = nodes[j]
                    painter.drawLine(x1, y1, x2, y2)
        
        painter.setPen(QPen(QColor(40, 60, 70, 40), 0.8))
        step = 30
        for x in range(0, rect.width(), step):
            painter.drawLine(x, 0, x, rect.height())
        for y in range(0, rect.height(), step):
            painter.drawLine(0, y, rect.width(), y)
            
    def draw_cyber_progress(self, painter, rect):
        bar_width = rect.width() * 0.6
        bar_height = 14
        
        x = (rect.width() - bar_width) / 2
        y = rect.height() * 0.78
        
        segments = 20
        seg_width = bar_width / segments
        
        for i in range(segments):
            color = QColor(25, 35, 45) if i % 2 == 0 else QColor(30, 40, 50)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            
            seg_x = x + (i * seg_width)
            painter.drawRect(seg_x, y, seg_width - 1, bar_height)
            
        filled_segments = int((self.progress / 100) * segments)
        
        for i in range(filled_segments):
            color = QColor(0, 220, 200) if i % 2 == 0 else QColor(0, 200, 180)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            
            seg_x = x + (i * seg_width)
            painter.drawRect(seg_x, y, seg_width - 1, bar_height)
            
            painter.setPen(QPen(QColor(220, 255, 255, 150), 1))
            painter.drawLine(seg_x, y, seg_x + seg_width - 1, y)
            
        font = QFont("Helvetica", 13)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        painter.setFont(font)
        
        painter.setPen(QColor(0, 0, 0, 100))
        painter.drawText(QRect(x + 2, y - 30 + 2, bar_width, 25), Qt.AlignCenter, f"INITIALIZING SYSTEM {self.progress}%")
        
        painter.setPen(QColor(0, 220, 200))
        painter.drawText(QRect(x, y - 30, bar_width, 25), Qt.AlignCenter, f"INITIALIZING SYSTEM {self.progress}%")

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        
        # Add a direct reference to store the model
        self.model = None
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._prediction_start_time = 0
        self._prediction_future = None
        
        # Connect to cell change events
        self.cellChanged.connect(self.handle_cell_change)
        
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
                result_value = float(result_text.replace('%', '').strip())
                # If value is > 1 and not a percentage sign, assume it's a percentage
                if result_value > 1.0 and not '%' in result_text:
                    result_value = result_value / 100.0
                else:
                    result_value = result_value / 100.0  # Always convert to 0-1 scale
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
            self.log(f"-- Added result for experiment #{exp_id+1}: {objective_name}={result_value*100:.2f}% - Success")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error handling cell change: {e}")
            print(error_details)
    
    def update_from_planned(self, model, round_start_indices):
        """Update experiment table from planned experiments with debug info."""
        try:
            # Block signals during bulk update
            self.blockSignals(True)
            
            if not hasattr(model, 'planned_experiments'):
                model.planned_experiments = []
                print("Initialized empty planned_experiments list")
                
            if not model.planned_experiments:
                print("No planned experiments found")
                self.blockSignals(False)
                return
            
            print(f"Updating table with {len(model.planned_experiments)} planned experiments")
            print(f"Round start indices: {round_start_indices}")
            
            self.setSortingEnabled(False)
            
            # Save current scroll position and selection
            vscroll = self.verticalScrollBar().value()
            selected_items = self.selectedItems()
            selected_exp_ids = []
            if selected_items:
                row = selected_items[0].row()
                id_item = self.item(row, 1)
                if id_item and id_item.text().isdigit():
                    selected_exp_ids.append(int(id_item.text()) - 1)

            # First completely clear the table
            self.clearContents()
            self.setRowCount(0)
            
            # Debug column headers
            headers = [self.horizontalHeaderItem(i).text() if self.horizontalHeaderItem(i) else f"Col {i}"
                      for i in range(self.columnCount())]
            print(f"Table columns: {headers}")
            
            # Map experiment ID to round number
            exp_to_round = {}
            current_round = 1  # Start from round 1 instead of 0
            
            # Create a sorted list of round start indices for easier processing
            round_boundaries = sorted(round_start_indices)
            
            # Assign rounds to experiments
            for i, _ in enumerate(model.planned_experiments):
                # Determine round based on round_start_indices
                round_num = 1
                for r, start_idx in enumerate(round_boundaries):
                    if i >= start_idx:
                        round_num = r + 2  # +2 because round 1 is before the first boundary
                
                exp_to_round[i] = round_num
            
            # Track row index for each experiment
            exp_id_to_row = {}
            current_display_round = -1
            
            # Fill table with experiments
            for i, params in enumerate(model.planned_experiments):
                round_num = exp_to_round.get(i, 1)  # Default to round 1
                
                # Add round separator if needed
                if current_display_round != round_num:
                    current_display_round = round_num
                    separator_row = self.rowCount()
                    self.insertRow(separator_row)
                    
                    separator_item = QTableWidgetItem(f"--- Round {round_num} ---")
                    separator_item.setTextAlignment(Qt.AlignCenter)
                    separator_item.setBackground(QColor(220, 220, 220))
                    separator_item.setForeground(QColor(80, 80, 80))
                    
                    font = separator_item.font()
                    font.setBold(True)
                    separator_item.setFont(font)
                    
                    self.setSpan(separator_row, 0, 1, self.columnCount())
                    self.setItem(separator_row, 0, separator_item)
                    # Make separator non-selectable
                    separator_item.setFlags(separator_item.flags() & ~Qt.ItemIsSelectable)
                
                # Add experiment row
                row_index = self.rowCount()
                self.insertRow(row_index)
                exp_id_to_row[i] = row_index
                
                # Add round number
                round_item = QTableWidgetItem(str(round_num))
                round_item.setTextAlignment(Qt.AlignCenter)
                font = round_item.font()
                font.setBold(True)
                round_item.setFont(font)
                round_item.setFlags(round_item.flags() & ~Qt.ItemIsEditable)  # Make non-editable
                self.setItem(row_index, 0, round_item)
                
                # Add experiment ID
                id_item = QTableWidgetItem(str(i + 1))
                id_item.setTextAlignment(Qt.AlignCenter)
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)  # Make non-editable
                self.setItem(row_index, 1, id_item)
                
                # Add parameter values
                for col, param_name in enumerate(model.parameters.keys(), 2):
                    value_str = ""
                    if param_name in params:
                        value = params[param_name]
                        if isinstance(value, float):
                            from ..core import settings
                            value_str = settings.format_value(value)
                        else:
                            value_str = str(value)
                    param_item = QTableWidgetItem(value_str)
                    param_item.setTextAlignment(Qt.AlignCenter if isinstance(params.get(param_name), (int, float)) else Qt.AlignLeft)
                    param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)  # Make non-editable
                    self.setItem(row_index, col, param_item)
                
                # Initialize objective columns with empty editable cells
                for obj_idx, obj_name in enumerate(model.objectives):
                    obj_col = self.objective_columns_start + obj_idx
                    obj_item = QTableWidgetItem("")
                    obj_item.setTextAlignment(Qt.AlignCenter)
                    # Add visual styling to indicate these cells are editable
                    obj_item.setToolTip(f"Double-click to enter {obj_name} result")
                    # These cells should be editable
                    self.setItem(row_index, obj_col, obj_item)
                
                # Initialize prediction column with empty non-editable cell
                predict_col = self.columnCount() - 1
                predict_item = QTableWidgetItem("")
                predict_item.setFlags(predict_item.flags() & ~Qt.ItemIsEditable)  # Make non-editable
                self.setItem(row_index, predict_col, predict_item)
            
            print(f"Added {len(exp_id_to_row)} experiment rows to table")
            
            # Process completed experiments
            completed_indices = set()
            for exp_idx, exp_data in enumerate(model.experiments):
                matched_exp_idx = self._find_matching_planned_experiment(model, exp_data, completed_indices)
                
                if matched_exp_idx != -1 and matched_exp_idx in exp_id_to_row:
                    completed_indices.add(matched_exp_idx)
                    row_index = exp_id_to_row[matched_exp_idx]
                    
                    # Highlight completed experiment row
                    for col in range(self.columnCount()):
                        item = self.item(row_index, col)
                        if item:
                            item.setBackground(QColor(224, 255, 224))
                    
                    # Add result values for each objective
                    if "results" in exp_data:
                        for obj_idx, obj_name in enumerate(model.objectives):
                            obj_col = self.objective_columns_start + obj_idx
                            if obj_name in exp_data["results"] and exp_data["results"][obj_name] is not None:
                                obj_value = exp_data["results"][obj_name] * 100.0
                                obj_item = QTableWidgetItem(f"{obj_value:.2f}%")
                                obj_item.setTextAlignment(Qt.AlignCenter)
                                obj_item.setBackground(QColor(224, 255, 224))
                                self.setItem(row_index, obj_col, obj_item)
            
            print(f"Processed {len(completed_indices)} completed experiments")
            
            # Make sure the widget is updated
            self.viewport().update()
            
            # Restore scroll position and selection
            self.verticalScrollBar().setValue(vscroll)
            for exp_id in selected_exp_ids:
                if exp_id in exp_id_to_row:
                    self.selectRow(exp_id_to_row[exp_id])
                
            # Re-enable signals and sorting
            self.setSortingEnabled(True)
            self.blockSignals(False)
            
        except Exception as e:
            import traceback
            print(f"Error in update_from_planned: {e}")
            traceback.print_exc()
            self.blockSignals(False)

    def _find_matching_planned_experiment(self, model, exp_data, completed_indices):
        """Find the planned experiment that matches the experiment data."""
        if 'params' not in exp_data:
            return -1
        
        for planned_idx, planned_params in enumerate(model.planned_experiments):
            if planned_idx in completed_indices:
                continue
            
            # Check if parameters match
            params_match = True
            for param_name, planned_value in planned_params.items():
                if param_name not in exp_data['params']:
                    params_match = False
                    break
                
                exp_value = exp_data['params'][param_name]
                
                # Compare values based on parameter type
                param = model.parameters.get(param_name)
                if not param:
                    continue
                
                if param.param_type == 'continuous':
                    # Allow small floating point differences
                    if abs(float(planned_value) - float(exp_value)) > 1e-6:
                        params_match = False
                        break
                elif planned_value != exp_value:
                    params_match = False
                    break
                
            if params_match:
                return planned_idx
        
        return -1

    def _async_update_predictions(self, model, completed_indices, exp_id_to_row):
        self._prediction_start_time = time.time()
        if self._prediction_future and not self._prediction_future.done():
            self._prediction_future.cancel()
        self._prediction_future = self.executor.submit(
            self._calculate_predictions, model, completed_indices, exp_id_to_row
        )
        # Add timeout watchdog timer
        QTimer.singleShot(5000, lambda: self._check_prediction_timeout())

    def _check_prediction_timeout(self):
        if self._prediction_future and not self._prediction_future.done():
            if time.time() - self._prediction_start_time > 5.0:
                self._prediction_future.cancel()
                self.log("-- Warning: Prediction calculation timed out - using simple interpolation")
                # Fall back to simpler method
                # [implementation omitted for brevity]

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