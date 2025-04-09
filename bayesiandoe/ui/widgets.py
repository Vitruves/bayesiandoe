import time
import random
import logging
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor, QPainter, QBrush, QPen, QLinearGradient, QImage
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QSplashScreen, QFrame, QHeaderView,
    QLabel
)
from ..core import _calculate_parameter_distance, settings

class LogDisplay:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger('BayesianDOE')
        
    def log(self, message):
        self.logger.info(message)

class SplashScreen(QSplashScreen):
    def __init__(self):
        self.splash_pix = QPixmap(800, 600)
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
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        
    def update_columns(self, model):
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
        
        # Set up prediction column
        predict_col = len(columns) - 1
        self.horizontalHeader().setSectionResizeMode(predict_col, QHeaderView.ResizeToContents)
        
    def update_from_planned(self, model, round_start_indices):
        # Make sure we preserve existing functionality but improve it
        # Add additional debugging code if necessary
        try:
            if not hasattr(model, 'planned_experiments') or not model.planned_experiments:
                return
            
            vscroll = self.verticalScrollBar().value()
            
            selected_exp_id = -1
            if self.selectedItems():
                selected_row_index = self.selectedItems()[0].row()
                id_item = self.item(selected_row_index, 1)
                if id_item and id_item.text().isdigit():
                    try:
                        selected_exp_id = int(id_item.text()) - 1
                    except ValueError:
                        pass

            self.setRowCount(0)
            current_round_num = 0
            
            exp_id_to_row = {}
            
            exp_to_round = {}
            next_round_idx = 0
            for i in range(len(model.planned_experiments)):
                if next_round_idx < len(round_start_indices) and i >= round_start_indices[next_round_idx]:
                    current_round_num += 1
                    next_round_idx += 1
                exp_to_round[i] = current_round_num
                
            current_display_round = 0
            for i, params in enumerate(model.planned_experiments):
                round_num = exp_to_round.get(i, 1)
                
                if current_display_round != round_num:
                    current_display_round = round_num
                    if self.rowCount() > 0:
                        separator_row_index = self.rowCount()
                        self.insertRow(separator_row_index)
                        separator_item = QTableWidgetItem(f"--- Round {round_num} ---")
                        separator_item.setTextAlignment(Qt.AlignCenter)
                        separator_item.setBackground(QColor(220, 220, 220))
                        separator_item.setForeground(QColor(80, 80, 80))
                        font = separator_item.font()
                        font.setBold(True)
                        separator_item.setFont(font)
                        self.setSpan(separator_row_index, 0, 1, self.columnCount())
                        self.setItem(separator_row_index, 0, separator_item)
                        separator_item.setFlags(separator_item.flags() & ~Qt.ItemIsSelectable)

                row_index = self.rowCount()
                self.insertRow(row_index)
                exp_id_to_row[i] = row_index
                
                round_item = QTableWidgetItem(str(round_num))
                round_item.setTextAlignment(Qt.AlignCenter)
                font = round_item.font()
                font.setBold(True)
                round_item.setFont(font)
                self.setItem(row_index, 0, round_item)
                
                id_item = QTableWidgetItem(str(i + 1))
                id_item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row_index, 1, id_item)
                
                # Set parameter values
                for col, param_name in enumerate(model.parameters.keys(), 2):
                    value_str = ""
                    if param_name in params:
                        value = params[param_name]
                        if isinstance(value, float):
                            value_str = settings.format_value(value)
                        else:
                            value_str = str(value)
                    param_item = QTableWidgetItem(value_str)
                    param_item.setTextAlignment(Qt.AlignCenter if isinstance(params.get(param_name), (int, float)) else Qt.AlignLeft)
                    self.setItem(row_index, col, param_item)
                
                # Initialize objective and prediction columns with empty cells
                for col_idx in range(2 + len(model.parameters), self.columnCount()):
                    self.setItem(row_index, col_idx, QTableWidgetItem(""))
            
            completed_indices = set()
            for completed_exp in model.experiments:
                found_match_idx = -1
                for planned_idx, planned_params in enumerate(model.planned_experiments):
                    if planned_idx in completed_indices:
                        continue
                        
                    matches_all_params = True
                    if set(planned_params.keys()) != set(k for k in completed_exp['params'] if k in model.parameters):
                         matches_all_params = False
                    else:
                        for p_name, p_value in planned_params.items():
                            if p_name not in completed_exp['params']:
                                matches_all_params = False
                                break
                                
                            comp_value = completed_exp['params'][p_name]
                            
                            param_type = model.parameters.get(p_name).param_type if p_name in model.parameters else "unknown"
                            if param_type == "continuous":
                                if abs(float(p_value) - float(comp_value)) > 1e-5:
                                    matches_all_params = False
                                    break
                            elif p_value != comp_value:
                                matches_all_params = False
                                break
                    
                    if matches_all_params:
                        found_match_idx = planned_idx
                        break

                if found_match_idx != -1 and found_match_idx in exp_id_to_row:
                    completed_indices.add(found_match_idx)
                    row_index = exp_id_to_row[found_match_idx]
                    
                    # Highlight the completed experiment row
                    for col in range(self.columnCount()):
                        item = self.item(row_index, col)
                        if item:
                            item.setBackground(QColor(224, 255, 224))
                    
                    # Fill in results for each objective
                    if "results" in completed_exp:
                        objective_base_col = 2 + len(model.parameters)
                        for obj_idx, obj_name in enumerate(model.objectives):
                            if obj_name in completed_exp["results"] and completed_exp["results"][obj_name] is not None:
                                obj_col_idx = objective_base_col + obj_idx
                                obj_value = completed_exp["results"][obj_name] * 100.0
                                obj_item = QTableWidgetItem(f"{obj_value:.2f}%")
                                obj_item.setTextAlignment(Qt.AlignCenter)
                                obj_item.setBackground(QColor(224, 255, 224))
                                self.setItem(row_index, obj_col_idx, obj_item)
                        
                        # Clear prediction since we have actual results
                        predict_col_idx = self.columnCount() - 1
                        pred_item = self.item(row_index, predict_col_idx)
                        if pred_item:
                             pred_item.setText("")

            # Generate predictions for incomplete experiments using BoTorch model
            if len(model.experiments) >= 3:
                try:
                    # Initialize BoTorch model for predictions
                    import torch
                    import gpytorch
                    from botorch.models import SingleTaskGP
                    from gpytorch.mlls import ExactMarginalLogLikelihood
                    from botorch.fit import fit_gpytorch_model
                    
                    # Extract training data
                    train_X, train_Y = model._extract_normalized_features_and_targets()
                    train_X = torch.tensor(train_X, dtype=torch.float64)
                    train_Y = torch.tensor(train_Y, dtype=torch.float64).reshape(-1, 1)
                    
                    # Normalize Y for numerical stability
                    Y_mean = train_Y.mean()
                    Y_std = train_Y.std()
                    if Y_std < 1e-6:
                        Y_std = torch.tensor(1.0)
                    train_Y_normalized = (train_Y - Y_mean) / Y_std
                    
                    # Build the GP model
                    gp = SingleTaskGP(train_X, train_Y_normalized)
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    
                    # Fit model
                    try:
                        fit_gpytorch_model(mll)
                    except Exception as e:
                        print(f"Warning: GP model fitting failed: {e}")
                    
                    # Set to evaluation mode
                    gp.eval()
                    
                    # Get predictions for all planned but incomplete experiments
                    for i, params in enumerate(model.planned_experiments):
                        if i not in completed_indices and i in exp_id_to_row:
                            row_index = exp_id_to_row[i]
                            
                            # Normalize parameters
                            test_X = torch.tensor([model._normalize_params(params)], dtype=torch.float64)
                            
                            # Predict
                            with torch.no_grad():
                                posterior = gp.posterior(test_X)
                                mean = posterior.mean.cpu()
                                std = posterior.variance.sqrt().cpu()
                                
                            # Denormalize prediction
                            pred_mean = mean * Y_std + Y_mean
                            
                            # Convert to percentage for display
                            pred_value = pred_mean.item() * 100.0
                            uncertainty = std.item() * Y_std.item() * 100.0
                            
                            # Set prediction in table with confidence interval
                            pred_col_idx = self.columnCount() - 1
                            pred_text = f"{settings.format_value(pred_value)}%±{settings.format_value(uncertainty)}%"
                            pred_item = QTableWidgetItem(pred_text)
                            
                            # Color code by prediction confidence
                            confidence_level = 1.0 - min(uncertainty / 20.0, 0.8)  # Max 80% fading
                            color_intensity = int(220 * confidence_level)
                            pred_item.setBackground(QColor(220, color_intensity, 255))
                            pred_item.setForeground(QColor(0, 0, 150))
                            pred_item.setTextAlignment(Qt.AlignCenter)
                            self.setItem(row_index, pred_col_idx, pred_item)
                            
                except Exception as e:
                    print(f"BoTorch prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to simpler prediction method
                    self._simple_distance_predictions(model, completed_indices, exp_id_to_row)
            else:
                # Not enough data for GP, use simpler method
                self._simple_distance_predictions(model, completed_indices, exp_id_to_row)

            self.verticalScrollBar().setValue(vscroll)
            
            if selected_exp_id >= 0 and selected_exp_id in exp_id_to_row:
                 self.selectRow(exp_id_to_row[selected_exp_id])
        except Exception as e:
            print(f"Error in update_from_planned: {e}")
            import traceback
            traceback.print_exc()

    def _simple_distance_predictions(self, model, completed_indices, exp_id_to_row):
        """Fallback prediction method using simple distance-based neighbor averaging"""
        import numpy as np
        
        for i, planned_params in enumerate(model.planned_experiments):
            if i in completed_indices or i not in exp_id_to_row:
                continue
            
            row_index = exp_id_to_row[i]
            predict_col_idx = self.columnCount() - 1
            
            # Only predict if we have the first objective (usually yield)
            if model.objectives and model.objectives[0] in model.objective_weights:
                try:
                    k = 5  # Number of nearest neighbors to consider
                    distances = []
                    target_obj = model.objectives[0]  # First objective to predict
                    
                    for completed_exp in model.experiments:
                        if "results" in completed_exp and target_obj in completed_exp['results'] and completed_exp['results'][target_obj] is not None:
                            dist = _calculate_parameter_distance(planned_params, completed_exp['params'], model.parameters)
                            distances.append((dist, completed_exp['results'][target_obj]))

                    if distances:
                        distances.sort(key=lambda x: x[0])
                        # Weight by inverse distance
                        total_weight = 0
                        weighted_sum = 0
                        for dist, value in distances[:k]:
                            if dist < 0.001:  # Avoid division by zero
                                weight = 1000
                            else:
                                weight = 1.0 / (dist * 10)
                            weighted_sum += weight * value
                            total_weight += weight
                        
                        if total_weight > 0:
                            pred_value = (weighted_sum / total_weight) * 100.0
                            pred_item = QTableWidgetItem(f"{settings.format_value(pred_value)}%?")
                            pred_item.setForeground(QColor(100, 100, 150))
                            pred_item.setTextAlignment(Qt.AlignCenter)
                            self.setItem(row_index, predict_col_idx, pred_item)
                        else:
                            # No meaningful weights, use simple average
                            neighbor_values = [y for _, y in distances[:k]]
                            if neighbor_values:
                                pred_value = np.mean(neighbor_values) * 100.0
                                pred_item = QTableWidgetItem(f"{settings.format_value(pred_value)}%?")
                                pred_item.setForeground(QColor(100, 100, 150))
                                pred_item.setTextAlignment(Qt.AlignCenter)
                                self.setItem(row_index, predict_col_idx, pred_item)
                except Exception as e:
                    print(f"Simple prediction error for exp {i+1}: {e}")
                    pred_item = self.item(row_index, predict_col_idx)
                    if pred_item:
                        pred_item.setText("")

class BestResultsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
    def update_columns(self, model):
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