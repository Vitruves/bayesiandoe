from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QComboBox, QGroupBox, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

class DesignMethodSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Design Method Selector")
        self.resize(800, 500)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Project details input section
        details_group = QGroupBox("Experiment Details")
        details_layout = QFormLayout(details_group)
        
        # Parameter types
        self.param_type_combo = QComboBox()
        self.param_type_combo.addItems([
            "Mostly continuous", 
            "Mostly categorical",
            "Mixed continuous and categorical",
            "Mostly discrete"
        ])
        details_layout.addRow("Parameter types:", self.param_type_combo)
        
        # Number of parameters
        self.param_count_combo = QComboBox()
        self.param_count_combo.addItems([
            "Few (1-3)",
            "Medium (4-8)",
            "Many (9+)"
        ])
        details_layout.addRow("Number of parameters:", self.param_count_combo)
        
        # Prior knowledge
        self.prior_combo = QComboBox()
        self.prior_combo.addItems([
            "No prior knowledge",
            "Some prior knowledge",
            "Strong prior knowledge"
        ])
        details_layout.addRow("Prior knowledge:", self.prior_combo)
        
        # Expected response shape
        self.response_combo = QComboBox()
        self.response_combo.addItems([
            "Simple (smooth, unimodal)",
            "Complex (multimodal, interactions)",
            "Unknown"
        ])
        details_layout.addRow("Expected response surface:", self.response_combo)
        
        # Objectives
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "Single objective",
            "Multiple objectives"
        ])
        details_layout.addRow("Optimization objectives:", self.objective_combo)
        
        # Computational resources
        self.compute_combo = QComboBox()
        self.compute_combo.addItems([
            "Limited (prefer fast methods)",
            "Adequate (balance speed/accuracy)",
            "High (accuracy is priority)"
        ])
        details_layout.addRow("Computational resources:", self.compute_combo)
        
        # Sample budget
        self.budget_combo = QComboBox()
        self.budget_combo.addItems([
            "Very limited (<10 experiments)",
            "Limited (10-20 experiments)",
            "Moderate (20-50 experiments)",
            "Large (50+ experiments)"
        ])
        details_layout.addRow("Experiment budget:", self.budget_combo)
        
        layout.addWidget(details_group)
        
        # Analysis button
        analyze_btn = QPushButton("Analyze and Recommend")
        analyze_btn.clicked.connect(self.analyze_and_recommend)
        layout.addWidget(analyze_btn)
        
        # Results table
        self.results_table = QTableWidget(8, 6)
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Suitability", "Speed", "Accuracy", "Handles Constraints", "Comments"
        ])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        
        # Initialize table with methods
        methods = ["BoTorch", "TPE", "GPEI", "CMA-ES", "NSGA-II", "Sobol", "Latin Hypercube", "Random"]
        for i, method in enumerate(methods):
            self.results_table.setItem(i, 0, QTableWidgetItem(method))
            # Initialize other cells
            for j in range(1, 6):
                self.results_table.setItem(i, j, QTableWidgetItem(""))
                
        layout.addWidget(self.results_table)
        
        # Button box for OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def analyze_and_recommend(self):
        """Analyze inputs and recommend design methods"""
        # Get inputs
        param_type = self.param_type_combo.currentText()
        param_count = self.param_count_combo.currentText()
        prior = self.prior_combo.currentText()
        response = self.response_combo.currentText()
        objective = self.objective_combo.currentText()
        compute = self.compute_combo.currentText()
        budget = self.budget_combo.currentText()
        
        # Clear previous results
        for i in range(self.results_table.rowCount()):
            for j in range(1, self.results_table.columnCount()):
                self.results_table.item(i, j).setText("")
                
        # Rate each method
        ratings = {
            "BoTorch": {
                "Suitability": 5, "Speed": 3, "Accuracy": 5, 
                "Constraints": "Yes", "Comments": "Best for continuous parameters and chemistry optimization"
            },
            "TPE": {
                "Suitability": 3, "Speed": 4, "Accuracy": 3, 
                "Constraints": "Yes", "Comments": "Good all-around choice"
            },
            "GPEI": {
                "Suitability": 3, "Speed": 2, "Accuracy": 5, 
                "Constraints": "Yes", "Comments": "Best for continuous parameters"
            },
            "CMA-ES": {
                "Suitability": 3, "Speed": 3, "Accuracy": 4, 
                "Constraints": "Yes", "Comments": "Good for complex landscapes"
            },
            "NSGA-II": {
                "Suitability": 2, "Speed": 2, "Accuracy": 4, 
                "Constraints": "Yes", "Comments": "Specialized for multi-objective"
            },
            "Sobol": {
                "Suitability": 3, "Speed": 5, "Accuracy": 3, 
                "Constraints": "No", "Comments": "Best for initial screening"
            },
            "Latin Hypercube": {
                "Suitability": 3, "Speed": 5, "Accuracy": 3, 
                "Constraints": "No", "Comments": "Good space coverage"
            },
            "Random": {
                "Suitability": 1, "Speed": 5, "Accuracy": 1, 
                "Constraints": "No", "Comments": "Baseline comparison only"
            }
        }
        
        # Adjust BoTorch rating for chemistry applications
        ratings["BoTorch"]["Comments"] = "State-of-the-art for chemistry optimization; uses GP with priors"
        
        # Adjust ratings based on inputs
        if param_type == "Mostly categorical":
            ratings["TPE"]["Suitability"] += 1
            ratings["GPEI"]["Suitability"] -= 1
            ratings["GPEI"]["Comments"] = "Not ideal for categorical parameters"
            ratings["BoTorch"]["Suitability"] -= 1
            
        if param_type == "Mostly continuous":
            ratings["GPEI"]["Suitability"] += 1
            ratings["BoTorch"]["Suitability"] += 1
            ratings["BoTorch"]["Comments"] = "Excellent for continuous chemistry parameters with priors"
            
        if param_count == "Many (9+)":
            ratings["GPEI"]["Suitability"] -= 1
            ratings["GPEI"]["Speed"] -= 1
            ratings["GPEI"]["Comments"] = "Computational cost increases with dimensions"
            ratings["Sobol"]["Suitability"] += 1
            ratings["BoTorch"]["Speed"] -= 1
            
        if prior == "Strong prior knowledge":
            ratings["TPE"]["Suitability"] += 1
            ratings["GPEI"]["Suitability"] += 1
            ratings["BoTorch"]["Suitability"] += 2
            ratings["BoTorch"]["Comments"] = "Excellent integration of chemistry priors for better optimization"
            ratings["Random"]["Suitability"] -= 1
            
        if response == "Complex (multimodal, interactions)":
            ratings["CMA-ES"]["Suitability"] += 1
            ratings["TPE"]["Suitability"] += 1
            ratings["BoTorch"]["Suitability"] += 1
            ratings["BoTorch"]["Comments"] = "Advanced GP models handle complex response surfaces well"
            
        if objective == "Multiple objectives":
            ratings["NSGA-II"]["Suitability"] += 2
            ratings["NSGA-II"]["Comments"] = "Specifically designed for multi-objective"
            ratings["BoTorch"]["Suitability"] += 1
            ratings["BoTorch"]["Comments"] = "Supports multi-objective optimization with known preferences"
            
        if compute == "Limited (prefer fast methods)":
            ratings["GPEI"]["Suitability"] -= 1
            ratings["BoTorch"]["Suitability"] -= 1
            ratings["Sobol"]["Suitability"] += 1
            ratings["Latin Hypercube"]["Suitability"] += 1
            
        if budget == "Very limited (<10 experiments)":
            ratings["Sobol"]["Suitability"] += 1
            ratings["Latin Hypercube"]["Suitability"] += 1
            ratings["Random"]["Suitability"] -= 1
            
        if budget == "Large (50+ experiments)":
            ratings["GPEI"]["Suitability"] += 1
            ratings["TPE"]["Suitability"] += 1
            ratings["BoTorch"]["Suitability"] += 1
            ratings["BoTorch"]["Comments"] = "Scales well with more experiments, learning better models"
            
        # Update table with calculated ratings
        for i, method in enumerate(["BoTorch", "TPE", "GPEI", "CMA-ES", "NSGA-II", "Sobol", "Latin Hypercube", "Random"]):
            r = ratings[method]
            # Suitability (1-5 stars)
            stars = "★" * r["Suitability"] + "☆" * (5 - r["Suitability"])
            self.results_table.item(i, 1).setText(stars)
            
            # Speed (1-5 stars)
            speed_stars = "★" * r["Speed"] + "☆" * (5 - r["Speed"])
            self.results_table.item(i, 2).setText(speed_stars)
            
            # Accuracy (1-5 stars)
            acc_stars = "★" * r["Accuracy"] + "☆" * (5 - r["Accuracy"])
            self.results_table.item(i, 3).setText(acc_stars)
            
            # Constraints
            self.results_table.item(i, 4).setText(r["Constraints"])
            
            # Comments
            self.results_table.item(i, 5).setText(r["Comments"])
            
        # Highlight recommended method
        best_method = max(ratings.items(), key=lambda x: x[1]["Suitability"])
        for i in range(self.results_table.rowCount()):
            if self.results_table.item(i, 0).text() == best_method[0]:
                for j in range(self.results_table.columnCount()):
                    self.results_table.item(i, j).setBackground(QColor(230, 255, 230)) 