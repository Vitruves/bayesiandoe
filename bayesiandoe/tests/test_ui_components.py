# scripts/test_ui_components.py
#!/usr/bin/env python3
"""
Interactive test script for UI components.
Logs visual verification steps for UI elements.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_ui_components():
    """Run interactive tests for UI components."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    import sys
    
    if not QApplication.instance():
        app = QApplication(sys.argv)
    
    print("-- Testing UI components")
    
    # Test canvas creation
    try:
        from bayesiandoe.ui.canvas import MplCanvas, Mpl3DCanvas
        from matplotlib.figure import Figure
        
        # 2D Canvas test
        canvas = MplCanvas()
        assert canvas.axes is not None
        assert isinstance(canvas.fig, Figure)
        print("-- 2D Canvas creation successful")
        
        # Draw something on the canvas
        canvas.axes.plot([0, 1, 2], [0, 1, 0])
        canvas.draw()
        print("-- 2D Canvas rendering successful")
        
        # 3D Canvas test
        canvas3d = Mpl3DCanvas()
        assert canvas3d.axes is not None
        assert canvas3d.axes.name == "3d"
        print("-- 3D Canvas creation successful")
        
        # Draw on 3D canvas
        import numpy as np
        x = np.linspace(-5, 5, 25)
        y = np.linspace(-5, 5, 25)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        canvas3d.axes.plot_surface(X, Y, Z, cmap='viridis')
        canvas3d.draw()
        print("-- 3D Canvas rendering successful")
    except Exception as e:
        print(f"-- ERROR: Canvas tests failed: {str(e)}")
    
    # Test experiment table
    try:
        from bayesiandoe.ui.widgets import ExperimentTable
        from bayesiandoe.core import OptunaBayesianExperiment
        from bayesiandoe.parameters import ChemicalParameter
        
        # Create a model
        model = OptunaBayesianExperiment()
        model.add_parameter(ChemicalParameter("temp", "continuous", 0, 100))
        model.add_parameter(ChemicalParameter("time", "continuous", 0, 60))
        model.add_parameter(ChemicalParameter("solvent", "categorical", choices=["A", "B", "C"]))
        model.set_objectives({"yield": 1.0})
        
        # Create planned experiments
        model.planned_experiments = [
            {"temp": 30, "time": 20, "solvent": "A"},
            {"temp": 50, "time": 30, "solvent": "B"},
            {"temp": 70, "time": 40, "solvent": "C"}
        ]
        
        # Add one completed experiment
        model.experiments = [
            {
                "params": {"temp": 30, "time": 20, "solvent": "A"},
                "results": {"yield": 0.75}
            }
        ]
        
        # Create table
        table = ExperimentTable()
        table.model = model  # Attach model reference
        table.update_columns(model)
        
        # Create round indices
        round_start_indices = [0]
        
        # Update table
        table.update_from_planned(model, round_start_indices)
        
        # Verify table content
        assert table.rowCount() > 0
        print(f"-- Experiment table populated with {table.rowCount()} rows")
        
        # Check completed experiment highlighting
        for row in range(table.rowCount()):
            id_item = table.item(row, 1)
            if id_item and id_item.text() == "1":  # First experiment is completed
                bg_color = id_item.background().color()
                is_green = bg_color.green() > bg_color.red() and bg_color.green() > bg_color.blue()
                if is_green:
                    print("-- Completed experiment highlighting successful")
                    break
        else:
            print("-- WARNING: Completed experiment not highlighted properly")
        
    except Exception as e:
        print(f"-- ERROR: Experiment table tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("-- UI component tests completed")
    return True

if __name__ == "__main__":
    print("=== BayesianDOE UI Test Suite ===")
    test_ui_components()