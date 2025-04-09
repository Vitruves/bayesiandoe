import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.rcParams['figure.autolayout'] = True
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            # Use tight_layout safely
            self.fig.tight_layout(pad=2.0)
        except:
            pass  # Skip if tight_layout fails
        self.draw_idle()

class Mpl3DCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.rcParams['figure.autolayout'] = True
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=False)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Use manual adjustment instead of tight_layout for 3D plots
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Don't use tight_layout for 3D plots - causes warning
        self.draw_idle()