import sys
from PySide6.QtWidgets import QApplication
from .ui.main_window import BayesianDOEApp
from .ui.widgets import SplashScreen
from PySide6.QtCore import QTimer

def main():
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    
    splash = SplashScreen()
    splash.show()
    splash.raise_()
    
    app.processEvents()
    
    window = BayesianDOEApp()
    
    def check_splash():
        if not splash.isVisible():
            timer.stop()
            window.showMaximized()
    
    timer = QTimer()
    timer.timeout.connect(check_splash)
    timer.start(100)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()