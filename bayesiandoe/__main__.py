#!/usr/bin/env python3

import sys
import os
import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

def main():
    # Create application before splash screen to ensure Qt is initialized
    app = QApplication(sys.argv)
    
    # Import has to be done after QApplication to avoid issues with PySide
    from bayesiandoe.ui.widgets import SplashScreen
    
    # Show splash screen immediately
    splash = SplashScreen()
    splash.show()
    
    # Process events to ensure splash shows immediately
    app.processEvents()
    
    # Use a timer to delay importing the main window
    # This allows the splash screen to be displayed immediately
    def load_main_window():
        # Import main window after showing splash to make loading visible
        from bayesiandoe.ui.main_window import BayesianDOEApp
        
        # Create main window
        main_window = BayesianDOEApp()
        
        # When splash finishes, show main window
        def show_main_window():
            main_window.show()
            main_window.raise_()
            main_window.activateWindow()
        
        # Connect splash finish to main window display
        splash.close = lambda: (super(type(splash), splash).close(), show_main_window())
    
    # Use a short delay to allow splash screen to render first
    QTimer.singleShot(50, load_main_window)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 