#!/usr/bin/env python3

import sys
import os
import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

def main():
    # Create application before splash screen to ensure Qt is initialized
    app = QApplication(sys.argv)
    
    # Apply global stylesheet for consistent visuals
    app.setStyle('Fusion')
    
    # Import has to be done after QApplication to avoid issues with PySide
    from bayesiandoe.ui.widgets import SplashScreen
    
    # Show splash screen immediately
    splash = SplashScreen()
    splash.show()
    
    # Force immediate processing for splash screen
    app.processEvents()
    
    # Use Metal renderer on macOS if available
    if sys.platform == 'darwin':
        try:
            from PySide6.QtGui import QSurfaceFormat
            format = QSurfaceFormat()
            format.setRenderableType(QSurfaceFormat.OpenGL)
            QSurfaceFormat.setDefaultFormat(format)
        except:
            pass
    
    # Use a timer to delay importing the main window
    # This gives time for the splash screen to be displayed
    def load_main_window():
        # Import main window after showing splash
        from bayesiandoe.ui.main_window import BayesianDOEApp
        
        # Create main window (hidden initially)
        main_window = BayesianDOEApp()
        
        # Override splash screen's close method to properly transition
        original_close = splash.close
        
        def on_splash_finished():
            # Make sure splash is fully closed
            if hasattr(original_close, '__call__'):
                original_close()
            
            # Show main window with a slight delay to ensure splash is gone
            QTimer.singleShot(100, lambda: (
                main_window.show(),
                main_window.raise_(),
                main_window.activateWindow()
            ))
        
        # Replace splash close with our transition function
        splash.close = on_splash_finished
    
    # Use a short delay to ensure splash screen renders first
    QTimer.singleShot(10, load_main_window)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 