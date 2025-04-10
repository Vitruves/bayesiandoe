from PySide6.QtWidgets import QTableWidget, QMainWindow
from PySide6.QtCore import Qt

def view_selected_round(self, selected_round):
    """Switch the experiment table view to show a specific round"""
    if not hasattr(self, 'experiment_table'):
        return
        
    if 1 <= selected_round <= self.current_round:
        # Temporarily store current round as view_round to avoid changing current_round
        self.view_round = selected_round
        
        # Update table to show selected round
        # Pass temporary flag to experiment_table to filter by view_round
        self.experiment_table.filter_by_round = selected_round
        self.experiment_table.update_from_planned(self.model, self.round_start_indices)
        
        # Visual indicator for historical data
        if selected_round < self.current_round:
            self.status_label.setText(f"Viewing historical data from Round {selected_round} (read-only)")
            # Set read-only mode for historical rounds
            self.experiment_table.setEditTriggers(QTableWidget.NoEditTriggers)
        else:
            self.status_label.setText(f"Current experiment Round {selected_round}")
            # Enable editing for current round
            self.experiment_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed) 