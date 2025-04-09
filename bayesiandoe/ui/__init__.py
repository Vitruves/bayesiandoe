from .main_window import BayesianDOEApp
from .canvas import MplCanvas, Mpl3DCanvas
from .widgets import (
    LogDisplay, SplashScreen, ParameterTable, ExperimentTable,
    BestResultsTable, AllResultsTable
)
from .dialogs import (
    ParameterDialog, ResultDialog, TemplateSelector, PriorDialog,
    OptimizationSettingsDialog
)
from .tab_setup import (
    setup_setup_tab, setup_prior_tab, setup_experiment_tab, 
    setup_results_tab, setup_analysis_tab
)
from .ui_actions import (
    add_parameter, edit_parameter, remove_parameter,
    load_template, add_from_registry, add_to_registry, remove_from_registry,
    generate_initial_experiments, generate_next_experiments,
    add_result_for_selected, new_project, open_project, save_project,
    import_data, export_results, statistical_analysis, plan_parallel_experiments,
    open_structure_editor, show_optimization_settings, show_preferences,
    show_documentation, show_about, add_substrate_parameter
)
from .ui_visualization import (
    update_prior_plot, update_results_plot, update_model_plot,
    update_surface_plot, update_convergence_plot, update_correlation_plot
)
from .ui_utils import (
    log, update_ui_from_model, update_parameter_combos, update_prior_table,
    update_prior_ui, update_prior_param_buttons, update_best_result_label,
    update_rounding_settings, update_std_from_confidence
)
from .ui_callbacks import (
    update_objectives, show_registry_item_tooltip, refresh_registry,
    on_prior_selected, update_best_results, show_result_details,
    show_prior_help, on_param_button_clicked
)