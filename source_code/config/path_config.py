import sys
from pathlib import Path

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.utilities.utilities as UTIL


# %% Pre-process:
def get_benign_packet_path():
    path = (
        CONFIG.DATASET_DIRECTORY
        + "N_BaIoT/extracted_packets/SimpleHome_XCS7_1003_WHT_Security_Camera_benign_aggregation_Source-IP_time_window_10-sec_stat_Number.csv"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_original_data_path():
    path = CONFIG.DATASET_DIRECTORY + "original_data/original_data.csv"
    UTIL.prepare_output_directory(path)
    return path


def get_benign_dataset_path(group_number):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/benign_data/benign_data_"
        + str(CONFIG.BEGIN_DATE)
        + "_"
        + str(CONFIG.END_DATE)
        + "_time_step_"
        + str(CONFIG.TIME_STEP)
        + "_num_ids_"
        + str(CONFIG.NUM_NODES)
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_metadata_metrics_path(group_number, metric, reassigned=False):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/metadata/"
        + metric
        + ("_reassigned" if reassigned else "")
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_attack_dataset_directory_path(group_number, data_type):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/"
        + "group_"
        + str(group_number)
        + "/attacked_data/"
        + data_type
        + "/"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_dataset_train_eval_path(
    group_number,
    router_topology,
    p2p_topology,
    data_type,
    reassigned=False,
    normalized=False,
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/"
        + data_type
        + "_data/"
        + router_topology
        + "/"
        + p2p_topology
        + "/"
        + data_type
        + "_data"
        + ("_reassigned" if reassigned else "")
        + ("_normalized" if normalized else "")
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_nodes_mapping_path(group_number, router_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/nodes_mapping/"
        + router_topology
        + "/"
        + "nodes_mapping.parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_scaler_path(group_number, router_topology, p2p_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/scaler/"
        + router_topology
        + "/"
        + p2p_topology
        + "/"
        + "scaler.pkl"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_graph_edges_path(group_number, router_topology, p2p_topology, reassigned=False):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/graph_edges/"
        + router_topology
        + "/"
        + p2p_topology
        + "/"
        + "graph_edges"
        + ("_reassigned" if reassigned else "")
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_all_graphs_edges_path(
    group_number, router_topology, p2p_topology, data_type, reassigned=False
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "pre_process/Output/group_"
        + str(group_number)
        + "/graph_edges/"
        + router_topology
        + "/"
        + p2p_topology
        + "/"
        + data_type
        + "_"
        + "all_graphs_edges"
        + ("_reassigned" if reassigned else "")
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


# %% NN Training:


def get_each_epoch_save_model_path(group_number, router_topology, p2p_topology, epoch):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/saved_models/all_epochs/model_epoch_"
        + str(epoch)
        + ".pth"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_best_model_path(
    group_number, router_topology, p2p_topology, data_type, saving_metric
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/saved_models/best_model_"
        + data_type
        + "_"
        + saving_metric
        + ".pth"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_training_logs_path(group_number, router_topology, p2p_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/logs/data/logs.parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_training_logs_plot_path_directory(group_number, router_topology, p2p_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/logs/plot/"
    )
    UTIL.prepare_output_directory(path)
    return path


# %% NN Evaluation:
def get_optimal_threshold_path(group_number, router_topology, p2p_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/optimal_threshold/"
        + "optimal_threshold.txt"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_prediction_probability_dataset_path(
    group_number, router_topology, p2p_topology, data_type
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/data/"
        + data_type
        + "/"
        + "prediction_probability.parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_confusion_matrix_path(group_number, router_topology, p2p_topology, data_type):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/data/"
        + data_type
        + "/"
        + "confusion_matrix.parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_aggregated_report_data(
    group_number, router_topology, p2p_topology, data_type, aggregate_parameter
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/data/"
        + data_type
        + "/"
        + "aggregated_report_"
        + aggregate_parameter
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_roc_data_path(group_number, router_topology, p2p_topology, data_type):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/data/"
        + data_type
        + "/"
        + "roc_data.parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_roc_plot_path(group_number, router_topology, p2p_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/plot/binary_metrics/"
        + "roc_curve.png"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_aggregated_report_plot_directory(group_number, router_topology, p2p_topology):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/plot/binary_metrics/"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_attack_prediction_vs_time_plot_path(
    group_number, router_topology, p2p_topology, data_type, average_all=False
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training/group_"
        + str(group_number)
        + "/GCN/"
        + router_topology
        + "/"
        + p2p_topology
        + "/reports/plot/attack_prediction_vs_time/"
        + data_type
        + "/"
        + ("ap_vs_time_avg.png" if average_all else "all/ap_vs_time_")
    )
    UTIL.prepare_output_directory(path)
    return path


# %% Model Analysis:


def get_model_analysis_aggregated_report_data_path(
    router_topology, p2p_topology, data_type, aggregate_parameters_list
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "model_analysis/GCN/data/"
        + router_topology
        + "/"
        + p2p_topology
        + "/"
        + data_type
        + "/"
        + "_".join(aggregate_parameters_list)
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_model_analysis_aggregated_report_data_combined_path(aggregate_parameters_list):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "model_analysis/GCN/data/combined_"
        + "_".join(aggregate_parameters_list)
        + ".parquet"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_model_analysis_confidence_interval_plot_path(
    filter_dataset_dict,
    aggregate_parameters_list,
    confidence_parameter,
    aggregate_metric,
    plot_metric,
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "model_analysis/GCN/plot/confidence_interval/"
        + "_".join(filter_dataset_dict.values())
        + "/"
        + "_".join(aggregate_parameters_list)
        + "_"
        + confidence_parameter
        + "_"
        + aggregate_metric
        + "_"
        + plot_metric
        + ".png"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_model_analysis_error_bar_plot_path(
    filter_dataset_dict,
    aggregate_parameters_list,
    confidence_parameter,
    aggregate_metric,
    plot_metric,
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "model_analysis/GCN/plot/error_bar/"
        + "_".join(filter_dataset_dict.values())
        + "/"
        + "_".join(aggregate_parameters_list)
        + "_"
        + confidence_parameter
        + "_"
        + aggregate_metric
        + "_"
        + plot_metric
        + ".png"
    )
    UTIL.prepare_output_directory(path)
    return path


def get_edge_analysis_confidence_interval_plot_path(
    filter_dataset_dict,
    aggregate_parameters_list,
    confidence_parameter,
    aggregate_metric,
    plot_metric,
):
    path = (
        CONFIG.OUTPUT_DIRECTORY
        + "edge_analysis/GCN/plot/confidence_interval/"
        + "_".join(filter_dataset_dict.values())
        + "/"
        + "_".join(aggregate_parameters_list)
        + "_"
        + confidence_parameter
        + "_"
        + aggregate_metric
        + "_"
        + plot_metric
        + ".png"
    )
    UTIL.prepare_output_directory(path)
    return path
