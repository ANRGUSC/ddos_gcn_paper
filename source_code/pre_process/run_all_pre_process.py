import os
from pathlib import Path
import sys

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG


def generate_terminal_command(python_file_path, parameters=None):
    command = "python3 " + python_file_path
    if parameters is not None:
        for key, value in parameters.items():
            command += " --" + key + " " + str(value)
    return command


def main():
    clean_dataset_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/clean_dataset.py"
    )
    generate_nodes_distance_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/generate_nodes_distance.py"
    )
    generate_nodes_pearson_correlation_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY
        + "pre_process/generate_nodes_pearson_correlation.py"
    )
    generate_attack_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/generate_attack.py"
    )
    generate_training_data_mean_values_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY
        + "pre_process/generate_training_data_mean_values.py"
    )
    generate_graph_edges_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/generate_graph_edges.py"
    )
    generate_router_training_data_mean_values_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY
        + "pre_process/generate_router_training_data_mean_values.py"
    )
    reassign_nodes_ids_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/reassign_nodes_ids.py"
    )
    normalize_dataset_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/normalize_dataset.py"
    )

    running_python_files = [
        clean_dataset_file_path,
        generate_nodes_distance_file_path,
        generate_nodes_pearson_correlation_file_path,
        generate_attack_file_path,
        generate_training_data_mean_values_file_path,
        generate_graph_edges_file_path,
        generate_router_training_data_mean_values_file_path,
        reassign_nodes_ids_file_path,
        normalize_dataset_file_path,
    ]

    for python_file_path in running_python_files:
        print("Running " + python_file_path)
        command = generate_terminal_command(python_file_path)
        os.system(command)


if __name__ == "__main__":
    main()
