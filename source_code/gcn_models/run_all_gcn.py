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
import source_code.utilities.utilities as UTIL


def generate_terminal_command(python_file_path, parameters=None):
    command = "python3 " + python_file_path
    if parameters is not None:
        for key, value in parameters.items():
            command += " --" + key + " " + str(value)
    return command


def main():
    train_gnn_file_path = CONFIG.SOURCE_CODE_DIRECTORY + "gcn_models/train_gcn.py"
    generate_results_gnn_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "gcn_models/generate_results_gcn.py"
    )
    model_analysis_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "gcn_models/compare_models_gcn.py"
    )

    running_python_files = [
        train_gnn_file_path,
        generate_results_gnn_file_path,
        model_analysis_file_path,
    ]

    for python_file_path in running_python_files:
        print("Running " + python_file_path)
        command = generate_terminal_command(python_file_path)
        os.system(command)


if __name__ == "__main__":
    main()
