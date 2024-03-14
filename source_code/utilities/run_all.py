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
    run_all_pre_process = (
        CONFIG.SOURCE_CODE_DIRECTORY + "pre_process/run_all_pre_process.py"
    )
    run_all_gcn = CONFIG.SOURCE_CODE_DIRECTORY + "gcn_models/run_all_gcn.py"

    running_python_files = [
        run_all_pre_process,
        run_all_gcn,
    ]

    for python_file_path in running_python_files:
        print("Running " + python_file_path)
        command = generate_terminal_command(python_file_path)
        os.system(command)


if __name__ == "__main__":
    main()
