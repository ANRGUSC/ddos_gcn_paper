import sys
from pathlib import Path
import os

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG


def prepare_output_directory(output_path, remove_old=False):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """

    dir_name = str(os.path.dirname(output_path))
    if remove_old:
        os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def save_dataframe(dataframe, output_path, save_csv=CONFIG.SAVE_CSV):
    """Save the dataframe to the output_path.

    Keyword arguments:
    dataframe -- the dataframe to be saved
    output_path -- path to the output directory
    save_csv -- if True, save the dataframe to csv file, otherwise save to parquet file
    """

    prepare_output_directory(output_path)
    dataframe.to_parquet(output_path, index=False)
    if save_csv:
        # replace the extension with .csv
        output_path = output_path.replace(".parquet", ".csv")
        dataframe.to_csv(output_path, index=False)


def str_to_dict(string):
    # remove the curly braces from the string
    string = string.strip("{}")

    # split the string into key-value pairs
    pairs = string.split(",")

    # use a dictionary comprehension to create the dictionary, converting the values to integers and removing the quotes from the keys
    return {str(key): str(value) for key, value in (pair.split(":") for pair in pairs)}


def extract_dataset_columns(dataset):
    all_columns = list(dataset.columns)
    index_columns = [
        "ATTACK_RATIO",
        "ATTACK_START_TIME",
        "ATTACK_DURATION",
        "ATTACK_PARAMETER",
        "TIME",
        "EDGES_CONNECTION_RATIO",
    ]
    feature_columns = [column for column in all_columns if "PACKET" in column]
    feature_columns.append("TIME_HOUR")
    label_column = ["ATTACKED"]
    node_id_column = ["NODE"]
    all_columns = index_columns + node_id_column + feature_columns + label_column
    return all_columns, index_columns, feature_columns, label_column, node_id_column


def save_float_to_file(float_value, file_name):
    with open(file_name, "w") as file:
        file.write(str(float_value))


def load_float_from_file(file_name):
    with open(file_name, "r") as file:
        integer_value = float(file.read())
    return integer_value
