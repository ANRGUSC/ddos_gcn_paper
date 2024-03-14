import sys
from pathlib import Path
import math
import pandas as pd
import numpy as np
import os
from itertools import product
from multiprocessing import Pool, Manager
from pickle import dump
import matplotlib.pyplot as plt
import glob

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


def generate_correlations(benign_data, node1, node2, correlation_data_rows):
    if node1 >= node2:
        return
    # print(node1, '   ,   ', node2)
    data1 = benign_data.loc[benign_data["NODE"] == node1].reset_index(drop=True)
    data2 = benign_data.loc[benign_data["NODE"] == node2].reset_index(drop=True)
    correlation = data1["PACKET"].corr(data2["PACKET"])
    temp_df = pd.DataFrame(
        {
            "NODE_1": [node1, node2],
            "LAT_1": [data1["LAT"][0], data2["LAT"][0]],
            "LNG_1": [data1["LNG"][0], data2["LNG"][0]],
            "NODE_2": [node2, node1],
            "LAT_2": [data2["LAT"][0], data1["LAT"][0]],
            "LNG_2": [data2["LNG"][0], data1["LNG"][0]],
            "CORRELATION": [correlation, correlation],
        }
    )
    correlation_data_rows.append(temp_df)


def main_generate_correlations(group_number):
    benign_dataset_path = PATH_CONFIG.get_benign_dataset_path(group_number)
    benign_dataset = pd.read_parquet(benign_dataset_path)
    nodes = list(benign_dataset["NODE"].unique())

    output_path = PATH_CONFIG.get_metadata_metrics_path(
        group_number, "CORRELATION", reassigned=False
    )

    correlation_data = pd.DataFrame(
        columns=["NODE_1", "LAT_1", "LNG_1", "NODE_2", "LAT_2", "LNG_2", "CORRELATION"]
    )

    manager = Manager()
    correlation_data_rows = manager.list([correlation_data])

    p = Pool(30)
    p.starmap(
        generate_correlations,
        product([benign_dataset], nodes, nodes, [correlation_data_rows]),
    )
    p.close()
    p.join()

    correlation_data = pd.concat(correlation_data_rows, ignore_index=True)
    correlation_data = correlation_data.sort_values(by=["NODE_1", "NODE_2"])

    column_types = {
        "NODE_1": "int32",
        "LAT_1": "float32",
        "LNG_1": "float32",
        "NODE_2": "int32",
        "LAT_2": "float32",
        "LNG_2": "float32",
        "CORRELATION": "float32",
    }
    correlation_data = correlation_data.astype(column_types)
    UTIL.save_dataframe(correlation_data, output_path)

    print(correlation_data)
    print("shape: ", correlation_data.shape)


def main():
    for group_number in range(CONFIG.NUM_GROUPS):
        main_generate_correlations(group_number)


if __name__ == "__main__":
    main()
