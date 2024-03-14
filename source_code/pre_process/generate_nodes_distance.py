import sys
from pathlib import Path
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    classification_report,
)
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from itertools import product
from multiprocessing import Pool, Manager

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


def generate_distances(benign_data, node1, node2, distance_data_rows):
    if node1 >= node2:
        return
    # print(node1, '   ,   ', node2)
    benign_data = benign_data.groupby(["NODE"]).max()[["LAT", "LNG"]]
    data1 = benign_data.loc[node1]
    data2 = benign_data.loc[node2]
    dist = math.sqrt(
        (data1["LAT"] - data2["LAT"]) ** 2 + (data1["LNG"] - data2["LNG"]) ** 2
    )
    temp_df = pd.DataFrame(
        {
            "NODE_1": [node1, node2],
            "LAT_1": [data1["LAT"], data2["LAT"]],
            "LNG_1": [data1["LNG"], data2["LNG"]],
            "NODE_2": [node2, node1],
            "LAT_2": [data2["LAT"], data1["LAT"]],
            "LNG_2": [data2["LNG"], data1["LNG"]],
            "DISTANCE": [dist, dist],
        }
    )
    distance_data_rows.append(temp_df)


def main_generate_distances(group_number):
    benign_dataset_path = PATH_CONFIG.get_benign_dataset_path(group_number)

    benign_dataset = pd.read_parquet(benign_dataset_path)
    nodes = list(benign_dataset["NODE"].unique())

    output_path = PATH_CONFIG.get_metadata_metrics_path(
        group_number, "DISTANCE", reassigned=False
    )

    distance_data = pd.DataFrame(
        columns=["NODE_1", "LAT_1", "LNG_1", "NODE_2", "LAT_2", "LNG_2", "DISTANCE"]
    )

    manager = Manager()
    distance_data_rows = manager.list([distance_data])

    p = Pool(30)
    p.starmap(
        generate_distances,
        product([benign_dataset], nodes, nodes, [distance_data_rows]),
    )
    p.close()
    p.join()

    distance_data = pd.concat(distance_data_rows, ignore_index=True)
    distance_data = distance_data.sort_values(by=["NODE_1", "NODE_2"])
    column_types = {
        "NODE_1": "int32",
        "LAT_1": "float32",
        "LNG_1": "float32",
        "NODE_2": "int32",
        "LAT_2": "float32",
        "LNG_2": "float32",
        "DISTANCE": "float32",
    }
    distance_data = distance_data.astype(column_types)
    UTIL.save_dataframe(distance_data, output_path)

    print(distance_data)
    print("shape: ", distance_data.shape)


def main():
    for group_number in range(CONFIG.NUM_GROUPS):
        main_generate_distances(group_number)


if __name__ == "__main__":
    main()
