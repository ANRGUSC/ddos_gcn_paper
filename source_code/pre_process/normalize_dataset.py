import glob
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pickle import dump, load
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
import multiprocessing
import statistics
import json

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


def generate_scaler(group_number, router_topology, p2p_topology):
    dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        router_topology,
        p2p_topology,
        "train",
        reassigned=True,
        normalized=False,
    )
    dataset = pd.read_parquet(dataset_path)

    (
        all_columns,
        index_columns,
        feature_columns,
        label_column,
        node_id_column,
    ) = UTIL.extract_dataset_columns(dataset)

    scaler = StandardScaler()
    scaler.fit_transform(dataset[feature_columns])
    scaler_path = PATH_CONFIG.get_scaler_path(
        group_number, router_topology, p2p_topology
    )
    dump(scaler, open(scaler_path, "wb"))


def normalize_dataset(group_number, router_topology, p2p_topology, data_type):
    dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        reassigned=True,
        normalized=False,
    )
    dataset = pd.read_parquet(dataset_path)

    (
        all_columns,
        index_columns,
        feature_columns,
        label_column,
        node_id_column,
    ) = UTIL.extract_dataset_columns(dataset)

    scaler_path = PATH_CONFIG.get_scaler_path(
        group_number, router_topology, p2p_topology
    )
    scaler = load(open(scaler_path, "rb"))
    dataset[feature_columns] = scaler.transform(dataset[feature_columns])

    dataset_output_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        reassigned=True,
        normalized=True,
    )
    UTIL.save_dataframe(dataset, dataset_output_path)


def main():
    for group_number in range(CONFIG.NUM_GROUPS):
        for router_topology in CONFIG.ROUTER_TOPOLOGY_LIST:
            for p2p_topology in CONFIG.P2P_TOPOLOGY_LIST:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                print(
                    "Normalizing dataset for group_number: ",
                    group_number,
                    "  --  router_topology: ",
                    router_topology,
                    "  --  p2p_topology: ",
                    p2p_topology,
                )
                generate_scaler(group_number, router_topology, p2p_topology)
                for data_type in ["train", "validation", "test"]:
                    normalize_dataset(
                        group_number, router_topology, p2p_topology, data_type
                    )


if __name__ == "__main__":
    main()
