import glob
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
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


def update_dataset_easy_undrestand(dataset, graph_data, router_id):
    # Define the new grouping and index columns
    index_columns = [
        "TIME",
        "ATTACK_RATIO",
        "ATTACK_START_TIME",
        "ATTACK_DURATION",
        "ATTACK_PARAMETER",
        "EDGES_CONNECTION_RATIO",
    ]

    # Define the columns to be updated
    update_columns = [
        "PACKET",
        "PACKET_30_MIN",
        "PACKET_1_HOUR",
        "PACKET_2_HOUR",
        "PACKET_4_HOUR",
    ]

    # Pre-filter dataset for router rows and set update columns to 0
    router_rows = dataset[dataset["NODE"] == router_id]
    router_rows[update_columns] = 0

    # Index the graph_data based on the index_columns
    graph_data.set_index(index_columns, inplace=True)
    dataset.set_index(index_columns, inplace=True)
    router_rows.set_index(index_columns, inplace=True)

    # Iterate over each router row
    for idx, router_row in router_rows.iterrows():
        # Access the relevant graph data
        graph_data_tmp = graph_data.loc[[idx], :]

        # Find nodes connected to the router
        nodes_connected_to_router = graph_data_tmp[
            graph_data_tmp["NODE_2"] == router_id
        ]["NODE_1"].unique()

        if nodes_connected_to_router.size > 0:
            # Sum the values of the specified columns for connected nodes

            sum_values = dataset.loc[
                (dataset["NODE"].isin(nodes_connected_to_router))
                & dataset.index.isin([idx], level=index_columns),
                update_columns,
            ].sum()

            # Update the router row
            router_rows.loc[idx, update_columns] = sum_values

    # Update the dataset with modified router rows
    dataset.update(router_rows[update_columns].reset_index(index_columns))

    return dataset


def update_dataset_optimized(
    dataset, graph_data, index_columns, update_columns, router_id
):
    # Pre-filter dataset for router rows and set update columns to 0
    router_rows = dataset[dataset["NODE"] == router_id].copy()
    router_rows[update_columns] = 0

    # Convert to MultiIndex for faster processing
    dataset.set_index(index_columns + ["NODE"], inplace=True, drop=False)
    dataset.sort_index(inplace=True)
    graph_data.set_index(index_columns + ["NODE_1", "NODE_2"], inplace=True, drop=False)
    graph_data.sort_index(inplace=True)
    router_rows.set_index(index_columns + ["NODE"], inplace=True, drop=False)
    router_rows.sort_index(inplace=True)

    # Iterate over the index_columns to find connected nodes and sum their values
    for idx in router_rows.index.droplevel("NODE"):
        # Find nodes that are connected to the router node in this group
        connected_nodes = (
            graph_data.loc[idx]
            .query("NODE_2 == @router_id")
            .index.get_level_values("NODE_1")
        )

        if connected_nodes.size > 0:
            # Sum the values of the specified columns for connected nodes in this group
            sum_values = (
                dataset.loc[idx]
                .loc[dataset.loc[idx, "NODE"].isin(connected_nodes)][update_columns]
                .sum()
            )
            router_idx = (*idx, router_id)
            router_rows.loc[router_idx, update_columns] = sum_values

    # Update the original dataset with modified router rows
    dataset.update(router_rows[update_columns])

    # Reset index if needed
    dataset.reset_index(inplace=True, drop=True)

    # sort the dataset
    dataset.sort_values(
        index_columns + ["NODE"],
        inplace=True,
    )

    return dataset


def main_update_router_training_data_mean_values(group_number, p2p_topology, data_type):
    dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        "ROUTER",
        p2p_topology,
        data_type,
        reassigned=False,
        normalized=False,
    )
    graph_path = PATH_CONFIG.get_all_graphs_edges_path(
        group_number, "ROUTER", p2p_topology, data_type, reassigned=False
    )

    dataset = pd.read_parquet(dataset_path)
    dataset["TIME"] = pd.to_datetime(dataset["TIME"])
    graph_data = pd.read_parquet(graph_path)

    # Define the new grouping and index columns
    index_columns = [
        "ATTACK_RATIO",
        "ATTACK_START_TIME",
        "ATTACK_DURATION",
        "ATTACK_PARAMETER",
        "EDGES_CONNECTION_RATIO",
        "TIME",
    ]

    # Define the columns to be updated
    update_columns = [
        "PACKET",
        "PACKET_30_MIN",
        "PACKET_1_HOUR",
        "PACKET_2_HOUR",
        "PACKET_4_HOUR",
    ]

    # update the dataset
    updated_dataset = update_dataset_optimized(
        dataset, graph_data, index_columns, update_columns, CONFIG.ROUTER_ID
    )

    UTIL.save_dataframe(updated_dataset, dataset_path)
    print(
        "Finished updating router training data mean values for group_number: ",
        group_number,
        "  --  p2p_topology: ",
        p2p_topology,
        "  --  data_type: ",
        data_type,
    )


def main_one_process():
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST

    if not CONFIG.USE_ROUTER:
        return

    for group_number in range(CONFIG.NUM_GROUPS):
        for p2p_topology in p2p_topology_list:
            for data_type in ["train", "validation", "test"]:
                print(
                    "group_number: ",
                    group_number,
                    "  --  p2p_topology: ",
                    p2p_topology,
                    "  --  data_type: ",
                    data_type,
                )
                main_update_router_training_data_mean_values(
                    group_number, p2p_topology, data_type
                )


def main_multi_process():
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST

    if not CONFIG.USE_ROUTER:
        return

    tasks = []

    for group_number in range(CONFIG.NUM_GROUPS):
        for p2p_topology in p2p_topology_list:
            for data_type in ["train", "validation", "test"]:
                tasks.append((group_number, p2p_topology, data_type))

    pool = multiprocessing.Pool(processes=3)
    pool.starmap(main_update_router_training_data_mean_values, tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main_multi_process()
