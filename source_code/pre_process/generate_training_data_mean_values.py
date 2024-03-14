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


def add_router_to_dataset(data):
    """
    This function takes a dataset as input, adds a new node to the dataset as per specified rules,
    and returns the updated dataset.

    Args:
    csv_file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The updated dataset with the new node added.
    """
    # Identify columns with only one unique value
    unique_value_columns = [col for col in data.columns if data[col].nunique() == 1]

    # Calculating average LAT and LNG for the new node
    average_lat = data["LAT"].mean()
    average_lng = data["LNG"].mean()

    # Creating a new dataframe for the new node
    new_node_df = data[["TIME"]].drop_duplicates()
    new_node_df["NODE"] = CONFIG.ROUTER_ID
    new_node_df["LAT"] = average_lat
    new_node_df["LNG"] = average_lng

    # Extracting the "hour" from "TIME" for the new node
    new_node_df["TIME_HOUR"] = pd.to_datetime(new_node_df["TIME"]).dt.hour

    # Calculating the sum of PACKET and the max of ACTIVE and ATTACKED for each TIME, across all nodes
    packet_sum = data.groupby("TIME")["PACKET"].sum().reset_index()
    active_max = data.groupby("TIME")["ACTIVE"].max().reset_index()
    attacked_max = data.groupby("TIME")["ATTACKED"].max().reset_index()

    # Merging these values into the new node dataframe
    new_node_df = new_node_df.merge(packet_sum, on="TIME", how="left")
    new_node_df = new_node_df.merge(active_max, on="TIME", how="left")
    new_node_df = new_node_df.merge(attacked_max, on="TIME", how="left")

    # Keeping the unique columns with their unique values
    for column in unique_value_columns:
        if column not in ["ATTACKED", "ACTIVE", "PACKET", "TIME"]:
            new_node_df[column] = data[column].iloc[0]

    # Adding the new node to the original dataset
    updated_data = pd.concat([data, new_node_df], ignore_index=True)

    return updated_data


def combine_data(router_topology, num_days, mean_windows, input_path, output_path):
    """Combine the csv files in the input_path directory and output the combined one to the output_path.

    Keyword arguments:
    input_path -- The path to the input directory.
    output_path -- The path to the output_directory for storing the combined data.
    """
    all_files = [fname for fname in glob.glob(os.path.join(input_path, "*.parquet"))]
    data_tmp = pd.read_parquet(all_files[0])
    combined_data = pd.DataFrame()

    for file in all_files:
        data_tmp = pd.read_parquet(file)
        data_tmp["TIME"] = pd.to_datetime(data_tmp["TIME"])

        if router_topology == "ROUTER":
            data_tmp = add_router_to_dataset(data_tmp)

        data_tmp.sort_values(by=["NODE", "TIME"], inplace=True)
        # Note that the time values are not unique in the dataset. The combination of (node, time) is unique but
        # Pandas is ok to have index values that are not unique.
        data_tmp.set_index("TIME", inplace=True)
        for mean_window, mean_window_title in mean_windows.items():
            df_avg = (
                data_tmp.groupby("NODE")["PACKET"]
                .rolling(window=mean_window)
                .mean()
                .reset_index()
            )
            df_avg.sort_values(by=["NODE", "TIME"], inplace=True)
            df_avg.set_index("TIME", inplace=True)
            data_tmp[mean_window_title] = df_avg["PACKET"]
        data_tmp.reset_index(inplace=True)
        begin_date = data_tmp["TIME"][0] + timedelta(days=2)
        end_date = data_tmp["TIME"][0] + timedelta(days=2 + num_days)
        data_tmp = data_tmp.loc[
            (data_tmp["TIME"] >= begin_date) & (data_tmp["TIME"] < end_date)
        ]
        data_tmp = data_tmp.sort_values(by=["NODE", "TIME"]).reset_index(drop=True)
        combined_data = pd.concat([combined_data, data_tmp])

    # Add the edges connection ratio to the combined data.
    final_data = pd.DataFrame()
    for edges_connetion_ratio in CONFIG.EDGES_CONNECTION_RATIO_LIST:
        combined_data["EDGES_CONNECTION_RATIO"] = edges_connetion_ratio
        final_data = pd.concat([final_data, combined_data])

    UTIL.save_dataframe(final_data, output_path)


def main_combine_data(
    group_number, router_topology, p2p_topology, data_type, num_days, mean_windows
):
    """The main function to be used for calling  combine_data function.

    Keyword arguments:
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    """
    input_path = PATH_CONFIG.get_attack_dataset_directory_path(group_number, data_type)
    output_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        reassigned=False,
        normalized=False,
    )
    # We do not need to pass p2p_topology to combine_data function because we only need to either add the router
    # node or not based on the router_topology. The p2p_topology is used for generating the graph edges and update
    # training data mean values for the router.
    combine_data(router_topology, num_days, mean_windows, input_path, output_path)
    print(
        "Finished generating dataset for group ",
        str(group_number),
        " -- data_type: ",
        data_type,
        " -- router_topology: ",
        router_topology,
        " -- p2p_topology: ",
        p2p_topology,
    )


def main_one_process():
    num_days = {}
    num_days["train"] = CONFIG.NUM_TRAIN_DAYS
    num_days["validation"] = CONFIG.NUM_VALIDATION_DAYS
    num_days["test"] = CONFIG.NUM_TEST_DAYS
    mean_windows = CONFIG.MEAN_WINDOWS
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST

    for group_number in range(CONFIG.NUM_GROUPS):
        for router_topology in router_topology_list:
            for p2p_topology in p2p_topology_list:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                for data_type in ["train", "validation", "test"]:
                    main_combine_data(
                        group_number,
                        router_topology,
                        p2p_topology,
                        data_type,
                        num_days[data_type],
                        mean_windows,
                    )


def main_multi_process():
    num_days = {}
    num_days["train"] = CONFIG.NUM_TRAIN_DAYS
    num_days["validation"] = CONFIG.NUM_VALIDATION_DAYS
    num_days["test"] = CONFIG.NUM_TEST_DAYS
    mean_windows = CONFIG.MEAN_WINDOWS
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST
    tasks = []

    for group_number in range(CONFIG.NUM_GROUPS):
        for router_topology in router_topology_list:
            for p2p_topology in p2p_topology_list:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                for data_type in ["train", "validation", "test"]:
                    tasks.append(
                        (
                            group_number,
                            router_topology,
                            p2p_topology,
                            data_type,
                            num_days[data_type],
                            mean_windows,
                        )
                    )

    pool = multiprocessing.Pool(processes=28)
    pool.starmap(main_combine_data, tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main_multi_process()
