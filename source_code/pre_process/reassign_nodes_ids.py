import glob
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
import multiprocessing
import statistics

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


def generate_mapping_of_nodes_to_ids(group_number, router_topology):
    benign_dataset = pd.read_parquet(PATH_CONFIG.get_benign_dataset_path(group_number))
    nodes_list = list(benign_dataset["NODE"].unique())
    if router_topology == "ROUTER":
        nodes_list.append(CONFIG.ROUTER_ID)

    nodes_list.sort()
    new_ids = list(range(len(nodes_list)))

    mapping_df = pd.DataFrame({"NODE": nodes_list, "NEW_ID": new_ids})
    nodes_mapping_output_path = PATH_CONFIG.get_nodes_mapping_path(
        group_number, router_topology
    )
    UTIL.save_dataframe(mapping_df, nodes_mapping_output_path)


def load_nodes_mapping(group_number, router_topology):
    nodes_mapping_path = PATH_CONFIG.get_nodes_mapping_path(
        group_number, router_topology
    )
    nodes_mapping_df = pd.read_parquet(nodes_mapping_path)
    mapping = dict(zip(nodes_mapping_df["NODE"], nodes_mapping_df["NEW_ID"]))
    return mapping


def reassign_node_ids_for_datasets(group_number, router_topology, p2p_topology):
    mapping = load_nodes_mapping(group_number, router_topology)

    for data_type in ["train", "validation", "test"]:
        dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
            group_number,
            router_topology,
            p2p_topology,
            data_type,
            reassigned=False,
            normalized=False,
        )
        dataset = pd.read_parquet(dataset_path)

        dataset["NODE"] = dataset["NODE"].map(mapping)
        dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
            group_number,
            router_topology,
            p2p_topology,
            data_type,
            reassigned=True,
            normalized=False,
        )
        UTIL.save_dataframe(dataset, dataset_path)


def reassign_node_ids_for_metadata_metrics(group_number, router_topology, p2p_topology):
    mapping = load_nodes_mapping(group_number, router_topology)

    metadata_metrics_path = PATH_CONFIG.get_metadata_metrics_path(
        group_number, p2p_topology, reassigned=False
    )
    dataset = pd.read_parquet(metadata_metrics_path)

    dataset["NODE_1"] = dataset["NODE_1"].map(mapping)
    dataset["NODE_2"] = dataset["NODE_2"].map(mapping)
    metadata_metrics_path = PATH_CONFIG.get_metadata_metrics_path(
        group_number, p2p_topology, reassigned=True
    )
    UTIL.save_dataframe(dataset, metadata_metrics_path)


def reassign_node_ids_for_graph_edges(group_number, router_topology, p2p_topology):
    mapping = load_nodes_mapping(group_number, router_topology)

    graph_path = PATH_CONFIG.get_graph_edges_path(
        group_number, router_topology, p2p_topology, reassigned=False
    )
    dataset = pd.read_parquet(graph_path)
    dataset["NODE_1"] = dataset["NODE_1"].map(mapping)
    dataset["NODE_2"] = dataset["NODE_2"].map(mapping)
    graph_path = PATH_CONFIG.get_graph_edges_path(
        group_number, router_topology, p2p_topology, reassigned=True
    )
    UTIL.save_dataframe(dataset, graph_path)

    for data_type in ["train", "validation", "test"]:
        graph_path = PATH_CONFIG.get_all_graphs_edges_path(
            group_number, router_topology, p2p_topology, data_type, reassigned=False
        )
        dataset = pd.read_parquet(graph_path)
        dataset["NODE_1"] = dataset["NODE_1"].map(mapping)
        dataset["NODE_2"] = dataset["NODE_2"].map(mapping)
        graph_path = PATH_CONFIG.get_all_graphs_edges_path(
            group_number, router_topology, p2p_topology, data_type, reassigned=True
        )
        UTIL.save_dataframe(dataset, graph_path)


def main():
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST

    # Generate the mapping of nodes to IDs which depends only on the router topology.
    for group_number in range(CONFIG.NUM_GROUPS):
        for router_topology in router_topology_list:
            print(
                "Generating mapping of nodes to IDs for group_number: ",
                group_number,
                "  --  router_topology: ",
                router_topology,
            )
            generate_mapping_of_nodes_to_ids(group_number, router_topology)

    # Reassign the node IDs for all the metadata metrics which depends on the p2p topologies.
    for group_number in range(CONFIG.NUM_GROUPS):
        for p2p_topology in p2p_topology_list:
            if p2p_topology == "NO_P2P":
                continue
            print(
                "Reassigning node IDs for metadata metrics for group_number: ",
                group_number,
                "  --  p2p_topology: ",
                p2p_topology,
            )
            reassign_node_ids_for_metadata_metrics(
                group_number, "NO_ROUTER", p2p_topology
            )

    # Reassign the node IDs for all the datasets and graph edges which depends on the router and p2p topologies.
    for group_number in range(CONFIG.NUM_GROUPS):
        for router_topology in router_topology_list:
            for p2p_topology in p2p_topology_list:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                print(
                    "Reassigning node IDs for datasets and graph edges for group_number: ",
                    group_number,
                    "  --  router_topology: ",
                    router_topology,
                    "  --  p2p_topology: ",
                    p2p_topology,
                )
                reassign_node_ids_for_datasets(
                    group_number, router_topology, p2p_topology
                )
                reassign_node_ids_for_graph_edges(
                    group_number, router_topology, p2p_topology
                )


if __name__ == "__main__":
    main()
