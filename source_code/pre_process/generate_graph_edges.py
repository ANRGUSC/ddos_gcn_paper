import glob
import sys
from pathlib import Path
import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import multiprocessing as mp
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


def random_edge_sampling(df, edges_connection_ratio, directed_graph=True):
    if directed_graph:
        # For directed graphs
        # Just perform random sampling
        sample_size = math.ceil(len(df) * edges_connection_ratio)
        new_df = df.sample(n=sample_size)
    else:
        # For undirected graphs
        # Step 1: Create a canonical representation of edges
        df_sorted = pd.DataFrame(np.sort(df.values, axis=1), columns=df.columns)

        # Step 2: Remove duplicate edges
        df_unique = df_sorted.drop_duplicates()

        # Step 3: Random sampling
        sample_size = math.ceil(len(df_unique) * edges_connection_ratio)
        sampled_edges = df_unique.sample(n=sample_size)

        # Step 4: Recreate edge list with both directions
        reversed_edges = sampled_edges.rename(
            columns={"NODE_1": "NODE_2", "NODE_2": "NODE_1"}
        )
        new_df = pd.concat([sampled_edges, reversed_edges])

    return new_df


def process_group_merge(args):
    (
        graph_edges,
        (index, group),
        edges_connection_ratio_index,
        index_columns,
        directed_graph,
    ) = args
    # Create a temporary dataframe with index_columns for merging
    temp = pd.DataFrame([index], columns=index_columns)

    # Randomly sample from the graph_edges dataframe
    # Note: for each group, which is based on the index_columns, i.e.
    # ['ATTACK_RATIO', 'ATTACK_START_TIME', 'ATTACK_DURATION', 'ATTACK_PARAMETER', 'TIME', 'EDGES_CONNECTION_RATIO'],
    # we sample the edges from the graph_edges. This means at each "TIME", we sample the edges from the graph_edges.
    edges_connection_ratio = index[edges_connection_ratio_index]
    graph_edges_tmp = random_edge_sampling(
        graph_edges, edges_connection_ratio, directed_graph
    )

    # Merge the sampled dataframe with graph_edges, keeping all combinations
    merged = graph_edges_tmp.merge(temp, how="cross")
    return merged


def generate_all_graphs_edges(
    group_number, data_type, router_topology, p2p_topology, directed_graph
):
    # Generate graph_edges for each data_type and consider all the edges_connection_ratio

    dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        reassigned=False,
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

    graph_edges_path = PATH_CONFIG.get_graph_edges_path(
        group_number, router_topology, p2p_topology, reassigned=False
    )
    graph_edges = pd.read_parquet(graph_edges_path)

    # Group the dataset by index_columns
    grouped = dataset.groupby(index_columns)
    # Get the index of EDGES_CONNECTION_RATIO in index_columns to be used for retrieving the value of EDGES_CONNECTION_RATIO
    edges_connection_ratio_index = index_columns.index("EDGES_CONNECTION_RATIO")

    # Initialize a list to store the resulting dataframes
    all_graphs_edges_list = []

    # Prepare the arguments for multiprocessing
    args = [
        (
            graph_edges,
            group,
            edges_connection_ratio_index,
            index_columns,
            directed_graph,
        )
        for group in grouped
    ]

    # Use a Pool of processes
    with Pool(28) as pool:
        all_graphs_edges_list = pool.map(process_group_merge, args)

    # Concatenate all the resulting dataframes
    all_graphs_edges = pd.concat(all_graphs_edges_list, ignore_index=True)
    all_graphs_edges_path = PATH_CONFIG.get_all_graphs_edges_path(
        group_number, router_topology, p2p_topology, data_type, reassigned=False
    )
    UTIL.save_dataframe(all_graphs_edges, all_graphs_edges_path)


def generate_graph_edges(
    group_number,
    router_topology,
    p2p_topology,
    num_edges_per_node,
    directed_graph=False,
    edge_direction="NODE_TO_NEIGHBORS",
):
    graph_edges_router = pd.DataFrame(columns=["NODE_1", "NODE_2"])
    graph_edges_p2p = pd.DataFrame(columns=["NODE_1", "NODE_2"])

    if router_topology != "NO_ROUTER":
        # get the nodes IDs
        benign_dataset = pd.read_parquet(
            PATH_CONFIG.get_benign_dataset_path(group_number)
        )
        nodes_list = benign_dataset["NODE"].unique()

        # Connect each node to the router
        for node in nodes_list:
            if directed_graph:
                if edge_direction == "NODE_TO_NEIGHBORS":
                    edges_tmp = pd.DataFrame(
                        {"NODE_1": node, "NODE_2": CONFIG.ROUTER_ID}, index=[0]
                    )
                elif edge_direction == "NEIGHBORS_TO_NODE":
                    edges_tmp = pd.DataFrame(
                        {"NODE_1": CONFIG.ROUTER_ID, "NODE_2": node}, index=[0]
                    )
                else:
                    raise ValueError(
                        "edge_direction should be either NODE_TO_NEIGHBORS or NEIGHBORS_TO_NODE"
                    )
            else:
                edges_tmp = pd.DataFrame(
                    {
                        "NODE_1": [node, CONFIG.ROUTER_ID],
                        "NODE_2": [CONFIG.ROUTER_ID, node],
                    }
                )
            graph_edges_router = pd.concat([graph_edges_router, edges_tmp])

    if p2p_topology != "NO_P2P":
        edges_dataset_path = PATH_CONFIG.get_metadata_metrics_path(
            group_number, p2p_topology, reassigned=False
        )
        all_edges_info = pd.read_parquet(edges_dataset_path)
        all_edges_info.sort_values(by=["NODE_1", p2p_topology], inplace=True)
        nodes = all_edges_info["NODE_1"].unique()
        for node_1 in nodes:
            # edges_tmp = []
            node_1_edges_df = all_edges_info.loc[all_edges_info["NODE_1"] == node_1]
            ascending_bool = True
            if p2p_topology == "DISTANCE":
                ascending_bool = True
            elif p2p_topology == "CORRELATION":
                ascending_bool = False
            node_1_edges_df = node_1_edges_df.sort_values(
                by=[p2p_topology], ascending=ascending_bool
            )
            node_1_edges = node_1_edges_df["NODE_2"].values

            for index in range(num_edges_per_node):
                node_pair = [node_1, node_1_edges[index]]
                if directed_graph:
                    if edge_direction == "NODE_TO_NEIGHBORS":
                        edges_tmp = pd.DataFrame(
                            {"NODE_1": node_pair[0], "NODE_2": node_pair[1]},
                            index=[index],
                        )
                    elif edge_direction == "NEIGHBORS_TO_NODE":
                        edges_tmp = pd.DataFrame(
                            {"NODE_1": node_pair[1], "NODE_2": node_pair[0]},
                            index=[index],
                        )
                    else:
                        raise ValueError(
                            "edge_direction should be either NODE_TO_NEIGHBORS or NEIGHBORS_TO_NODE"
                        )
                else:
                    edges_tmp = pd.DataFrame(
                        {
                            "NODE_1": [node_pair[0], node_pair[1]],
                            "NODE_2": [node_pair[1], node_pair[0]],
                        }
                    )
                graph_edges_p2p = pd.concat([graph_edges_p2p, edges_tmp])

    graph_edges = pd.concat([graph_edges_router, graph_edges_p2p])

    # remove duplicate rows
    # connecting each node to other nodes based on a metric may result in duplicate rows
    print("before removing", graph_edges.shape)
    graph_edges.drop_duplicates(inplace=True)
    print("after removing", graph_edges.shape)

    column_types = {"NODE_1": "int32", "NODE_2": "int32"}
    graph_edges = graph_edges.reset_index(drop=True)
    graph_edges = graph_edges.astype(column_types)

    output_path = PATH_CONFIG.get_graph_edges_path(
        group_number, router_topology, p2p_topology, reassigned=False
    )
    UTIL.save_dataframe(graph_edges, output_path)


def main_generate_graph_edges(
    group_number,
    router_topology,
    p2p_topology,
    num_edges_per_node,
    directed_graph,
    edge_direction,
):
    """The main function to be used for calling  combine_data function.

    Keyword arguments:
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    """

    generate_graph_edges(
        group_number,
        router_topology,
        p2p_topology,
        num_edges_per_node,
        directed_graph,
        edge_direction,
    )
    for data_type in ["train", "validation", "test"]:
        # Generate graph_edges for each data_type and consider all the edges_connection_ratio
        generate_all_graphs_edges(
            group_number, data_type, router_topology, p2p_topology, directed_graph
        )
        print(
            "Finished generating all graphs edges for group {} and data_type {} and router_topology {} and p2p_topology {}".format(
                group_number, data_type, router_topology, p2p_topology
            )
        )


def main():
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST
    num_edges_per_node = CONFIG.NUM_EDGES_PER_NODE
    directed_graph = CONFIG.DIRECTED_GRAPH
    edge_direction = CONFIG.EDGE_DIRECTION

    for group_number in range(CONFIG.NUM_GROUPS):
        for router_topology in router_topology_list:
            for p2p_topology in p2p_topology_list:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                main_generate_graph_edges(
                    group_number,
                    router_topology,
                    p2p_topology,
                    num_edges_per_node,
                    directed_graph,
                    edge_direction,
                )


if __name__ == "__main__":
    main()
