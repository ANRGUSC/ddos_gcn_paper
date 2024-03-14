import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import WeightedRandomSampler

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import gc

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL
import source_code.gcn_models.gcn_utilities as GCN_UTIL
import source_code.utilities.nn_utilities as NN_UTIL


def main_generate_results_data(
    group_number, router_topology, p2p_topology, data_type, aggregate_parameter
):
    dataset_loader, dataset_df = GCN_UTIL.create_data_loader(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        batch_size=CONFIG.NUM_BATCHES,
        shuffle=False,
        use_normalized_features=True,
        return_dataset_df=True,
    )

    device = NN_UTIL.get_device(use_cpu=False)
    model = GCN_UTIL.load_trained_model(
        group_number, router_topology, p2p_topology, dataset_loader, device
    )

    GCN_UTIL.generate_trained_model_predictions(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        model,
        dataset_df,
        dataset_loader,
        device,
    )
    if data_type == "train":
        threshold = GCN_UTIL.find_optimal_threshold(
            group_number, router_topology, p2p_topology, data_type
        )
    else:
        optimal_threshold_path = PATH_CONFIG.get_optimal_threshold_path(
            group_number, router_topology, p2p_topology
        )
        threshold = UTIL.load_float_from_file(optimal_threshold_path)

    GCN_UTIL.generate_confusion_matrix_data(
        group_number, router_topology, p2p_topology, data_type, threshold
    )
    GCN_UTIL.generate_aggregated_metrics_using_confusion_matrix(
        group_number, router_topology, p2p_topology, data_type, aggregate_parameter
    )

    GCN_UTIL.generate_roc_data(group_number, router_topology, p2p_topology, data_type)


def main_generate_results_plots(
    group_number, router_topology, p2p_topology, aggregate_parameter
):
    GCN_UTIL.plot_metrics(
        group_number, router_topology, p2p_topology, aggregate_parameter
    )
    GCN_UTIL.plot_roc(group_number, router_topology, p2p_topology)
    GCN_UTIL.plot_attack_prediction_vs_time(
        group_number, router_topology, p2p_topology, "train"
    )
    GCN_UTIL.plot_attack_prediction_vs_time(
        group_number, router_topology, p2p_topology, "test"
    )


def main(num_groups):
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST
    aggregate_parameter = "ATTACK_PARAMETER"
    for group_number in range(num_groups):
        for router_topology in router_topology_list:
            for p2p_topology in p2p_topology_list:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                for data_type in ["train", "test"]:
                    print(
                        "group_number: ",
                        group_number,
                        " -- router_topology: ",
                        router_topology,
                        " -- p2p_topology: ",
                        p2p_topology,
                        " -- data_type: ",
                        data_type,
                    )
                    main_generate_results_data(
                        group_number,
                        router_topology,
                        p2p_topology,
                        data_type,
                        aggregate_parameter,
                    )
                main_generate_results_plots(
                    group_number, router_topology, p2p_topology, aggregate_parameter
                )


if __name__ == "__main__":
    main(CONFIG.NUM_GROUPS)
