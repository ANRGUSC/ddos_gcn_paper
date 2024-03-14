import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# enable the following for debug
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
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
    precision_recall_fscore_support,
)
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import random
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
import source_code.utilities.utilities as UTIL
import source_code.gcn_models.gcn_utilities as GCN_UTIL
import source_code.utilities.nn_utilities as NN_UTIL
import source_code.config.path_config as PATH_CONFIG


def initialize_training(
    group_number, router_topology, p2p_topology, batch_size=CONFIG.NUM_BATCHES
):
    # Create the data loader
    train_dataset_loader = GCN_UTIL.create_data_loader(
        group_number,
        router_topology,
        p2p_topology,
        "train",
        batch_size,
        shuffle=True,
        use_normalized_features=True,
    )
    validation_dataset_loader = GCN_UTIL.create_data_loader(
        group_number,
        router_topology,
        p2p_topology,
        "validation",
        batch_size,
        shuffle=True,
        use_normalized_features=True,
    )

    device = NN_UTIL.get_device(use_cpu=False)

    # Create the model
    num_node_features = train_dataset_loader.dataset.num_node_features
    model, optimizer = GCN_UTIL.create_model(num_node_features, device)
    loss_fn = GCN_UTIL.create_loss_function(train_dataset_loader, device)

    return (
        train_dataset_loader,
        validation_dataset_loader,
        device,
        num_node_features,
        model,
        optimizer,
        loss_fn,
    )


def train_gcn(model, optimizer, loss_fn, batch):
    model.train()
    optimizer.zero_grad()
    out = model(batch)
    loss = loss_fn(out, batch.y)
    loss.backward()
    optimizer.step()


def main_train_gcn(group_number, router_topology, p2p_topology):
    batch_size = CONFIG.NUM_BATCHES
    (
        train_dataset_loader,
        validation_dataset_loader,
        device,
        num_node_features,
        model,
        optimizer,
        loss_fn,
    ) = initialize_training(group_number, router_topology, p2p_topology, batch_size)

    print("Start Training")
    # Train the model
    log = []
    highest_score = 0
    restart_training = False
    num_retruies = 0
    max_retries = 5
    epoch = 0
    while epoch < CONFIG.NUM_EPOCHS:
        if restart_training and num_retruies < max_retries:
            batch_size += 32
            (
                train_dataset_loader,
                validation_dataset_loader,
                device,
                num_node_features,
                model,
                optimizer,
                loss_fn,
            ) = initialize_training(
                group_number, router_topology, p2p_topology, batch_size
            )
            epoch = 0
            restart_training = False
            num_retruies += 1

        for batch in train_dataset_loader:
            batch.to(device)
            train_gcn(model, optimizer, loss_fn, batch)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            train_preds, train_labels = NN_UTIL.evaluate(
                model, train_dataset_loader, device
            )
            val_preds, val_labels = NN_UTIL.evaluate(
                model, validation_dataset_loader, device
            )

            train_metrics = NN_UTIL.calculate_metrics_training_phase(
                train_preds, train_labels, loss_fn
            )
            val_metrics = NN_UTIL.calculate_metrics_training_phase(
                val_preds, val_labels, loss_fn
            )
            metrics = {"train": train_metrics, "val": val_metrics}

            log_entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            print("epoch " + str(epoch) + " - train - " + str(train_metrics))
            print("epoch " + str(epoch) + " - valid - " + str(val_metrics))
            log.append(log_entry)

            # Check for UndefinedMetricWarning
            if any(
                isinstance(warning.message, UndefinedMetricWarning) for warning in w
            ):
                print("Caught UndefinedMetricWarning. Restarting training.")
                restart_training = True

        GCN_UTIL.save_best_model_fn(
            highest_score,
            model,
            optimizer,
            epoch,
            metrics,
            group_number,
            router_topology,
            p2p_topology,
        )
        GCN_UTIL.save_model_each_epoch_fn(
            model,
            optimizer,
            epoch,
            metrics,
            group_number,
            router_topology,
            p2p_topology,
        )
        epoch += 1

    log_df = pd.DataFrame(log)
    GCN_UTIL.save_logs_fn(group_number, router_topology, p2p_topology, log_df)
    GCN_UTIL.plot_log_df(group_number, router_topology, p2p_topology, log_df)


def main(num_groups):
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST

    for group_number in range(num_groups):
        for router_topology in router_topology_list:
            for p2p_topology in p2p_topology_list:
                if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                    continue
                print(
                    "Group Number: ",
                    group_number,
                    " -- Router Topology: ",
                    router_topology,
                    " -- P2P Topology: ",
                    p2p_topology,
                )
                main_train_gcn(group_number, router_topology, p2p_topology)


if __name__ == "__main__":
    main(CONFIG.NUM_GROUPS)
