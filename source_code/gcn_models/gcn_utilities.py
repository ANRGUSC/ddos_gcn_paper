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
    roc_auc_score,
    roc_curve,
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
import matplotlib.dates as mdates
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
import source_code.utilities.nn_utilities as NN_UTIL


# Custom dataset class
class GraphDataset_old(Dataset):
    def __init__(
        self,
        edge_index,
        node_features_list,
        node_labels_list,
        transform=None,
        pre_transform=None,
    ):
        self.edge_index = edge_index
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        # self.weights = self.compute_weights()
        super(GraphDataset, self).__init__(None, transform, pre_transform)

    def __len__(self):
        return len(self.node_features_list)

    def num_node_features(self) -> int:
        return self.node_features_list[0].shape[1]

    def __getitem__(self, idx):
        x = self.node_features_list[idx]
        y = self.node_labels_list[idx]

        return Data(x=x, edge_index=self.edge_index, y=y)

    def compute_weights(self):
        y = []
        for i in range(len(self.node_labels_list)):
            y += (
                self.node_labels_list[i]
                .reshape(self.node_labels_list[i].shape[0])
                .tolist()
            )
        n_samples = len(y)
        n_positive = sum(y)
        n_negative = n_samples - n_positive
        weight_positive = n_negative / n_positive
        weight_negative = 1.0
        return torch.tensor([weight_negative, weight_positive], dtype=torch.float)

    def get_weighted_sampler(self):
        y = []
        for i in range(len(self.node_labels_list)):
            y += (
                self.node_labels_list[i]
                .reshape(self.node_labels_list[i].shape[0])
                .tolist()
            )
        class_counts = [len(y) - sum(y), sum(y)]
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[int(t)] for t in y]
        return WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )


class GraphDataset(Dataset):
    def __init__(self, train_dataset, graph_edges, transform=None, pre_transform=None):
        (
            self.all_columns,
            self.index_columns,
            self.feature_columns,
            self.label_column,
            self.node_id_column,
        ) = UTIL.extract_dataset_columns(train_dataset)

        # sort train_dataset and graph_edges by self.index_columns ascending
        tmp_column = self.index_columns + self.node_id_column
        train_dataset = train_dataset.sort_values(by=tmp_column, ascending=True)
        graph_edges = graph_edges.sort_values(by=self.index_columns, ascending=True)

        self.complete_dataset_length = train_dataset.shape[0]
        grouped_train_dataset = train_dataset.groupby(
            self.index_columns, group_keys=False
        )
        grouped_graph_edges = graph_edges.groupby(self.index_columns, group_keys=False)

        num_graphs = len(grouped_train_dataset)
        self.graph_data_list = [None] * num_graphs

        for idx, (train_group, edges_group) in enumerate(
            zip(grouped_train_dataset, grouped_graph_edges)
        ):
            # Get the node features
            x = torch.tensor(
                train_group[1][self.feature_columns].values, dtype=torch.float
            )

            # Get the node labels
            y = torch.tensor(
                train_group[1][self.label_column].values, dtype=torch.float
            )

            # Create the edge_index tensor from the graph_edges_data
            edge_index = (
                torch.tensor(
                    edges_group[1][["NODE_1", "NODE_2"]].values, dtype=torch.long
                )
                .t()
                .contiguous()
            )

            self.graph_data_list[idx] = Data(x=x, edge_index=edge_index, y=y)

        super(GraphDataset, self).__init__(transform, pre_transform)

    def __len__(self):
        return len(self.graph_data_list)

    def __getitem__(self, idx):
        return self.graph_data_list[idx]

    get = __getitem__
    len = __len__


def create_data_loader(
    group_number,
    router_topology,
    p2p_topology,
    data_type,
    batch_size=32,
    shuffle=True,
    use_normalized_features=True,
    return_dataset_df=False,
):
    graph_dataset_path = PATH_CONFIG.get_dataset_train_eval_path(
        group_number,
        router_topology,
        p2p_topology,
        data_type,
        reassigned=True,
        normalized=use_normalized_features,
    )
    graph_dataset_df = pd.read_parquet(graph_dataset_path)
    graph_edges_path = PATH_CONFIG.get_all_graphs_edges_path(
        group_number, router_topology, p2p_topology, data_type, reassigned=True
    )
    graph_edges_df = pd.read_parquet(graph_edges_path)
    (
        all_columns,
        index_columns,
        feature_columns,
        label_column,
        node_id_column,
    ) = UTIL.extract_dataset_columns(graph_dataset_df)
    graph_dataset_df.sort_values(by=index_columns + node_id_column, inplace=True)

    dataset = GraphDataset(graph_dataset_df, graph_edges_df)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    if return_dataset_df:
        return data_loader, graph_dataset_df
    else:
        return data_loader


# Define the GNN model
class GCN_1layer(torch.nn.Module):
    def __init__(self, num_node_features, num_classes=1):
        super(GCN_1layer, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return x


class GCN_2layer(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class GCN_3layer(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        super(GCN_3layer, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)


def create_model(num_node_features, device):
    hidden_channels = 1024
    model = GCN_2layer(num_node_features, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


# Create the loss function given the class imbalance
def create_loss_function(train_dataset_loader, device):
    train_labels = torch.cat([data.y for data in train_dataset_loader], dim=0)
    class_counts = torch.tensor(
        [(train_labels == c).sum().item() for c in range(2)], dtype=torch.float
    )
    class_weights = (1 / class_counts) * class_counts.sum() / 2
    pos_weight = class_weights[1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn


def load_trained_model(
    group_number, router_topology, p2p_topology, dataset_loader, device
):
    num_node_features = dataset_loader.dataset.num_node_features
    model, optimizer = create_model(num_node_features, device)
    trained_model_path = PATH_CONFIG.get_best_model_path(
        group_number, router_topology, p2p_topology, "val", "f1_score"
    )
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def save_best_model_fn(
    highest_score,
    model,
    optimizer,
    epoch,
    metrics,
    group_number,
    router_topology,
    p2p_topology,
):
    save_best_model = CONFIG.SAVE_BEST_MODEL
    model_output_path = PATH_CONFIG.get_best_model_path(
        group_number,
        router_topology,
        p2p_topology,
        save_best_model["data"],
        save_best_model["metric"],
    )
    NN_UTIL.save_best_model_fn(
        highest_score, model, optimizer, epoch, metrics, model_output_path
    )


def save_model_each_epoch_fn(
    model,
    optimizer,
    epoch,
    metrics,
    group_number,
    router_topology,
    p2p_topology,
):
    model_output_path = PATH_CONFIG.get_each_epoch_save_model_path(
        group_number, router_topology, p2p_topology, epoch
    )
    NN_UTIL.save_model_each_epoch_fn(
        model,
        optimizer,
        epoch,
        metrics,
        model_output_path,
    )


def save_logs_fn(group_number, router_topology, p2p_topology, log_df):
    logs_output_path = PATH_CONFIG.get_training_logs_path(
        group_number, router_topology, p2p_topology
    )
    NN_UTIL.save_logs_fn(log_df, logs_output_path)


def plot_log_df(group_number, router_topology, p2p_topology, log_df):
    plot_output_path_directory = PATH_CONFIG.get_training_logs_plot_path_directory(
        group_number, router_topology, p2p_topology
    )
    NN_UTIL.plot_log_df(log_df, plot_output_path_directory)


# %% Generate Results


def generate_trained_model_predictions(
    group_number,
    router_topology,
    p2p_topology,
    data_type,
    model,
    dataset_df,
    dataset_loader,
    device,
):
    output_path = PATH_CONFIG.get_prediction_probability_dataset_path(
        group_number, router_topology, p2p_topology, data_type
    )
    NN_UTIL.generate_trained_model_predictions(
        model, dataset_df, dataset_loader, device, output_path
    )


def find_optimal_threshold(group_number, router_topology, p2p_topology, data_type):
    pred_prob_df_path = PATH_CONFIG.get_prediction_probability_dataset_path(
        group_number, router_topology, p2p_topology, data_type
    )
    optimal_threshold_save_path = PATH_CONFIG.get_optimal_threshold_path(
        group_number, router_topology, p2p_topology
    )

    return NN_UTIL.find_optimal_threshold(
        pred_prob_df_path, optimal_threshold_save_path
    )


def generate_confusion_matrix_data(
    group_number, router_topology, p2p_topology, data_type, threshold=0.5
):
    prediction_probability_dataset_path = (
        PATH_CONFIG.get_prediction_probability_dataset_path(
            group_number, router_topology, p2p_topology, data_type
        )
    )
    output_path = PATH_CONFIG.get_confusion_matrix_path(
        group_number, router_topology, p2p_topology, data_type
    )

    NN_UTIL.generate_confusion_matrix_data(
        prediction_probability_dataset_path, output_path, threshold
    )


def generate_aggregated_metrics_using_confusion_matrix(
    group_number, router_topology, p2p_topology, data_type, aggregate_parameter
):
    confusion_matrix_path = PATH_CONFIG.get_confusion_matrix_path(
        group_number, router_topology, p2p_topology, data_type
    )
    output_path = PATH_CONFIG.get_aggregated_report_data(
        group_number, router_topology, p2p_topology, data_type, aggregate_parameter
    )
    NN_UTIL.generate_aggregated_metrics_using_confusion_matrix(
        aggregate_parameter, confusion_matrix_path, output_path
    )


def generate_roc_data(group_number, router_topology, p2p_topology, data_type):
    pred_prob_df_path = PATH_CONFIG.get_prediction_probability_dataset_path(
        group_number, router_topology, p2p_topology, data_type
    )
    output_path = PATH_CONFIG.get_roc_data_path(
        group_number, router_topology, p2p_topology, data_type
    )
    NN_UTIL.generate_roc_data(pred_prob_df_path, output_path)


def plot_metrics(group_number, router_topology, p2p_topology, aggregate_parameter):
    train_dataset_df = pd.read_parquet(
        PATH_CONFIG.get_aggregated_report_data(
            group_number, router_topology, p2p_topology, "train", aggregate_parameter
        )
    )
    test_dataset_df = pd.read_parquet(
        PATH_CONFIG.get_aggregated_report_data(
            group_number, router_topology, p2p_topology, "test", aggregate_parameter
        )
    )
    output_path_directory = PATH_CONFIG.get_aggregated_report_plot_directory(
        group_number, router_topology, p2p_topology
    )
    NN_UTIL.plot_metrics(
        train_dataset_df, test_dataset_df, output_path_directory, aggregate_parameter
    )


def plot_roc(group_number, router_topology, p2p_topology):
    train_roc_data_path = PATH_CONFIG.get_roc_data_path(
        group_number, router_topology, p2p_topology, "train"
    )
    test_roc_data_path = PATH_CONFIG.get_roc_data_path(
        group_number, router_topology, p2p_topology, "test"
    )
    output_path = PATH_CONFIG.get_roc_plot_path(
        group_number, router_topology, p2p_topology
    )
    NN_UTIL.plot_roc(train_roc_data_path, test_roc_data_path, output_path)


def plot_attack_prediction_vs_time(
    group_number, router_topology, p2p_topology, data_type
):
    dataset_path = PATH_CONFIG.get_confusion_matrix_path(
        group_number, router_topology, p2p_topology, data_type
    )
    output_path = PATH_CONFIG.get_attack_prediction_vs_time_plot_path(
        group_number, router_topology, p2p_topology, data_type, average_all=False
    )
    NN_UTIL.plot_attack_prediction_vs_time(dataset_path, output_path)
