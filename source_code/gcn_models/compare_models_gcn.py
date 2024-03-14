import sys
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
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
import seaborn as sns
import numpy as np

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


label_dict = {
    "binary_accuracy": "Binary Accuracy",
    "f1_score": "F1 Score",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "TP": "True Positive",
    "FP": "False Positive",
    "TN": "True Negative",
    "FN": "False Negative",
    "PREDICTION_PROBABILITY": "Prediction Probability",
    "LABEL": "Label",
    "ATTACK_PARAMETER": "Attack Packet Volume Distribution Parameter ($k$)",
}


def combine_confusion_matrices(router_topology, p2p_topology, data_type):
    confusion_matrix = pd.DataFrame()
    for group_number in range(CONFIG.NUM_GROUPS):
        df_path = PATH_CONFIG.get_confusion_matrix_path(
            group_number, router_topology, p2p_topology, data_type
        )
        tmp_df = pd.read_parquet(df_path)
        tmp_df["GROUP"] = group_number
        confusion_matrix = pd.concat([confusion_matrix, tmp_df])
    return confusion_matrix


def generate_aggregated_metrics_using_confusion_matrix(
    router_topology, p2p_topology, data_type, aggregate_parameters_list
):
    confusion_matrix = combine_confusion_matrices(
        router_topology, p2p_topology, data_type
    )
    # Group by the specified column and calculate the sum of 'TP', 'FP', 'TN', 'FN'
    grouped_data = confusion_matrix.groupby(aggregate_parameters_list).agg(
        {"TP": "sum", "FP": "sum", "TN": "sum", "FN": "sum"}
    )

    # Calculate the metrics
    grouped_data["binary_accuracy"] = (grouped_data["TP"] + grouped_data["TN"]) / (
        grouped_data["TP"]
        + grouped_data["FP"]
        + grouped_data["TN"]
        + grouped_data["FN"]
    )
    grouped_data["precision"] = grouped_data["TP"] / (
        grouped_data["TP"] + grouped_data["FP"] + sys.float_info.epsilon
    )
    grouped_data["recall"] = grouped_data["TP"] / (
        grouped_data["TP"] + grouped_data["FN"] + sys.float_info.epsilon
    )
    grouped_data["specificity"] = grouped_data["TN"] / (
        grouped_data["TN"] + grouped_data["FP"] + sys.float_info.epsilon
    )
    grouped_data["f1_score"] = (
        2
        * (grouped_data["precision"] * grouped_data["recall"])
        / (grouped_data["precision"] + grouped_data["recall"] + sys.float_info.epsilon)
    )

    # Calculate ROC AUC score for each group
    roc_auc_scores = []
    for group in grouped_data.index:
        query_string = " & ".join(
            [f"{col} == {val!r}" for col, val in zip(aggregate_parameters_list, group)]
        )
        filtered_df = confusion_matrix.query(query_string)
        true_labels = filtered_df["LABEL"]
        predicted_probabilities = filtered_df["PREDICTION_PROBABILITY"]
        try:
            roc_auc = roc_auc_score(true_labels, predicted_probabilities)
        except ValueError:
            roc_auc = 1
        roc_auc_scores.append(roc_auc)
    grouped_data["auc"] = roc_auc_scores

    # Reset the index
    grouped_data.reset_index(inplace=True)
    output_path = PATH_CONFIG.get_model_analysis_aggregated_report_data_path(
        router_topology, p2p_topology, data_type, aggregate_parameters_list
    )
    UTIL.save_dataframe(grouped_data, output_path, save_csv=True)
    # print(grouped_data)
    return grouped_data


def get_datasets_for_plot(
    filter_dataset_dict,
    aggregate_parameters_list,
    each_curve_parameters_list,
):
    """Get the datasets for plotting the curves
    Seprate the data into groups based on the parameters in each_curve_parameters_list
    Extract the columns to plot

    Args:
        data_type: 'train' or 'test'
        aggregate_parameters_list: list of parameters to aggregate the data
        each_curve_parameters_list: list of parameters to plot each curve
    Returns:
        df_groups: grouped dataframes
        columns_to_plot: list of columns to plot
    """
    dataset_df_path = (
        PATH_CONFIG.get_model_analysis_aggregated_report_data_combined_path(
            aggregate_parameters_list
        )
    )
    dataset_df_path = dataset_df_path.replace(".parquet", ".csv")
    dataset_df = pd.read_csv(dataset_df_path)

    # Remove the scnarios with no p2p topology for the directed graphs
    if CONFIG.DIRECTED_GRAPH:
        dataset_df = dataset_df.loc[dataset_df["P2P_TOPOLOGY"] != "NO_P2P"]

    for key, value in filter_dataset_dict.items():
        dataset_df = dataset_df.loc[dataset_df[key] == value]

    # Change the EDGES_CONNECTION_RATIO to represent the connection loss
    if "EDGES_CONNECTION_RATIO" in each_curve_parameters_list:
        dataset_df["EDGES_CONNECTION_RATIO"] = (
            1 - dataset_df["EDGES_CONNECTION_RATIO"]
        ).round(1)
    df_groups = dataset_df.groupby(each_curve_parameters_list)

    return df_groups


def plot_confidence_intervals(
    filter_dataset_dict,
    aggregate_parameters_list,
    confidence_parameter,
    aggregate_metric,
    each_curve_parameters_list,
    columns_to_plot,
):
    # Get the grouped data and columns to plot
    df_groups_each_plot = get_datasets_for_plot(
        filter_dataset_dict,
        aggregate_parameters_list,
        each_curve_parameters_list,
    )

    # Creating each plot based on the metric
    for col in columns_to_plot:
        plt.figure()

        # Plot the confidence intervals for each group
        for index, group in df_groups_each_plot:
            # check if index is a tuple
            if not isinstance(index, tuple):
                index = [index]
            index = [str(i).capitalize() for i in index]
            dataset_df = (
                group.groupby([confidence_parameter, aggregate_metric])[col]
                .mean()
                .reset_index()
            )
            # Group the dataset by the current column and aggregate_metric, then calculate the mean and standard deviation
            grouped_data = (
                dataset_df.groupby([aggregate_metric])[col]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Plot the mean and confidence intervals
            if len(index) == 1:
                # define the lable to show loss
                label = "$l$=" + index[0]
            else:
                label = "_".join(index)

            sns.lineplot(
                x=aggregate_metric,
                y="mean",
                data=grouped_data,
                marker="o",
                markersize=5,
                label=label,
            )

            # Calculate the confidence intervals
            lower_bound = grouped_data["mean"] - 1.96 * grouped_data["std"] / np.sqrt(
                grouped_data["mean"].count()
            )
            upper_bound = grouped_data["mean"] + 1.96 * grouped_data["std"] / np.sqrt(
                grouped_data["mean"].count()
            )

            # Plot the confidence intervals
            plt.fill_between(
                grouped_data[aggregate_metric], lower_bound, upper_bound, alpha=0.1
            )

        # Set the title and labels
        # plt.title(f"{col} vs {aggregate_metric} with Confidence Intervals")
        plt.xlabel(label_dict[aggregate_metric])
        plt.ylabel(label_dict[col])
        plt.legend()
        plot_path = PATH_CONFIG.get_model_analysis_confidence_interval_plot_path(
            filter_dataset_dict,
            aggregate_parameters_list,
            confidence_parameter,
            aggregate_metric,
            plot_metric=col,
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def plot_error_bar(
    filter_dataset_dict,
    aggregate_parameters_list,
    confidence_parameter,
    aggregate_metric,
    each_curve_parameters_list,
    columns_to_plot,
):
    # Get the grouped data and columns to plot
    df_groups_each_plot = get_datasets_for_plot(
        filter_dataset_dict,
        aggregate_parameters_list,
        each_curve_parameters_list,
    )

    # Creating each plot based on the metric
    for col in columns_to_plot:
        plt.figure()

        for index, group in df_groups_each_plot:
            # check if index is a tuple
            if not isinstance(index, tuple):
                index = [index]
            index = [str(i).capitalize() for i in index]
            dataset_df = (
                group.groupby([confidence_parameter, aggregate_metric])[col]
                .mean()
                .reset_index()
            )
            # Group the dataset by the current column and aggregate_metric, then calculate the mean and standard deviation
            grouped_data = (
                dataset_df.groupby([aggregate_metric])[col]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Plot the mean with error bars
            if len(index) == 1:
                # define the lable to show loss
                label = "$l$=" + index[0]
            else:
                label = "_".join(index)
            plt.errorbar(
                x=grouped_data[aggregate_metric],
                y=grouped_data["mean"],
                yerr=1.96 * grouped_data["std"] / np.sqrt(grouped_data["mean"].count()),
                marker="o",
                markersize=5,
                label=label,
                # capsize=3,
            )

        # Set the title and labels
        # plt.title(f"{col} vs {aggregate_metric} with Confidence Intervals")
        plt.xlabel(label_dict[aggregate_metric])
        plt.ylabel(label_dict[col])
        plt.legend()
        plot_path = PATH_CONFIG.get_model_analysis_error_bar_plot_path(
            filter_dataset_dict,
            aggregate_parameters_list,
            confidence_parameter,
            aggregate_metric,
            plot_metric=col,
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def main_generate_data(aggregate_parameters_list):
    df_combined = pd.DataFrame()
    for router_topology in CONFIG.ROUTER_TOPOLOGY_LIST:
        for p2p_topology in CONFIG.P2P_TOPOLOGY_LIST:
            if router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P":
                continue
            for data_type in ["train", "test"]:
                aggregated_metrics_df = (
                    generate_aggregated_metrics_using_confusion_matrix(
                        router_topology,
                        p2p_topology,
                        data_type,
                        aggregate_parameters_list,
                    )
                )
                aggregated_metrics_df["ROUTER_TOPOLOGY"] = router_topology
                aggregated_metrics_df["P2P_TOPOLOGY"] = p2p_topology
                aggregated_metrics_df["DATA_TYPE"] = data_type
                df_combined = pd.concat([df_combined, aggregated_metrics_df])

    output_path = PATH_CONFIG.get_model_analysis_aggregated_report_data_combined_path(
        aggregate_parameters_list
    )
    UTIL.save_dataframe(df_combined, output_path, save_csv=True)


def main_generate_plots(
    aggregate_parameters_list,
    filter_dataset_dict,
    confidence_parameter,
    aggregate_metric,
    each_curve_parameters_list,
    columns_to_plot,
):
    for data_type in ["train", "test"]:
        filter_dataset_dict["DATA_TYPE"] = data_type
        plot_confidence_intervals(
            filter_dataset_dict,
            aggregate_parameters_list,
            confidence_parameter,
            aggregate_metric,
            each_curve_parameters_list,
            columns_to_plot,
        )
        plot_error_bar(
            filter_dataset_dict,
            aggregate_parameters_list,
            confidence_parameter,
            aggregate_metric,
            each_curve_parameters_list,
            columns_to_plot,
        )


def main_plot_all_together(
    aggregate_parameters_list, confidence_parameter, aggregate_metric, columns_to_plot
):
    # Define the parameters to be used for plotting each curve on a figure
    each_curve_parameters_list = [
        "ROUTER_TOPOLOGY",
        "P2P_TOPOLOGY",
        "EDGES_CONNECTION_RATIO",
    ]

    filter_dataset_dict = {}
    main_generate_plots(
        aggregate_parameters_list,
        filter_dataset_dict,
        confidence_parameter,
        aggregate_metric,
        each_curve_parameters_list,
        columns_to_plot,
    )


def main_plot_router_only(
    aggregate_parameters_list, confidence_parameter, aggregate_metric, columns_to_plot
):
    # Define the parameters to be used for plotting each curve on a figure
    each_curve_parameters_list = [
        # "ROUTER_TOPOLOGY",
        "P2P_TOPOLOGY",
        "EDGES_CONNECTION_RATIO",
    ]

    filter_dataset_dict = {"ROUTER_TOPOLOGY": "ROUTER"}
    main_generate_plots(
        aggregate_parameters_list,
        filter_dataset_dict,
        confidence_parameter,
        aggregate_metric,
        each_curve_parameters_list,
        columns_to_plot,
    )


def main_plot_p2p_only(
    aggregate_parameters_list, confidence_parameter, aggregate_metric, columns_to_plot
):
    # Define the parameters to be used for plotting each curve on a figure
    each_curve_parameters_list = [
        # "ROUTER_TOPOLOGY",
        "P2P_TOPOLOGY",
        "EDGES_CONNECTION_RATIO",
    ]

    filter_dataset_dict = {"ROUTER_TOPOLOGY": "NO_ROUTER"}
    main_generate_plots(
        aggregate_parameters_list,
        filter_dataset_dict,
        confidence_parameter,
        aggregate_metric,
        each_curve_parameters_list,
        columns_to_plot,
    )


def main_plot_separate(
    aggregate_parameters_list, confidence_parameter, aggregate_metric, columns_to_plot
):
    # Define the parameters to be used for plotting each curve on a figure
    each_curve_parameters_list = [
        # "ROUTER_TOPOLOGY",
        # "P2P_TOPOLOGY",
        "EDGES_CONNECTION_RATIO"
    ]

    filter_dataset_dict = {}
    router_topology_list = CONFIG.ROUTER_TOPOLOGY_LIST
    p2p_topology_list = CONFIG.P2P_TOPOLOGY_LIST
    for router_topology in router_topology_list:
        for p2p_topology in p2p_topology_list:
            if (router_topology == "NO_ROUTER" and p2p_topology == "NO_P2P") or (
                CONFIG.DIRECTED_GRAPH and p2p_topology == "NO_P2P"
            ):
                continue
            filter_dataset_dict["ROUTER_TOPOLOGY"] = router_topology
            filter_dataset_dict["P2P_TOPOLOGY"] = p2p_topology
            main_generate_plots(
                aggregate_parameters_list,
                filter_dataset_dict,
                confidence_parameter,
                aggregate_metric,
                each_curve_parameters_list,
                columns_to_plot,
            )


def main():
    # paramters to aggregate over for creating the corresponding confusion matrix for the purpose of plotting
    aggregate_parameters_list = ["GROUP", "ATTACK_PARAMETER", "EDGES_CONNECTION_RATIO"]
    main_generate_data(aggregate_parameters_list)

    # paramter to help with plotting the confidence intervals
    confidence_parameter = "GROUP"

    # parameter to plot the confidence intervals for, i.e. the x-axis
    aggregate_metric = "ATTACK_PARAMETER"

    columns_to_plot = [
        "TP",
        "FP",
        "TN",
        "FN",
        "binary_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "auc",
    ]

    main_plot_all_together(
        aggregate_parameters_list,
        confidence_parameter,
        aggregate_metric,
        columns_to_plot,
    )
    main_plot_router_only(
        aggregate_parameters_list,
        confidence_parameter,
        aggregate_metric,
        columns_to_plot,
    )
    main_plot_p2p_only(
        aggregate_parameters_list,
        confidence_parameter,
        aggregate_metric,
        columns_to_plot,
    )
    main_plot_separate(
        aggregate_parameters_list,
        confidence_parameter,
        aggregate_metric,
        columns_to_plot,
    )


if __name__ == "__main__":
    main()
