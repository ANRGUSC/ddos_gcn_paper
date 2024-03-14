import pandas as pd
import os
import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


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
    "NUM_EDGES_PER_NODE": "Number of Edges per Node",
}


def concat_csv_files(input_path):
    # Ensure the path ends with a slash
    if not input_path.endswith("/"):
        input_path += "/"

    # List all CSV files in the directory
    csv_files = [file for file in os.listdir(input_path) if file.endswith(".csv")]

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through all CSV files, read them, and append to the list
    for file in csv_files:
        df = pd.read_csv(input_path + file)
        dataframes.append(df)

    # Concatenate all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df = (
        combined_df.groupby(
            [
                "GROUP",
                "ROUTER_TOPOLOGY",
                "P2P_TOPOLOGY",
                "NUM_EDGES_PER_NODE",
                "DATA_TYPE",
            ]
        )
        .mean()
        .reset_index()
    )

    # Output the concatenated dataframe to a new CSV file
    combined_df.to_csv(input_path + "edges_combined.csv", index=False)

    print(f"Combined CSV created at: {input_path}edges_combined.csv")


def get_datasets_for_plot(
    filter_dataset_dict,
    aggregate_parameters_list,
    each_curve_parameters_list,
):
    """Get the datasets for plotting the curves
    Seprate the data into groups based on the parameters in each_curve_parameters_list
    Extract the columns to plot
    """
    dataset_df_path = "path_to_combined_csv_file"
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

            # Convert aggregate_metric to string for categorical x-axis
            grouped_data[aggregate_metric] = grouped_data[aggregate_metric].astype(str)

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

            # # Plot the confidence intervals
            # plt.fill_between(
            #     grouped_data[aggregate_metric], lower_bound, upper_bound, alpha=0.1
            # )

            # Since x-axis is categorical, we use positions based on index for fill_between
            plt.fill_between(
                range(len(grouped_data[aggregate_metric])),
                lower_bound,
                upper_bound,
                alpha=0.1,
            )

        # Set the title and labels
        # plt.title(f"{col} vs {aggregate_metric} with Confidence Intervals")
        plt.xlabel(label_dict[aggregate_metric])
        plt.ylabel(label_dict[col])
        plt.legend()
        plot_path = PATH_CONFIG.get_edge_analysis_confidence_interval_plot_path(
            filter_dataset_dict,
            aggregate_parameters_list,
            confidence_parameter,
            aggregate_metric,
            plot_metric=col,
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


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


def main_plot_p2p_only(
    aggregate_parameters_list, confidence_parameter, aggregate_metric, columns_to_plot
):
    # Define the parameters to be used for plotting each curve on a figure
    each_curve_parameters_list = [
        # "ROUTER_TOPOLOGY",
        "P2P_TOPOLOGY",
        # "EDGES_CONNECTION_RATIO",
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


def main_plots():
    # paramters to aggregate over for creating the corresponding confusion matrix for the purpose of plotting
    aggregate_parameters_list = ["GROUP", "EDGES_CONNECTION_RATIO"]

    # paramter to help with plotting the confidence intervals
    confidence_parameter = "GROUP"

    # parameter to plot the confidence intervals for, i.e. the x-axis
    aggregate_metric = "NUM_EDGES_PER_NODE"

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

    main_plot_p2p_only(
        aggregate_parameters_list,
        confidence_parameter,
        aggregate_metric,
        columns_to_plot,
    )


def main():
    input_path = "path_to_csv_files_containing_the_results_of_the_experiments"
    concat_csv_files(input_path)

    main_plots()


if __name__ == "__main__":
    main()
