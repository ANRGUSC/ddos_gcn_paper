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


def get_device(use_cpu=False):
    if use_cpu:
        return torch.device("cpu")
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %% Training functions


def evaluate(model, loader, device):
    model.eval()

    # calculate the number of samples in the loader considering the batch size of the last batch
    n_samples = loader.dataset.complete_dataset_length

    all_preds = torch.zeros(n_samples, dtype=torch.float, device=device)
    all_labels = torch.zeros(n_samples, dtype=torch.float, device=device)

    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze()

            num_samples = batch.num_nodes
            all_preds[idx : idx + num_samples] = pred
            all_labels[idx : idx + num_samples] = batch.y.squeeze()
            idx += num_samples

    return all_preds.cpu(), all_labels.cpu()


def calculate_metrics_training_phase(preds, labels, loss_fn, threshold=0.5):
    loss_fn_copy = loss_fn
    loss_fn_copy.to("cpu")
    loss = loss_fn_copy(preds, labels)
    preds = (preds > threshold).float()
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    specificity = tn / (tn + fp)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "loss": loss.item(),
    }


def save_best_model_fn(
    highest_score, model, optimizer, epoch, metrics, model_output_path
):
    save_best_model = CONFIG.SAVE_BEST_MODEL
    if save_best_model["enabled"]:
        metric = metrics[save_best_model["data"]][save_best_model["metric"]]
        if save_best_model["condition"] == "max":
            if metric >= highest_score:
                highest_score = metric
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    model_output_path,
                )
        elif save_best_model["condition"] == "min":
            if metric <= highest_score:
                highest_score = metric
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    model_output_path,
                )
        else:
            raise Exception(
                'Invalid save_best_model["condition"] value. It should be either "max" or "min"'
            )


def save_model_each_epoch_fn(model, optimizer, epoch, metrics, model_output_path):
    if CONFIG.SAVE_MODEL_EACH_EPOCH:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            model_output_path,
        )


def save_logs_fn(log_df, logs_output_path):
    UTIL.save_dataframe(log_df, logs_output_path)
    return log_df


def plot_log_df(log_df, plot_output_path_directory):
    # Extract unique keys from the 'train' dictionary
    keys = set()
    for _, row in log_df.iterrows():
        keys.update(row["train"].keys())

    for key in keys:
        # Initialize a figure and axis
        fig, ax = plt.subplots()

        # Plot 'train' data
        train_data = [row["train"][key] for _, row in log_df.iterrows()]
        ax.plot(log_df["epoch"], train_data, label=f"Train {key}")

        # Plot 'val' data
        val_data = [row["val"][key] for _, row in log_df.iterrows()]
        ax.plot(log_df["epoch"], val_data, label=f"Val {key}")

        # Set the labels and legend
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs. Epoch")
        ax.legend()

        # Show the plot
        output_path = plot_output_path_directory + "logs_" + key + ".png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


# %% Generate Results


def add_confusion_matrix_columns(dataset_df):
    labels = dataset_df["LABEL"].values
    predictions = dataset_df["PREDICTION"].values
    labels = [labels == 0, labels == 1]
    predictions = [predictions == 0, predictions == 1]

    tp = labels[1] & predictions[1]
    fp = labels[0] & predictions[1]
    tn = labels[0] & predictions[0]
    fn = labels[1] & predictions[0]

    dataset_df["TP"] = tp.astype(int)
    dataset_df["FP"] = fp.astype(int)
    dataset_df["TN"] = tn.astype(int)
    dataset_df["FN"] = fn.astype(int)

    return dataset_df


def generate_trained_model_predictions(
    model, dataset_df, dataset_loader, device, output_path
):
    all_preds, all_labels = evaluate(model, dataset_loader, device)
    dataset_df["LABEL"] = all_labels
    dataset_df["PREDICTION_PROBABILITY"] = all_preds
    # check if all values in the "label" column are the same as the "ATTACKED" column
    assert dataset_df["LABEL"].equals(dataset_df["ATTACKED"])

    UTIL.save_dataframe(dataset_df, output_path)


def find_optimal_threshold(pred_prob_df_path, optimal_threshold_save_path):
    pred_prob_df = pd.read_parquet(pred_prob_df_path)
    labels = pred_prob_df["LABEL"].values.astype(int)
    # Initialize variables for the optimal threshold and the highest F1 score
    optimal_threshold = 0
    highest_f1_score = 0

    # Iterate through a range of thresholds
    for threshold in np.arange(0.05, 1, 0.2):
        # Create binary predictions using the current threshold
        binary_predictions = (
            pred_prob_df["PREDICTION_PROBABILITY"] >= threshold
        ).astype(int)

        # Calculate the F1 score using the true labels and the binary predictions
        current_f1_score = f1_score(labels, binary_predictions)

        # Update the optimal threshold and the highest F1 score if the current F1 score is higher
        if current_f1_score > highest_f1_score:
            optimal_threshold = threshold
            highest_f1_score = current_f1_score

    # Return the optimal threshold
    UTIL.save_float_to_file(optimal_threshold, optimal_threshold_save_path)
    return optimal_threshold


def generate_confusion_matrix_data(
    prediction_probability_dataset_path, output_path, threshold=0.5
):
    predic_prob_df = pd.read_parquet(prediction_probability_dataset_path)
    predic_prob_df["PREDICTION"] = (
        predic_prob_df["PREDICTION_PROBABILITY"] > threshold
    ).astype(float)
    predic_prob_df = add_confusion_matrix_columns(predic_prob_df)
    # save the dataset with the confusion matrix columns
    UTIL.save_dataframe(predic_prob_df, output_path)


def generate_aggregated_metrics_using_confusion_matrix(
    aggregate_parameter, confusion_matrix_path, output_path
):
    confusion_matrix = pd.read_parquet(confusion_matrix_path)

    # Group by the specified column and calculate the sum of 'TP', 'FP', 'TN', 'FN'
    grouped_data = confusion_matrix.groupby(aggregate_parameter).agg(
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
        true_labels = confusion_matrix.loc[
            confusion_matrix[aggregate_parameter] == group, "LABEL"
        ]
        predicted_probabilities = confusion_matrix.loc[
            confusion_matrix[aggregate_parameter] == group, "PREDICTION_PROBABILITY"
        ]
        try:
            roc_auc = roc_auc_score(true_labels, predicted_probabilities)
        except ValueError:
            roc_auc = 1
        roc_auc_scores.append(roc_auc)
    grouped_data["auc"] = roc_auc_scores

    # Reset the index
    grouped_data.reset_index(inplace=True)
    UTIL.save_dataframe(grouped_data, output_path)
    print(grouped_data)
    return grouped_data


def generate_roc_data(pred_prob_df_path, output_path):
    pred_prob_df = pd.read_parquet(pred_prob_df_path)
    y_true = pred_prob_df["LABEL"]
    y_score = pred_prob_df["PREDICTION_PROBABILITY"]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
    UTIL.save_dataframe(roc_data, output_path)


def plot_metrics(
    train_dataset_df, test_dataset_df, output_path_directory, aggregate_parameter
):
    # Extract the evaluation_metrics by excluding the aggregate_metric column
    evaluation_metrics = [
        col for col in train_dataset_df.columns if col != aggregate_parameter
    ]

    # Loop over the evaluation_metrics and create plots
    for metric in evaluation_metrics:
        plt.figure()  # Create a new figure for each plot
        plt.plot(
            train_dataset_df[aggregate_parameter],
            train_dataset_df[metric],
            label="train",
        )
        plt.plot(
            test_dataset_df[aggregate_parameter], test_dataset_df[metric], label="test"
        )
        plt.xlabel(aggregate_parameter)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {aggregate_parameter}")
        plt.legend()
        plot_save_path = (
            output_path_directory + aggregate_parameter + "_" + metric + ".png"
        )
        plt.savefig(plot_save_path, bbox_inches="tight")
        plt.close()


def plot_roc(train_roc_data_path, test_roc_data_path, output_path):
    train_roc_data = pd.read_parquet(train_roc_data_path)
    test_roc_data = pd.read_parquet(test_roc_data_path)

    plt.figure()
    plt.plot(
        train_roc_data["FPR"],
        train_roc_data["TPR"],
        label="Train ROC curve (area = %0.2f)"
        % np.trapz(train_roc_data["TPR"], train_roc_data["FPR"]),
    )
    plt.plot(
        test_roc_data["FPR"],
        test_roc_data["TPR"],
        label="Test ROC curve (area = %0.2f)"
        % np.trapz(test_roc_data["TPR"], test_roc_data["FPR"]),
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_attack_prediction_vs_time(dataset_path, output_path):
    dataset_df = pd.read_parquet(dataset_path)

    # Step 1: Create the attack_properties_columns list
    attack_properties_columns = [col for col in dataset_df.columns if "ATTACK" in col]
    attack_properties_columns.remove("ATTACKED")

    # Step 2: Get unique combinations of values in the attack_properties_columns
    unique_combinations = dataset_df[attack_properties_columns].drop_duplicates()

    # Create a directory to save the plots
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # Step 3: Loop through the unique combinations
    for index, combination in unique_combinations.iterrows():
        # Filter the dataset based on the unique combination
        condition = True
        for col in attack_properties_columns:
            condition &= dataset_df[col] == combination[col]
        filtered_df = dataset_df[condition]

        # Step 4: Group the filtered dataset by 'TIME' and calculate the average of 'LABEL', 'TP', and 'FP'
        grouped_df = (
            filtered_df.groupby("TIME")
            .agg({"LABEL": "mean", "TP": "mean", "FP": "mean"})
            .reset_index()
        )

        # Step 5: Create plots for each combination
        plt.figure(figsize=(12, 8))
        plt.plot(grouped_df["TIME"], grouped_df["LABEL"], label="LABEL")
        plt.plot(grouped_df["TIME"], grouped_df["TP"], label="TP")
        plt.plot(grouped_df["TIME"], grouped_df["FP"], label="FP")

        # Configure x-axis
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y - %H"))
        plt.xticks(rotation=30)

        # Set labels and title
        plt.xlabel("Hour")
        plt.ylabel("Ratio of The Nodes Under Attack")
        title = "\n".join(
            [f"{key}: {value}" for key, value in combination.to_dict().items()]
        )
        plt.title(f"{title}")
        plt.legend()

        # Save the plot with an intuitive name
        filename = (
            "_".join([f"{col}_{combination[col]}" for col in attack_properties_columns])
            + ".png"
        )
        plot_path = output_path + filename
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
