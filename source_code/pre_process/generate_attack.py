import math
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from multiprocessing import Pool
from itertools import product
from scipy.stats import norm, cauchy
import tensorflow_probability as tfp

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


def generate_attack(
    benign_data,
    attack_begin_date,
    attack_end_date,
    attacked_ratio_nodes,
    attack_duration,
    attack_start_time,
    attack_packet_cauchy,
    attack_parameter,
    num_days,
    output_path,
    data_type,
):
    """Create attack in the benign dataset for the given features based on the data type.

    Keyword arguments:
    benign_data -- benign dataset to be used for attacking
    attack_begin_date -- the begin date of the attack
    attack_end_date -- the end date of the attack
    attacked_ratio_nodes -- the ratio of the nodes in the benign dataset to be attacked.
    attack_duration -- the duration of the attack
    attack_start_times -- the start times of the attacks withing the attack_begin_date and attack_end_date
    output_path -- the output path for storing the attacked dataset
    attack_packet_cauchy -- truncated cauchy distribution parameters for generating attack volume
    attack_parameter -- the k value for generating attack packet volume
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    """

    data_selected_nodes = benign_data.loc[
        (benign_data["TIME"] >= attack_begin_date)
        & (benign_data["TIME"] < attack_end_date)
    ]
    nodes = list(data_selected_nodes["NODE"].unique())
    num_attacked_nodes = math.ceil(len(nodes) * attacked_ratio_nodes)

    attacked_nodes = list(random.sample(nodes, k=num_attacked_nodes))
    for attack_day in range(num_days):
        attack_start_time = attack_start_time + timedelta(days=attack_day)
        attack_finish_time = attack_start_time + attack_duration

        select_rows = (
            (benign_data["NODE"].isin(attacked_nodes))
            & (benign_data["TIME"] >= attack_start_time)
            & (benign_data["TIME"] < attack_finish_time)
        )

        benign_data.loc[select_rows, "ACTIVE"] = 1
        benign_data.loc[select_rows, "ATTACKED"] = 1
        packet_dist = tfp.distributions.TruncatedCauchy(
            loc=attack_packet_cauchy[0],
            scale=attack_packet_cauchy[1],
            low=0,
            high=attack_packet_cauchy[2],
        )
        packet = packet_dist.sample([benign_data.loc[select_rows].shape[0]])
        packet = np.ceil(packet)
        benign_data.loc[select_rows, "PACKET"] = packet

    benign_data["BEGIN_DATE"] = attack_begin_date
    benign_data["END_DATE"] = attack_end_date
    benign_data["NUM_NODES"] = len(nodes)
    benign_data["ATTACK_RATIO"] = attacked_ratio_nodes
    benign_data["ATTACK_START_TIME"] = attack_start_time
    benign_data["ATTACK_DURATION"] = attack_duration
    benign_data["ATTACK_PARAMETER"] = attack_parameter

    benign_data = benign_data[
        [
            "BEGIN_DATE",
            "END_DATE",
            "NUM_NODES",
            "ATTACK_RATIO",
            "ATTACK_START_TIME",
            "ATTACK_DURATION",
            "ATTACK_PARAMETER",
            "NODE",
            "LAT",
            "LNG",
            "TIME",
            "TIME_HOUR",
            "ACTIVE",
            "PACKET",
            "ATTACKED",
        ]
    ]

    output_path += (
        "attacked_data_"
        + data_type
        + "_"
        + str(attack_begin_date)
        + "_"
        + str(attack_end_date)
        + "_ratio_"
        + str(attacked_ratio_nodes)
        + "_start_time_"
        + str(attack_start_time)
        + "_duration_"
        + str(attack_duration)
        + "_k_"
        + str(attack_parameter)
        + ".parquet"
    )
    column_types = {
        "NODE": "int32",
        "LAT": "float32",
        "LNG": "float32",
        "TIME": "datetime64[s]",
        "ACTIVE": "float32",
        "PACKET": "float32",
        "ATTACKED": "float32",
        "BEGIN_DATE": "datetime64[s]",
        "END_DATE": "datetime64[s]",
        "NUM_NODES": "int32",
        "ATTACK_RATIO": "float32",
        "ATTACK_START_TIME": "datetime64[s]",
        "ATTACK_DURATION": "timedelta64[s]",
        "ATTACK_PARAMETER": "float32",
        "TIME_HOUR": "float32",
    }
    benign_data = benign_data.astype(column_types)
    UTIL.save_dataframe(benign_data, output_path)


def main_generate_attack(
    group_number,
    data_type,
):
    """The main function to be used for calling generate_attack function

    Keyword arguments:
    benign_dataset_path -- the path to the benign dataset to be used for attacking
    data_type -- could be 'train' or 'test'. For the test data_type, we select the attacked nodes in the way that
                higher ratio attacked nodes contain the lower ratio attacked nodes.
    num_train_days -- the number of days to be used for creating the attacked dataset for training purposes
    num_test_days -- the number of days to be used for creating the attacked dataset for testing purposes
    k_list -- different k values for generating training/testing dataset
    time_step -- the time step used for generating benign dataset
    """
    time_step = CONFIG.TIME_STEP
    num_train_days = CONFIG.NUM_TRAIN_DAYS
    num_validation_days = CONFIG.NUM_VALIDATION_DAYS
    num_test_days = CONFIG.NUM_TEST_DAYS
    k_list = CONFIG.K_LIST
    attacked_ratio_nodes_list = CONFIG.ATTACKED_RATIO_NODES_LIST
    attack_duration_list = [
        timedelta(hours=float(i)) for i in CONFIG.ATTACK_DURATION_LIST
    ]
    attack_start_times_list = [
        timedelta(hours=float(i)) for i in CONFIG.ATTACK_START_TIME_LIST
    ]

    benign_dataset_path = PATH_CONFIG.get_benign_dataset_path(group_number)
    benign_data = pd.read_parquet(benign_dataset_path)
    # set the begin and end date of the dataset to be attacked
    if data_type == "train":
        attack_begin_date = benign_data.loc[0, "TIME"] + timedelta(days=2)
        attack_end_date = benign_data.loc[0, "TIME"] + timedelta(
            days=2 + num_train_days
        )
        num_days = num_train_days
    elif data_type == "validation":
        attack_begin_date = benign_data.loc[0, "TIME"] + timedelta(
            days=2 + num_train_days + 2
        )
        attack_end_date = benign_data.loc[0, "TIME"] + timedelta(
            days=2 + num_train_days + 2 + num_validation_days
        )
        num_days = num_validation_days
    elif data_type == "test":
        attack_begin_date = benign_data.loc[0, "TIME"] + timedelta(
            days=2 + num_train_days + 2 + num_validation_days + 2
        )
        attack_end_date = benign_data.loc[0, "TIME"] + timedelta(
            days=2 + num_train_days + 2 + num_validation_days + 2 + num_test_days
        )
        num_days = num_test_days

    output_path = PATH_CONFIG.get_attack_dataset_directory_path(group_number, data_type)

    slice_benign_data_start = attack_begin_date - timedelta(days=2)
    slice_benign_data_end = attack_end_date

    benign_data = benign_data.loc[
        (benign_data["TIME"] >= slice_benign_data_start)
        & (benign_data["TIME"] < slice_benign_data_end)
    ]
    benign_data_save = benign_data.copy()
    nodes = list(benign_data_save["NODE"].unique())
    benign_data_save["BEGIN_DATE"] = attack_begin_date
    benign_data_save["END_DATE"] = attack_end_date
    benign_data_save["NUM_NODES"] = len(nodes)
    benign_data_save["ATTACK_RATIO"] = 0.0
    benign_data_save["ATTACK_START_TIME"] = attack_begin_date
    benign_data_save["ATTACK_DURATION"] = timedelta(hours=0)
    benign_data_save["ATTACK_PARAMETER"] = 0.0
    benign_data_save = benign_data_save[
        [
            "BEGIN_DATE",
            "END_DATE",
            "NUM_NODES",
            "ATTACK_RATIO",
            "ATTACK_START_TIME",
            "ATTACK_DURATION",
            "ATTACK_PARAMETER",
            "NODE",
            "LAT",
            "LNG",
            "TIME",
            "TIME_HOUR",
            "ACTIVE",
            "PACKET",
            "ATTACKED",
        ]
    ]
    output_path_benign = (
        output_path
        + "attacked_data_"
        + data_type
        + "_"
        + str(attack_begin_date)
        + "_"
        + str(attack_end_date)
        + "_ratio_0_start_time_"
        + str(attack_begin_date)
        + "duration_0_k_0"
        + ".parquet"
    )
    column_types = {
        "NODE": "int32",
        "LAT": "float32",
        "LNG": "float32",
        "TIME": "datetime64[s]",
        "ACTIVE": "float32",
        "PACKET": "float32",
        "ATTACKED": "float32",
        "BEGIN_DATE": "datetime64[s]",
        "END_DATE": "datetime64[s]",
        "NUM_NODES": "int32",
        "ATTACK_RATIO": "float",
        "ATTACK_START_TIME": "datetime64[ns]",
        "ATTACK_DURATION": "timedelta64[s]",
        "ATTACK_PARAMETER": "float32",
        "TIME_HOUR": "float32",
    }
    benign_data_save = benign_data_save.astype(column_types)
    UTIL.save_dataframe(benign_data_save, output_path_benign)

    attack_start_times_list = [attack_begin_date + i for i in attack_start_times_list]

    benign_packet_path = PATH_CONFIG.get_benign_packet_path()
    benign_packet = pd.read_csv(benign_packet_path)
    benign_packet_cauchy = list(cauchy.fit(benign_packet["PACKET"]))
    benign_packet_cauchy.append(math.ceil(max(benign_packet["PACKET"])))

    for k in k_list:
        k = round(k, 2)
        print("k : ", k)

        scale_factor = time_step / 10
        packet_loc = (1 + k) * benign_packet_cauchy[0] * scale_factor
        packet_scale = (1 + k) * benign_packet_cauchy[1] * scale_factor
        packet_max = (1 + k) * benign_packet_cauchy[2] * scale_factor
        packet_cauchy = [packet_loc, packet_scale, packet_max]

        p = Pool()
        p.starmap(
            generate_attack,
            product(
                [benign_data],
                [attack_begin_date],
                [attack_end_date],
                attacked_ratio_nodes_list,
                attack_duration_list,
                attack_start_times_list,
                [packet_cauchy],
                [k],
                [num_days],
                [output_path],
                [data_type],
            ),
        )
        p.close()
        p.join()


def main():
    for group_number in range(CONFIG.NUM_GROUPS):
        for data_type in ["train", "validation", "test"]:
            main_generate_attack(
                group_number,
                data_type,
            )
            print("generate data done: " + data_type)


if __name__ == "__main__":
    main()
