import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, Manager
import sys
from pathlib import Path
import random
import os
from geopandas import GeoSeries
from shapely.geometry import Point
from haversine import haversine
from scipy.stats import norm, cauchy
import tensorflow_probability as tfp

source_code_dir = Path(__file__).resolve()
while source_code_dir.name != "source_code":
    source_code_dir = source_code_dir.parent
source_code_dir = source_code_dir.parent
sys.path.insert(0, str(source_code_dir))

import source_code.config.project_config as CONFIG
import source_code.config.path_config as PATH_CONFIG
import source_code.utilities.utilities as UTIL


def load_dataset(path):
    """Load the dataset and change the type of the 'TIME' column to datetime.

    Keyword arguments:
    path -- path to the dataset
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def create_benign_data_for_node(
    data, dates, begin_date, time_step, node, benign_data, benign_data_rows
):
    """Create benign dataset for a given dataset and node, and dates.

    Keyword arguments:
    data -- the dataset to be used for generating benign data
    dates -- a list of dates to be used for assigning the occupancy status
    begin_date -- the begin date of assignment
    time_step -- the time steps between dates
    node -- the node to be used for assigning the occupancy status
    output_path -- the path for storing the benign dataset
    """
    # benign_data = pd.DataFrame()
    benign_data["TIME"] = dates
    benign_data["NODE"] = node
    benign_data["LAT"] = list(data.loc[data["NODE"] == node, "LAT"])[0]
    benign_data["LNG"] = list(data.loc[data["NODE"] == node, "LNG"])[0]
    benign_data["ACTIVE"] = 0
    benign_data = benign_data[["NODE", "LAT", "LNG", "TIME", "ACTIVE"]]
    benign_data = benign_data.sort_values(by=["TIME"])

    data_sid = data.loc[data["NODE"] == node]
    data_sid = data_sid.sort_values(by=["TIME"])
    start_time = begin_date
    for index, row in data_sid.iterrows():
        finish_time = row["TIME"]
        benign_data.loc[
            (benign_data["TIME"] >= row["TIME"])
            & (benign_data["TIME"] < row["TIME"] + timedelta(seconds=time_step)),
            "ACTIVE",
        ] = row["ACTIVE"]

        benign_data.loc[
            (benign_data["TIME"] >= start_time) & (benign_data["TIME"] < finish_time),
            "ACTIVE",
        ] = int(not (row["ACTIVE"]))

        start_time = row["TIME"] + timedelta(seconds=time_step)

    # benign_data.to_csv(output_path, mode='a', header=False, index=False)
    benign_data_rows.append(benign_data)

    # return benign_data


def create_benign_dataset(
    data, begin_date, end_date, time_step, nodes, benign_packet_cauchy, output_path
):
    """Create benign dataset for a given dataset. Benign dataset contains the occupancy status of each node starting
    from the begin_date to end_date with the step of time_step. num_nodes will be used to randomly select num_nodes
    nodes from all of the nodes in the original dataset.

    Keyword arguments:
    data -- the dataset to be used for generating benign data
    begin_date -- the begin date of assignment
    emd_date -- the end date of assignment
    time_step -- the time steps between dates
    num_nodes -- number of nodes to be selected out the whole nodes in the dataset
    output_path -- the path for storing the benign dataset
    """
    dates = []
    date = begin_date
    while date < end_date:
        dates.append(date)
        date += timedelta(seconds=time_step)

    benign_data = pd.DataFrame(
        columns=["NODE", "LAT", "LNG", "TIME", "ACTIVE", "ATTACKED"]
    )
    # benign_data.to_csv(output_path, index=False)

    manager = Manager()
    benign_data_rows = manager.list([benign_data])

    p = Pool()
    p.starmap(
        create_benign_data_for_node,
        product(
            [data],
            [dates],
            [begin_date],
            [time_step],
            nodes,
            [benign_data],
            [benign_data_rows],
        ),
    )
    p.close()
    p.join()

    benign_data = pd.concat(benign_data_rows, ignore_index=True)
    benign_data = benign_data.sort_values(by=["NODE", "TIME"])

    packet_dist = tfp.distributions.TruncatedCauchy(
        loc=benign_packet_cauchy[0],
        scale=benign_packet_cauchy[1],
        low=0,
        high=benign_packet_cauchy[2],
    )
    packet = packet_dist.sample([benign_data.shape[0]])
    packet = np.ceil(packet)

    benign_data["PACKET"] = packet
    benign_data["ATTACKED"] = 0
    benign_data.loc[benign_data["ACTIVE"] == 0, "PACKET"] = 0
    benign_data["TIME_HOUR"] = benign_data["TIME"].dt.hour
    benign_data = benign_data[
        ["NODE", "LAT", "LNG", "TIME", "TIME_HOUR", "ACTIVE", "PACKET", "ATTACKED"]
    ]
    column_types = {
        "NODE": "int32",
        "LAT": "float32",
        "LNG": "float32",
        "TIME": "datetime64[s]",
        "TIME_HOUR": "float32",
        "ACTIVE": "float32",
        "PACKET": "float32",
        "ATTACKED": "float32",
    }

    benign_data = benign_data.astype(column_types)
    UTIL.save_dataframe(benign_data, output_path)


def main_generate_benign_data():
    original_data_path = PATH_CONFIG.get_original_data_path()
    original_data = load_dataset(original_data_path)

    benign_packet_path = PATH_CONFIG.get_benign_packet_path()
    benign_packet = pd.read_csv(benign_packet_path)
    benign_packet_cauchy = list(cauchy.fit(benign_packet["PACKET"]))
    benign_packet_cauchy.append(math.ceil(max(benign_packet["PACKET"])))

    begin_date = datetime.strptime(CONFIG.BEGIN_DATE, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(CONFIG.END_DATE, "%Y-%m-%d %H:%M:%S")
    time_step = CONFIG.TIME_STEP

    original_data = original_data.loc[
        (original_data["TIME"] >= begin_date) & (original_data["TIME"] <= end_date)
    ].reset_index(drop=True)

    nodes = list(original_data["NODE"].unique())
    print("len(nodes): ", len(nodes))

    num_nodes_per_group = CONFIG.NUM_NODES
    num_groups = CONFIG.NUM_GROUPS
    scale_factor = time_step / 10
    benign_packet_cauchy = [scale_factor * i for i in benign_packet_cauchy]

    for index in range(num_groups):
        selected_nodes = list(random.sample(nodes, k=num_nodes_per_group))
        benign_data_output_path = PATH_CONFIG.get_benign_dataset_path(index)
        UTIL.prepare_output_directory(benign_data_output_path)
        create_benign_dataset(
            original_data,
            begin_date,
            end_date,
            time_step,
            selected_nodes,
            benign_packet_cauchy,
            benign_data_output_path,
        )
        for node in selected_nodes:
            nodes.remove(node)


def main():
    main_generate_benign_data()


if __name__ == "__main__":
    main()
