import math

OUTPUT_DIRECTORY = "path_to_output_directory_for_saving_results"
DATASET_DIRECTORY = "path_to_dataset_directory"
SOURCE_CODE_DIRECTORY = "path_to_source_code_directory"

NUM_NODES = 50
NUM_GROUPS = 10

SAVE_CSV = False

# %% Benign data properties
BEGIN_DATE = "2021-01-02 00:00:00"
END_DATE = "2021-02-01 23:59:58"
TIME_STEP = 60 * 10
NUM_EPOCHS = 50
NUM_BATCHES = 1024
SAVE_BEST_MODEL = {
    "enabled": True,
    "data": "val",
    "metric": "f1_score",
    "condition": "max",
}
SAVE_MODEL_EACH_EPOCH = False
DIRECTED_GRAPH = False
# EDGE_DIRECTION assign the direction of edges in the graph which could be "NODE_TO_NEIGHBORS" or "NEIGHBORS_TO_NODE"
EDGE_DIRECTION = "NODE_TO_NEIGHBORS"

USE_ROUTER = True
if USE_ROUTER:
    ROUTER_ID = -1
    P2P_TOPOLOGY_LIST = ["DISTANCE", "CORRELATION", "NO_P2P"]
    ROUTER_TOPOLOGY_LIST = ["ROUTER", "NO_ROUTER"]
else:
    P2P_TOPOLOGY_LIST = ["DISTANCE", "CORRELATION"]
    ROUTER_TOPOLOGY_LIST = ["NO_ROUTER"]
NUM_EDGES_PER_NODE = 4


# Main parameters
# %% Attack data properties
K_LIST = [0, 0.1, 0.3, 0.5, 0.7, 1]
ATTACKED_RATIO_NODES_LIST = [0.5, 1]  # percentage
ATTACK_DURATION_LIST = [4, 8, 16]  # hours
ATTACK_START_TIME_LIST = [2, 6, 12]  # 24-hour format


# %% Train/Test Parameters
NUM_TRAIN_DAYS = 4
NUM_VALIDATION_DAYS = 1
NUM_TEST_DAYS = 3
MEAN_WINDOWS = {
    "30min": "PACKET_30_MIN",
    "60min": "PACKET_1_HOUR",
    "120min": "PACKET_2_HOUR",
    "240min": "PACKET_4_HOUR",
}
EDGES_CONNECTION_RATIO_LIST = [0.5, 0.7, 1]
