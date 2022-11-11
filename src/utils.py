import itertools
import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def load_yaml(file: str) -> dict:
    with open(file, "r") as stream:
        dict = yaml.safe_load(stream)
    return dict


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def logging_setup(debug: bool, outfile: str) -> None:
    if debug:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging_level,
        format=log_fmt,
        force=True,
        handlers=[
            logging.FileHandler(outfile, "w"),
            logging.StreamHandler(),
        ],
    )


def grab_time() -> str:
    dt = datetime.now()
    str_date_time = dt.strftime("%y-%m-%dT%H:%M:%S")
    return str_date_time


def model_timestamp(
    model_name: str,
    attribute: str = None,
) -> str:
    """grab model name and combine it with timestamp
    (combined_name): fasterrcnn_test_2022-11-11-11-30-01
    """
    time_now = grab_time()

    # you can add attribute for easier finding special tests
    if attribute:
        combined_name = model_name + "_" + attribute
    else:
        combined_name = model_name

    return combined_name + "_" + time_now


def create_model_folders(config_path: str, model_folder: str, model_name: str) -> str:
    """Initialize folder structure for model"""
    folder_path = os.path.join(model_folder, model_name)
    weights_path = os.path.join(folder_path, "weights")
    checkpoints_path = os.path.join(folder_path, "checkpoints")

    # establishing model directory
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # copy config file over
    shutil.copy(config_path, folder_path)

    return weights_path, checkpoints_path
