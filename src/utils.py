import glob
import itertools
import logging
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import __main__
import numpy as np
import torch
import yaml

""" General function utility files concerning:
    - path handling
    - os related listing and folder creation
    - seed setting
"""


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


def logging_setup(config_path: str = "") -> None:
    """Setup logging"""

    # create log folder
    os.makedirs(".logging", exist_ok=True)

    # load logging level
    if config_path != "":
        config = load_yaml(config_path)
    else:
        config = {"debug": False}

    if config["debug"]:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    # restrict logging for specific modules
    logging.getLogger("PIL").setLevel(logging.WARNING)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging_level,
        format=log_fmt,
        force=True,
        handlers=[
            logging.FileHandler(".logging/{}.log".format(Path(__main__.__file__).stem), "w"),
            logging.StreamHandler(),
        ],
    )


def grab_time() -> str:
    dt = datetime.now()
    str_date_time = dt.strftime("%y-%m-%dT%H%M%S")  # change to be windows compatible
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


def check_if_model_timestamped(config: str) -> bool:
    """check if model name is already timestamped"""
    regex = "_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]$"
    config_name = str(Path(config).stem)
    if re.findall(config_name, regex):
        return True
    else:
        return False


def create_paths(
    model_folder: str,
    model_name: str,
    assert_paths: bool = True,
):
    logging.info("creating paths for model: {}".format(model_name))
    # create paths to existing folders
    folder_path = os.path.join(model_folder, model_name)
    weights_path = os.path.join(folder_path, "weights")
    checkpoints_path = os.path.join(folder_path, "checkpoints")
    config_path = os.path.join(folder_path, "{}.yaml".format(model_name))
    manifest_path = os.path.join(folder_path, "manifest.json")

    if assert_paths:
        # assertions as sanity check
        assert not (folder_path.exists()), "model folder does not exist"
        assert not (weights_path.exists()), "weights folder does not exist"
        assert not (checkpoints_path.exists()), "checkpoints folder does not exist"
        assert not (config_path.exists()), "config file does not exist"
        assert not (manifest_path.exists()), "manifest file does not exist"

    return {
        "folder_path": folder_path,
        "weights_path": weights_path,
        "checkpoints_path": checkpoints_path,
        "config_path": config_path,
        "manifest_path": manifest_path,
    }


def create_model_folders(
    config_old_path: str,
    manifest_old_path: str,
    path_dict: dict,
):
    """Initialize folder structure for model
    debug: if True, pass paths as None to avoid creating folders
    """

    # establishing model directory
    os.makedirs(path_dict["weights_path"], exist_ok=True)
    os.makedirs(path_dict["checkpoints_path"], exist_ok=True)

    # copy config- and move manifest file over
    shutil.copy(config_old_path, path_dict["config_path"])
    shutil.move(manifest_old_path, path_dict["manifest_path"])

    # logs
    logging.info("model folder created: {}".format(path_dict["folder_path"]))
    logging.info("config file moved: {}".format(Path(config_old_path).name))


def list_files_with_extension(path: str, extension: str, format: str) -> list:
    """List all files with a given extension in a directory
    format defines the return structure
    stem: returns only the filename without extension
    name: returns the filename with extension
    path: returns the full path"""

    assert path[-1] == "/", "path should end with /"
    assert extension[0] == ".", "extension should start with ."
    assert format in ["stem", "name", "path"], "format should be stem, name or path"

    files_list = glob.glob(path + "*" + extension)
    if format == "stem":
        files_list = [Path(file).stem for file in files_list]
    elif format == "name":
        files_list = [Path(file).name for file in files_list]
    return sorted(files_list)
