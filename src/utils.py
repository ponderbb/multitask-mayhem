import itertools
import random

import numpy as np
import torch
import yaml
import logging


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def load_yaml(file):
    with open(file, "r") as stream:
        dict = yaml.safe_load(stream)
    return dict


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def logging_setup(debug, outfile)-> None:
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