import itertools
import yaml


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
