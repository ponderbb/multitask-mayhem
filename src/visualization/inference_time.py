import logging

import numpy as np
import torch
from tqdm import tqdm

import src.utils as utils
from src.models.model_loader import ModelLoader

import csv

configs_list = [
    "configs/model-zoo/deeplabv3.yaml",
    "configs/model-zoo/lraspp.yaml",
    "configs/model-zoo/ssdlite.yaml",
    "configs/model-zoo/frcnn.yaml",
    # "configs/model-zoo/frcnn-resnet.yaml",
    "configs/model-zoo/lraspp-hybrid.yaml",
    "configs/model-zoo/ssdlite-hybrid.yaml",
    "configs/model-zoo/frcnn-hybrid.yaml",
]

utils.logging_setup()
utils.set_seeds()

with open("reports/figures/inference_time.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "params", "mean", "std"])


    for path in configs_list:

        config = utils.load_yaml(path)
        model = ModelLoader.grab_model(config=config)
        device = torch.device("cuda")
        model.to(device)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters]) / 1e6

        dummy_input = torch.randn(1, 3, 480, 640, dtype=torch.float).to(device)
        dummy_target = torch.randn(1, 1, 480, 640, dtype=torch.float).to(device)
        # logging.info(sum(p.numel() for p in model.parameters()) / 1e6)
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        model.eval()
        for _ in range(10):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)

        logging.info("Model: {} mean: {} std: {}".format(config["model"], mean_syn, std_syn))

        writer.writerow([config["model"], params, mean_syn, std_syn])
