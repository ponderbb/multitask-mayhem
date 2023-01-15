from src.models.model_loader import ModelLoader
import src.utils as utils
import torch
import numpy as np
from tqdm import tqdm
import logging

configs_list = [
    # "configs/model-zoo/deeplabv3.yaml",
    # "configs/model-zoo/fasterrcnn_mob.yaml",
    # "configs/model-zoo/fasterrcnn_res.yaml",
    # "configs/model-zoo/ssdlite.yaml",
    "configs/hybrid_dryrun.yaml"
]

utils.logging_setup()

for path in configs_list:
    config = utils.load_yaml(path)
    model = ModelLoader.grab_model(config=config)
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(1, 3,480,640, dtype=torch.float).to(device)
    dummy_target = torch.randn(1, 1,480,640, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
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