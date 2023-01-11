import torch

import src.utils as utils
from src.models.model_loader import ModelLoader

CONFIG_PATH = "configs/debug_foo.yaml"

config = utils.load_yaml(CONFIG_PATH)
model = ModelLoader.grab_model(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(2, 3, 480, 640, dtype=torch.float).to(device)
dummy_target = {
    "boxes": torch.empty(size=[0, 4], dtype=torch.float32).to(device),  # N x 4
    "labels": torch.empty(size=[0], dtype=torch.int64).to(device),  # N
    "masks": torch.randn(1, 1, 480, 640, dtype=torch.float).to(device),  # C x H x W
}


model.to(device)
model.eval()

preds = model(dummy_input)
