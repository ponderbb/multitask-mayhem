import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
import wandb

import src.utils as utils
from src.visualization.draw_things import draw_bounding_boxes


class plUtils:
    @classmethod
    def resolve_paths(cls, config_path: str) -> Tuple[str, dict]:
        """Resolving paths and folder generation
        Args: master config file
        Returns: model name and dictionary of paths
        """
        config = utils.load_yaml(config_path)

        # check if config file is for trained model or not
        if cls._check_if_model_timestamped(config_path):
            logging.info("Config file is timestamped, creating paths for trained model")

            # model name is loaded from paths of config file
            model_name = str(Path(config_path).stem)

            path_dict = cls._create_paths(
                model_name=model_name,
                model_folder=config["model_out_path"],
            )

        else:
            logging.info("Config file is not timestamped, creating paths for new model")

            # create timestamped name for model
            model_name = cls._model_timestamp(model_name=config["model"], attribute=config["attribute"])

            # create paths for training
            path_dict = cls._create_paths(
                model_name=model_name, model_folder=config["model_out_path"], assert_paths=False
            )

            if not config["debug"]:
                logging.info("Generating folders for new model")
                # create folders for weights and checkpoints
                cls._create_model_folders(
                    config_old_path=config_path,
                    manifest_old_path=config["data_root"] + "/manifest.json",
                    path_dict=path_dict,
                )

        return model_name, path_dict

    @classmethod
    def _log_validation_images(
        cls, epoch, class_lookup, model_type, sanity_epoch, sanity_num, image_batch, prediction_batch, target_batch
    ):

        if epoch == 1 or epoch % sanity_epoch == 0:

            zipped_batch = list(zip(image_batch, prediction_batch, target_batch))
            sampled_batch = random.sample(zipped_batch, sanity_num)
            img_list = []

            for (image, prediction, target) in sampled_batch:

                image_tensor = image.mul(255).type(torch.uint8).squeeze(0)
                image = image.mul(255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

                if model_type == "segmentation":

                    prediction, target = cls._wandb_segmentation_formatting(prediction, target)

                    img = wandb.Image(
                        image,
                        masks={
                            "predictions": {
                                "mask_data": prediction,
                                "class_labels": class_lookup["sseg_rev"],
                            },
                            "ground_truth": {
                                "mask_data": target,
                                "class_labels": class_lookup["sseg_rev"],
                            },
                        },
                    )

                elif model_type == "detection":

                    # box_data_list = cls._wandb_bbox_formatting(
                    #     prediction["boxes"], prediction["labels"], prediction["scores"]
                    # )

                    # img = wandb.Image(
                    #     image,
                    #     boxes={
                    #         "predictions": {
                    #             "box_data": box_data_list,
                    #             "class_labels": class_lookup["bbox_rev"],
                    #         },
                    #     },
                    # )

                    # FIXME: solve wandb logging
                    prediction = cls._filter_predicitions(prediction, score_threshold=0.3)

                    drawn_image = draw_bounding_boxes(
                        image=image_tensor,
                        boxes=prediction["boxes"],
                        labels=[class_lookup["bbox_rev"][label.item()] for label in prediction["labels"]],
                        scores=prediction["scores"],
                    )
                    img = wandb.Image(T.ToPILImage()(drawn_image))

                img_list.append(img)

            wandb.log({"val_images": img_list})

    @staticmethod
    def tuple_of_tensors_to_tensor(tuple_of_tensors):
        return torch.stack(list(tuple_of_tensors), dim=0)

    @staticmethod
    def _wandb_segmentation_formatting(prediction, target):
        """formatting segmentation results for wandb"""
        prediction = torch.sigmoid(prediction) > 0.5
        prediction = prediction.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        target = target.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return prediction, target

    @staticmethod
    def _wandb_bbox_formatting(bboxes, labels, scores):
        """formatting bboxes for wandb"""
        box_data_list = []

        for box, label, score in zip(bboxes, labels, scores):
            box_data_list.append(
                {
                    "position": {
                        "minX": box[0].item(),
                        "maxX": box[2].item(),
                        "minY": box[1].item(),
                        "maxY": box[3].item(),
                    },
                    "class_id": label.item(),
                    "scores": {"acc": score.item()},
                }
            )

        return box_data_list

    @staticmethod
    def _model_timestamp(
        model_name: str,
        attribute: str = None,
    ) -> str:
        """grab model name and combine it with timestamp
        (combined_name): fasterrcnn_test_2022-11-11-11-30-01
        """
        time_now = utils.grab_time()

        # you can add attribute for easier finding special tests
        if attribute:
            combined_name = model_name + "_" + attribute
        else:
            combined_name = model_name

        return combined_name + "_" + time_now

    @staticmethod
    def _check_if_model_timestamped(config: str) -> bool:
        """check if model name is already timestamped"""
        regex = "_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]$"
        config_name = str(Path(config).stem)
        if re.findall(regex, config_name):
            return True
        else:
            return False

    @staticmethod
    def _create_paths(
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
            assert Path(folder_path).exists(), "model folder {} does not exist".format(folder_path)
            assert Path(weights_path).exists(), "weights folder {} does not exist".format(weights_path)
            assert Path(checkpoints_path).exists(), "checkpoints folder {} does not exist".format(checkpoints_path)
            assert Path(config_path).exists(), "config file {} does not exist".format(config_path)
            assert Path(manifest_path).exists(), "manifest file {} does not exist".format(manifest_path)

        return {
            "folder_path": folder_path,
            "weights_path": weights_path,
            "checkpoints_path": checkpoints_path,
            "config_path": config_path,
            "manifest_path": manifest_path,
        }

    @staticmethod
    def _create_model_folders(
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

    @staticmethod
    def _filter_predicitions(predictions, score_threshold):
        """filter predictions with score below threshold"""
        score_mask = predictions["scores"] > score_threshold
        predictions["boxes"] = predictions["boxes"][score_mask]
        predictions["labels"] = predictions["labels"][score_mask]
        predictions["scores"] = predictions["scores"][score_mask]
        return predictions
