import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class filterImages:
    def __init__(
        self,
        similarity_limit: float,
        input_root: Union[str, Path],
        output_root: Union[str, Path],
        force: bool,
        debug: bool = False,
    ) -> None:
        self.input = input_root
        self.output = output_root
        self.sim_lim = similarity_limit
        self.force = force
        self.debug = debug
        self.image_topic = "synchronized_l515_image"
        self.depth_topic = "synchronized_l515_depth_image"
        self.pcl_topic = "synchronized_velodyne"

    def filter_all(self):
        logging.info("Listing bags from {}".format(self.input))
        bags_list = self._listbags_fullpath(self.input)
        assert (
            bags_list != None
        ), "Folder list is empty, did you want to filter a single bag?"

        if self.debug:
            logging.debug("### DEBUGGING: only using last bag ###")
            bags_list = bags_list[-1:]
            self.force = True  # assumes the incentive to rewrite

        for i, self.bag in enumerate(bags_list):
            if self.check_output():
                logging.info(
                    "Bag already exists, not overwritten: {}".format(
                        Path(self.bag).stem
                    )
                )
            else:
                logging.info(
                    "{}/{} Pruning bag: {}".format(len(bags_list), i + 1, self.bag)
                )
                rgb_images = self._list_images(self.bag)
                self.compare_images(rgb_images)

                self.construct_paths()
                self.copy_files(self.output_image_path, "rgb")
                self.save_image_list()
                self.copy_files(self.output_depth_path, "depth")
                self.copy_files(self.output_pcl_path, "pcl")

    def filter_specific(self):
        self.bag = self.input
        self.force = True  # assumes the incentive to rewrite
        self.check_output()

        logging.info("Pruning specific bag: {}".format(self.bag))
        rgb_images = self._list_images(self.bag)
        self.compare_images(rgb_images)

        self.input = Path(self.input).parents[0]
        self.construct_paths()
        self.copy_files(self.output_image_path, "rgb")
        self.save_image_list()
        self.copy_files(self.output_depth_path, "depth")
        self.copy_files(self.output_pcl_path, "pcl")

    def compare_images(self, images_list: list):
        self.pruned_list = []
        count, count_lim = 0, 0

        first_image = images_list[0]
        for current_image in tqdm(images_list[1:]):
            im1 = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
            if not np.array_equal(im1, im2):
                count = count + 1
                sim_res = ssim(im1, im2)  # TODO change this
                if sim_res < float(self.sim_lim):
                    count_lim = count_lim + 1
                    self.pruned_list.append(first_image)
                    first_image = current_image

        percentage_pruned = (1 - (len(self.pruned_list) / len(images_list))) * 100

        logging.info(
            "All: {} | Non-dupicate: {} | Under-limit: {} | Pruned: {}% | SSIM: {}".format(
                len(images_list), count, count_lim, percentage_pruned, self.sim_lim
            )
        )

        return self.pruned_list

    def copy_files(self, out_path, cp_type: str):
        # try:
        logging.info("Creating folder {}".format(out_path))
        os.makedirs(out_path)

        logging.info("Copying {}:".format(cp_type))
        for image in tqdm(self.pruned_list):
            if cp_type == "depth":
                out_name = Path(image).name
                out_file = os.path.join(
                    self.input,
                    Path(self.bag).stem,
                    self.depth_topic,
                    out_name,
                )
            elif cp_type == "pcl":
                out_name = Path(image).stem + ".pcd"
                out_file = os.path.join(
                    self.input,
                    Path(self.bag).stem,
                    self.pcl_topic,
                    out_name,
                )
            elif cp_type == "rgb":
                out_name = Path(image).name
                out_file = image
            else:
                raise TypeError(
                    "wrong type at copying: {}, it should be [depth, rgb, pcl]".format(
                        cp_type
                    )
                )

            shutil.copyfile(out_file, os.path.join(out_path, out_name))

    # except:
    # logging.warning("Copying {} for bag {} failed".format(cp_type, self.bag))

    def construct_paths(self):
        self.output_image_path = os.path.join(
            self.output, Path(self.bag).stem, self.image_topic
        )
        self.output_depth_path = os.path.join(
            self.output, Path(self.bag).stem, self.depth_topic
        )
        self.output_pcl_path = os.path.join(
            self.output, Path(self.bag).stem, self.pcl_topic
        )

    def save_image_list(self):
        with open(
            f"{self.output_bag_path}/log_lim{self.sim_lim}_{Path(self.bag).stem}.json",
            "w",
        ) as f:
            json.dump(self.pruned_list, f)

    def check_output(self):
        self.output_bag_path = os.path.join(self.output, Path(self.bag).stem)
        if os.path.exists(self.output_bag_path):
            if self.force:
                shutil.rmtree(self.output_bag_path)
                return False
            else:
                return True
        else:
            return False

    @staticmethod
    def _listbags_fullpath(dir_path: Union[str, Path]):
        folder_list = [
            os.path.join(dir_path, folder)
            for folder in os.listdir(dir_path)
            if not folder.startswith(".")
        ]

        assert (
            len(folder_list) != 0
        ), "Folder list is empty, check bag files and input folder!"

        return sorted(folder_list)

    @staticmethod
    def _list_images(bag_path: Union[str, Path]):
        rgb_path_list = []
        for root, __, files in os.walk(bag_path):
            for name in files:
                if name.endswith((".png")) & ("depth" not in str(root)):
                    rgb_path_list.append(os.path.join(root, name))

        assert len(rgb_path_list) != 0, "There are no images in {}".format(bag_path)

        return sorted(rgb_path_list)


def main(args):

    filt = filterImages(
        similarity_limit=args.simlim,
        input_root=args.input,
        output_root=args.output,
        force=args.force,
        debug=args.debug,
    )
    if args.bag:
        logging
        filt.filter_specific()
    else:
        filt.filter_all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--simlim",
        default="0.9",
        help="Similarity limit of images, float between [0,1]",
    )
    parser.add_argument("-i", "--input", default="data/raw/")
    parser.add_argument("-o", "--output", default="data/interim/")
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        help="Include debug level information in the logging.",
    )
    parser.add_argument(
        "-f", "--force", default=False, help="Force overwrite existing folders."
    )
    parser.add_argument(
        "-b", "--bag", default=False, help="Unpack and overwrite specific bag."
    )

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=log_fmt,
        force=True,
        handlers=[
            logging.FileHandler(".logging/filter_images.log", "w"),
            logging.StreamHandler(),
        ],
    )

    main(args)
