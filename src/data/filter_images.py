import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from skimage.metrics import normalized_root_mse as skiNRMSE
from skimage.metrics import structural_similarity as skiSSIM
from tqdm import tqdm

import src.utils as utils
from src.data.manifests import cvat_to_dict


class filterImages:
    def __init__(
        self,
        similarity_limit: float,
        input_root: Union[str, Path],
        output_root: Union[str, Path],
        force: bool,
        rgb_only: bool = False,
        debug: bool = False,
    ) -> None:
        self.input = input_root
        self.output = output_root
        self.sim_lim = similarity_limit
        self.force = force
        self.rgb_only = rgb_only
        self.debug = debug
        self.image_topic = "synchronized_l515_image"
        self.depth_topic = "synchronized_l515_depth_image"
        self.pcl_topic = "synchronized_velodyne"

    def filter_all(self):
        logging.info("Listing bags from {}".format(self.input))
        bags_list = self._listbags_fullpath(self.input)
        assert bags_list is not None, "Folder list is empty, did you want to filter a single bag?"

        if self.debug:
            logging.debug("### DEBUGGING: only using last bag ###")
            bags_list = bags_list[-1:]
            self.force = True  # assumes the incentive to rewrite

        for i, self.bag in enumerate(bags_list):
            if self.check_output():
                logging.info("Bag already exists, not overwritten: {}".format(Path(self.bag).stem))
            else:
                logging.info("{}/{} Pruning bag: {}".format(i + 1, len(bags_list), self.bag))
                rgb_images = self._list_images(self.bag)
                self.compare_images(rgb_images)

                self.construct_paths()
                self.copy_files(self.output_image_path, "rgb")
                self.save_image_list()
                if not self.rgb_only:
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
        if not self.rgb_only:
            self.copy_files(self.output_depth_path, "depth")
            self.copy_files(self.output_pcl_path, "pcl")

    def filter_on_annotation(self, annotation_path: Union[str, Path]):

        self.bag = Path(annotation_path).stem
        self.force = True
        self.check_output()

        with open(annotation_path, "r", encoding="utf-8") as file:
            label_xml = file.read()

            # translates the xml to a dictionary (plus creates the mask folder)
            label_dict = cvat_to_dict(
                xml_file=label_xml,
                collection=Path(annotation_path).stem,
                data_root=self.input,
                create_mask=False,
            )

        self.pruned_list = [label["path"] for label in label_dict]
        self.construct_paths()
        self.copy_files(self.output_image_path, "rgb")
        self.save_image_list()
        if not self.rgb_only:
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
                sim_res = self._similarity_check(im1, im2)
                if sim_res < self.sim_lim:
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
                    "wrong type at copying: {}, it should be [depth, rgb, pcl]".format(cp_type)
                )

            shutil.copyfile(out_file, os.path.join(out_path, out_name))

    def construct_paths(self):
        self.output_image_path = os.path.join(self.output, Path(self.bag).stem, self.image_topic)
        self.output_depth_path = os.path.join(self.output, Path(self.bag).stem, self.depth_topic)
        self.output_pcl_path = os.path.join(self.output, Path(self.bag).stem, self.pcl_topic)

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
    def _similarity_check(im1, im2):
        ssim_res = skiSSIM(im1, im2, win_size=111)  # higher the more similar
        # TODO: find correct window size
        nrmse_res = skiNRMSE(im1, im2)  # lower the more similar
        similarity_index = ssim_res * 0.4 + (1 - nrmse_res) * 0.6  # TODO: better define the ratio
        return similarity_index

    @staticmethod
    def _listbags_fullpath(dir_path: Union[str, Path]):
        folder_list = [
            os.path.join(dir_path, folder)
            for folder in os.listdir(dir_path)
            if not folder.startswith(".")
        ]

        assert len(folder_list) != 0, "Folder list is empty, check bag files and input folder!"

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
        rgb_only=args.rgb,
        debug=args.debug,
    )

    # NOTE: remove after debugging
    # args.label = None
    args.label = "data/external/2022-09-23-11-03-28.xml"

    if args.bag:
        filt.filter_specific()
    elif args.label:
        filt.filter_on_annotation(args.label)
    else:
        filt.filter_all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--simlim",
        default=0.7,
        type=float,
        help="Similarity limit of images, float between [0,1]",
    )
    parser.add_argument("-i", "--input", default="data/raw/")
    parser.add_argument("-o", "--output", default="data/interim/")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Include debug level information in the logging.",
    )
    parser.add_argument(
        "-r",
        "--rgb",
        action="store_true",
        help="Only copy filtered rgb images.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite existing folders."
    )
    parser.add_argument(
        "-b", "--bag", action="store_true", help="Filter and overwrite specific bag."
    )
    parser.add_argument("-l", "--label", type=str, help="Path to annotation, filter images.")

    args = parser.parse_args()

    utils.logging_setup(debug=args.debug, outfile=".logging/filter_images.log")

    main(args)
