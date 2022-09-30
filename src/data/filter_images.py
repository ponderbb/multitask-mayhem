import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

ROOT_DATA_DIR = "data/raw/"
OUTPUT_DATA_DIR = "data/interim/"


def listbags_fullpath(dir_path: str):
    folder_list = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)]
    folder_list.sort()

    return folder_list


def listimages(bag_path: str):
    rgb_path_list = []
    for root, dirs, files in os.walk(bag_path):
        for name in files:
            if name.endswith((".png")) & ("depth" not in str(root)):
                rgb_path_list.append(os.path.join(root, name))

    return rgb_path_list


def compare_images(images_list: list, ssim_limit: float):
    pruned_list = []
    count = 0
    count_lim = 0
    first_image = images_list[0]
    for current_image in tqdm(images_list):
        im1 = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
        if not np.array_equal(im1, im2):
            count = count + 1
            ssim_result = ssim(im1, im2)
            if ssim_result < ssim_limit:
                count_lim = count_lim + 1
                pruned_list.append(first_image)
                first_image = current_image
    logging.info("non_equal: {} different enough: {} with SSIM: {}".format(count, count_lim, ssim_limit))

    return pruned_list


def copy_pruned_images(pruned_list: list, bag_path: str, output_root: str):

    output_full_path = os.path.join(output_root, Path(bag_path).stem, "synchronized_l515_image")

    logging.info("Creating folder {} at {}".format(Path(bag_path).stem, output_root))
    if os.path.exists(output_full_path):
        logging.debug("Remove previous directory")
        shutil.rmtree(output_full_path)
    os.makedirs(output_full_path)

    logging.info("Copying images.")
    for image in tqdm(pruned_list):
        shutil.copyfile(image, os.path.join(output_full_path, Path(image).name))


def main():
    logging.info("Listing bags")
    bags_list = listbags_fullpath(ROOT_DATA_DIR)
    for i, bag in enumerate(bags_list[-3:-2]):
        logging.info("{}/{} Pruning bag: {}".format(len(bags_list), i, bag))
        rgb_images = listimages(bag)
        pruned_images = compare_images(rgb_images, ssim_limit=0.75)
        logging.info("Pruned images from {} -> {}.".format(len(rgb_images), len(pruned_images)))
        copy_pruned_images(pruned_images, bag, OUTPUT_DATA_DIR)


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_fmt,
        force=True,
        handlers=[
            logging.FileHandler(".logging/filter_images.log", "w"),
            logging.StreamHandler(),
        ],
    )
    main()
