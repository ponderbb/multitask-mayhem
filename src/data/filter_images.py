import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def listbags_fullpath(dir_path: str):
    folder_list = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path) if not folder.startswith('.')]
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
    images_list = sorted(images_list)
    first_image = images_list[0]
    for current_image in tqdm(images_list[1:]):
        im1 = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
        if not np.array_equal(im1, im2):
            count = count + 1
            ssim_result = ssim(im1, im2)
            if ssim_result < ssim_limit:
                count_lim = count_lim + 1
                pruned_list.append(first_image)
                first_image = current_image
    logging.info(
        "Non-dupicate: {} | Under-limit: {} | SSIM: {}".format(
            count, count_lim, ssim_limit
        )
    )

    return pruned_list


def copy_pruned_images(
    pruned_list: list,
    bag_path: str,
    input_root: str,
    output_root: str,
    depth=False,
    pcl=False,
):

    # Creating path
    output_bag_path = os.path.join(output_root, Path(bag_path).stem)
    output_image_path = os.path.join(
        output_root, Path(bag_path).stem, "synchronized_l515_image"
    )
    output_depth_path = os.path.join(
        output_root, Path(bag_path).stem, "synchronized_l515_depth_image"
    )
    output_pcl_path = os.path.join(
        output_root, Path(bag_path).stem, "synchronized_velodyne"
    )

    if os.path.exists(output_bag_path):
        logging.debug("Remove previous bag directory")
        shutil.rmtree(output_bag_path)

    try:
        logging.info("Creating folder {}".format(output_image_path))
        os.makedirs(output_image_path)

        logging.info("Copying images.")
        for image in tqdm(pruned_list):
            shutil.copyfile(image, os.path.join(output_image_path, Path(image).name))
    except:
        logging.warning("Copying images for bag {} failed".format(bag_path))

    if depth:
        try:
            logging.info("Creating folder {}".format(output_depth_path))
            os.makedirs(output_depth_path)

            logging.info("Copying depth images.")
            for image in tqdm(pruned_list):
                image = os.path.join(
                    input_root,
                    Path(bag_path).stem,
                    "synchronized_l515_depth_image",
                    Path(image).name,
                )  # FIXME: not bulletproof declaration of paths, ask them as arguments
                shutil.copyfile(
                    image, os.path.join(output_depth_path, Path(image).name)
                )
        except:
            logging.warning("Copying depth images for bag {} failed".format(bag_path))

    if pcl:
        try:
            logging.info("Creating folder {}".format(output_pcl_path))
            os.makedirs(output_pcl_path)

            logging.info("Copying pointcloud scans.")
            for image in tqdm(pruned_list):
                image = os.path.join(
                    input_root,
                    Path(bag_path).stem,
                    "synchronized_velodyne",
                    (Path(image).stem + ".pcd"),
                )  # FIXME: not bulletproof declaration of paths, ask them as arguments
                shutil.copyfile(
                    image, os.path.join(output_pcl_path, (Path(image).stem + ".pcd"))
                )
        except:
            logging.warning(
                "Copying pointcloud scans for bag {} failed".format(bag_path)
            )


def main(args):
    logging.info("Listing bags")
    bags_list = listbags_fullpath(args.input)

    if args.debug:
        bags_list = bags_list[-1:]

    for i, bag in enumerate(bags_list):
        logging.info("{}/{} Pruning bag: {}".format(len(bags_list), i, bag))
        rgb_images = listimages(bag)
        pruned_images = compare_images(rgb_images, ssim_limit=float(args.ssim))
        logging.info(
            "Pruned {:.3f} percentage of images from {} -> {}.".format(
                (1 - (len(pruned_images) / len(rgb_images))) * 100,
                len(rgb_images),
                len(pruned_images),
            )
        )
        copy_pruned_images(
            pruned_list=pruned_images,
            bag_path=bag,
            input_root=args.input,
            output_root=args.output,
            depth=True,
            pcl=True,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--ssim",
        default="0.7",
        help="SSIM cutoff limit of images, float between [0,1]",
    )
    parser.add_argument("-i", "--input", default="data/raw/")
    parser.add_argument("-o", "--output", default="data/interim/")
    parser.add_argument("-d", "--debug", default=False)

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
