import json
import os
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
import xmltodict
from tqdm import tqdm

from src.utils import load_yaml


def generate_manifest(collections: List[str], data_root: str):
    """
    - generate a dictionary for each labelled image
      with the following keys:
        [name, path, mask, shape, bbox, polygon]
    - save segmentation masks into the /data_root/collection folder
    - dump a manifest including all collections into /data_root
    """
    manifest_list = []
    for collection in tqdm(collections):

        # from the imagefolder, move to the dataroot "/data/interim" -> "/data"
        data_parent_folder = str(Path(data_root).parents[0])

        collection_label_path = os.path.join(
            data_parent_folder, "annotations/{}.xml".format(collection)
        )

        assert Path(collection_label_path).exists(), "label path -> {} does not exist".format(
            collection_label_path
        )

        with open(collection_label_path, "r", encoding="utf-8") as file:
            label_xml = file.read()
        manifest_list.extend(cvat_to_dict(label_xml, collection, data_root))
        with open(os.path.join(data_root, "manifest.json"), "w") as outfile:
            json.dump(manifest_list, outfile)

    return manifest_list


def mask_from_poly(shape, out_path, polygon=None):
    """
    grab polygon points from cvat annotation file
    and create a mask under /data_root/collection/name.png
    """
    mask = np.zeros(shape)
    if polygon is not None:
        for poly in polygon:
            points = np.array([(int(p[0]), int(p[1])) for p in poly["nodes"]])
            points = points.astype(int)
            mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    cv2.imwrite(out_path, mask)


def label_encoding(annotation, label):
    assert Path("configs/class_lookup.yaml").exists(), "did not find labels lookup table"
    lookup_dict = load_yaml("configs/class_lookup.yaml")
    return lookup_dict[annotation][label]


def cvat_to_dict(xml_file, collection, data_root, create_mask: bool = True):
    """
    - extract cvat labels to readable format
    - check for corresponding image paths
    - return dataloader manifest
    - #FIXME: too complex, break to pieces
    """

    label_dicts_list = []

    if create_mask:
        # clean directory for new masks
        mask_out_path = os.path.join(data_root, collection, "synchronized_l515_mask")
        if Path(mask_out_path).exists():
            shutil.rmtree(mask_out_path)
        os.makedirs(mask_out_path)

    # generated dictionary with original html file inside
    raw_label_dict = xmltodict.parse(xml_file)

    # streamline it into a simpler dictionary
    for image_label in raw_label_dict["annotations"]["image"]:

        label_dict = {}

        # fill the annotation details into the new dictionary
        label_dict["name"] = image_label["@name"]

        # find corresponding image path
        image_path = os.path.join(
            data_root, collection, "synchronized_l515_image", label_dict["name"]
        )

        # check if the image referenced by the label exists
        assert Path(image_path).exists(), "Cannot find corresponding image to label {}".format(
            label_dict["name"]
        )

        # fill the annotation details into the new dictionary
        label_dict["path"] = image_path
        if create_mask:
            label_dict["mask"] = os.path.join(mask_out_path, label_dict["name"])
        label_dict["shape"] = [int(image_label["@height"]), int(image_label["@width"])]

        # find all the bounding boxes and save them as a list
        empty_bbox_list = []
        if "box" in image_label:

            # catch if there is only one entry in the annotation type
            if type(image_label["box"]) == dict:
                image_label["box"] = [image_label["box"]]

            for bbox in image_label["box"]:
                empty_bbox_list.append(
                    {
                        "label": bbox["@label"],
                        "class": label_encoding("bbox", bbox["@label"]),
                        "corners": [
                            float(bbox["@xtl"]),
                            float(bbox["@ytl"]),
                            float(bbox["@xbr"]),
                            float(bbox["@ybr"]),
                        ],
                    }
                )
            label_dict["bbox"] = empty_bbox_list

        if create_mask:
            # find all the segmentation polygons and save them as a list
            empty_polygon_list = []
            if "polygon" in image_label:

                # catch if there is only one entry in the annotation type
                if type(image_label["polygon"]) == dict:
                    image_label["polygon"] = [image_label["polygon"]]

                for poly in image_label["polygon"]:

                    # convert nodes from str to floats
                    nodes = []
                    for node in poly["@points"].split(";"):
                        nodes.append(list(map(float, node.split(","))))

                    empty_polygon_list.append({"label": poly["@label"], "nodes": nodes})

                label_dict["polygon"] = empty_polygon_list

                # save the mask in a .png format under /data_root/collection/image_name.png
                mask_from_poly(
                    shape=label_dict["shape"],
                    out_path=label_dict["mask"],
                    polygon=label_dict["polygon"],
                )

            else:
                # if no segmentation in the image, save an empty mask
                mask_from_poly(shape=label_dict["shape"], out_path=label_dict["mask"])

        # appending of label to image[i] to the list
        label_dicts_list.append(label_dict)

    # return list of labels for annotation file of collection[i]
    return label_dicts_list
