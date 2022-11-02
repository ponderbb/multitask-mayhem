from email.mime import image
import itertools
from multiprocessing.context import assert_spawning
import yaml
import glob
import os
from pathlib import Path
import xmltodict


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

def list_image_paths(collections: list):
    all_image_paths = []
    for collection in collections:
        collection_path = os.path.join("data/interim", collection, "synchronized_l515_image/")
        assert Path(collection_path).exists(), "collection path -> {} does not exists".format(collection_path)
        collection_images = glob.glob(collection_path+"*.png")
        assert len(collection_images) != 0, "collection -> {} is empty".format(collection)
        all_image_paths.extend(collection_images)
    return all_image_paths

def list_labels(collections: list):
    label_list = []
    for collection in collections:
        collection_label_path = os.path.join(
            "data/interim",
            collection,
            "{}.xml".format(collection)
        )
        assert Path(collection_label_path).exists(), "label path -> {} does not exist".format(collection_label_path)
        with open(collection_label_path, 'r', encoding='utf-8') as file:
            label_xml = file.read()
        label_list.extend(cvat_to_dict(label_xml, collection))

    return label_list

def cvat_to_dict(xml_file, collection):
    '''
    extract cvat labels
    '''

    label_dicts_list = []
    

    # generated dictionary with original html file inside
    raw_label_dict = xmltodict.parse(xml_file)

    # streamline it into a simpler dictionary
    for image_label in raw_label_dict["annotations"]["image"]:

        label_dict= {}

        label_dict["name"] = image_label["@name"]
        image_path = os.path.join("data/interim", collection, "synchronized_l515_image", label_dict["name"])
        assert Path(image_path).exists(), "Cannot find corresponding image to label {}".format(label_dict["name"])
        label_dict["path"] = image_path
        label_dict["shape"] = [
            int(image_label["@width"]),
            int(image_label["@height"])
        ]

        empty_bbox_list = []
        if "box" in image_label:

            # catch if there is only one entry in the annotation type
            if type(image_label["box"]) == dict:
                image_label["box"] = [image_label["box"]]

            for bbox in image_label["box"]:
                empty_bbox_list.append(
                    {
                        "label": bbox["@label"],
                        "corners": [
                            float(bbox["@xtl"]),
                            float(bbox["@ytl"]),
                            float(bbox["@xbr"]),
                            float(bbox["@ybr"])
                        ]
                    }
                )
            label_dict["bbox"] = empty_bbox_list

        empty_polygon_list = []
        if "polygon" in image_label:

            # catch if there is only one entry in the annotation type
            if type(image_label["polygon"]) == dict:
                image_label["polygon"] = [image_label["polygon"]]

            for poly in image_label["polygon"]:
                
                # convert nodes from str to floats
                nodes = []
                for node in poly["@points"].split(";"):
                    nodes.append(list(map(float,node.split(","))))

                empty_polygon_list.append(
                    {
                        "label": poly["@label"],
                        "nodes": nodes
                    }
                )

            label_dict["polygon"] = empty_polygon_list

        # appending of label to image[i] to the list
        label_dicts_list.append(label_dict)

    # return list of labels for annotation file of collection[i]
    return label_dicts_list
