import xmltodict
import os
from pathlib import Path
from tqdm import tqdm


def generate_manifest(collections: list, data_root: str):
    manifest_list = []
    for collection in tqdm(collections):
        collection_label_path = os.path.join(
            data_root, collection, "{}.xml".format(collection)
        )
        assert Path(
            collection_label_path
        ).exists(), "label path -> {} does not exist".format(collection_label_path)
        with open(collection_label_path, "r", encoding="utf-8") as file:
            label_xml = file.read()
        manifest_list.extend(cvat_to_dict(label_xml, collection, data_root))

    return manifest_list


def cvat_to_dict(xml_file, collection, data_root):
    """
    - extract cvat labels to readable format
    - check for corresponding image paths
    - return dataloader manifest
    """

    label_dicts_list = []

    # generated dictionary with original html file inside
    raw_label_dict = xmltodict.parse(xml_file)

    # streamline it into a simpler dictionary
    for image_label in raw_label_dict["annotations"]["image"]:

        label_dict = {}

        label_dict["name"] = image_label["@name"]

        # find corresponding image path
        image_path = os.path.join(
            data_root, collection, "synchronized_l515_image", label_dict["name"]
        )  # FIXME: image folder name is fixed
        assert Path(
            image_path
        ).exists(), "Cannot find corresponding image to label {}".format(
            label_dict["name"]
        )
        label_dict["path"] = image_path
        label_dict["shape"] = [int(image_label["@width"]), int(image_label["@height"])]

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
                            float(bbox["@ybr"]),
                        ],
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
                    nodes.append(list(map(float, node.split(","))))

                empty_polygon_list.append({"label": poly["@label"], "nodes": nodes})

            label_dict["polygon"] = empty_polygon_list

        # appending of label to image[i] to the list
        label_dicts_list.append(label_dict)

    # return list of labels for annotation file of collection[i]
    return label_dicts_list
