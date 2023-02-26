import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rosbag
from numpy_pc2 import pointcloud2_to_xyz_array
from tqdm import tqdm

import src.utils as utils


class unPackROSBag:
    def __init__(self, force: bool) -> None:
        self.root_dir = "data/rosbags/"
        self.export_dir = "data/raw"
        self.extension = ".bag"
        self.force = force

        self.image_topics = [
            "/synchronized_l515_image",
            "/synchronized_l515_depth_image",
        ]
        self.imu_topics = {
            "gyro": "/synchronized_gyro_sample",
            "accel": "/synchronized_accel_sample",
        }

        self.pcl_topics = ["/synchronized_velodyne"]

        self.collect_bags()

    def collect_bags(self):
        """
        - runs when initializing the class
        - collects the bag files in the defined data directory
        """

        self.bags_list = [
            str(Path(dir, file_))
            for dir, __, files in os.walk(self.root_dir)
            for file_ in files
            if Path(file_).suffix == self.extension
        ]

        self.bags_list.sort()

        if self.bags_list:
            logging.info("{} bag(s) found".format(len(self.bags_list)))
        else:
            raise FileNotFoundError("No files with .bag extension on the following path: {}".format(self.root_dir))

    def extract_bags(self):
        """
        - extracts the topics of interest within three categories [image, signal and pointcloud]
        - output folders are named <bag_name>/<topic_name>/...
        """

        for i, bag_path in enumerate(self.bags_list):

            bag_name = Path(bag_path).stem

            # create a folder with bag name in data/raw
            if self.check_output(os.path.join(self.export_dir, bag_name)):
                logging.info(
                    "{}/{}: bag extraction already exists, not overwritten: {}".format(
                        i + 1, len(self.bags_list), bag_name
                    )
                )
            else:
                os.makedirs(os.path.join(self.export_dir, bag_name), exist_ok=True)
                logging.info("{}/{}: unpacking {} ".format(i + 1, len(self.bags_list), bag_name))

                bag = rosbag.Bag(bag_path)
                self._write_images(bag, bag_name)
                self._write_imu(bag, bag_name)
                self._write_pcl(bag, bag_name)

                bag.close()

    def _write_images(self, bag, bag_name):
        """
        - converting bagmessages to opencv images
        - opencv issues: boost_cvbridge error if using cvbridge for decoding message
        - specific to encoding types
        """

        for topic in self.image_topics:

            # os.join throws error if topic starts with '/'
            topic_dir = os.path.join(self.export_dir, bag_name, topic[1:])
            os.makedirs(topic_dir, exist_ok=True)

            topic_read = bag.read_messages(topic)
            check_topic(topic, topic_read)
            logging.debug(topic)

            for i, msg in enumerate(tqdm(topic_read)):

                msg = msg.message

                if msg.encoding == "rgb8":
                    channels = 3
                    dtype = np.dtype("uint8")  # hardcode 8bits
                elif msg.encoding == "16UC1":
                    channels = 1
                    dtype = np.dtype("uint16")
                else:
                    raise TypeError("image encoding problem, found {}".format(msg.encoding))

                dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")

                image = np.ndarray(
                    shape=(msg.height, msg.width, channels),
                    dtype=dtype,
                    buffer=msg.data,
                )

                # convert to bgr
                if msg.encoding == "rgb8":
                    image = cv2.cvtColor(
                        image,
                        cv2.COLOR_RGB2BGR,
                    )

                # if the byte order is different between the message and the system.
                if msg.is_bigendian == (sys.byteorder == "little"):
                    image = image.byteswap().newbyteorder()

                cv2.imwrite(
                    os.path.join(topic_dir, "{}-{}.png".format(bag_name, str(i).zfill(6))),
                    image,
                )

    def check_output(self, bag_out_path):
        if os.path.exists(bag_out_path):
            if self.force:
                shutil.rmtree(bag_out_path)
                return False
            else:
                return True
        else:
            return False

    def _write_imu(self, bag, bag_name):

        gyro_topic = bag.read_messages(self.imu_topics["gyro"])
        accel_topic = bag.read_messages(self.imu_topics["accel"])
        check_topic(self.imu_topics["gyro"], gyro_topic)
        check_topic(self.imu_topics["accel"], accel_topic)
        logging.debug("{}, {}".format(self.imu_topics["gyro"], self.imu_topics["accel"]))

        out_csv = open(os.path.join(self.export_dir, bag_name, "imu_{}.csv".format(bag_name)), "w")

        out_csv.write("# gyro_timestamp accel_timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z\n")

        for i, (gyro_msg, accel_msg) in enumerate(tqdm(zip(gyro_topic, accel_topic))):

            g_msg = gyro_msg.message
            a_msg = accel_msg.message

            out_csv.write(
                "%s %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                % (
                    str(i).zfill(6),
                    g_msg.header.stamp.to_sec(),
                    a_msg.header.stamp.to_sec(),
                    g_msg.angular_velocity.x,
                    g_msg.angular_velocity.y,
                    g_msg.angular_velocity.z,
                    a_msg.linear_acceleration.x,
                    a_msg.linear_acceleration.y,
                    a_msg.linear_acceleration.z,
                )
            )

    def _write_pcl(self, bag, bag_name):

        for topic in self.pcl_topics:
            """
            [x, y, z, intensity, ring, time]
            """

            # os.join throws error if topic starts with '/'
            topic_dir = os.path.join(self.export_dir, bag_name, topic[1:])
            os.makedirs(topic_dir, exist_ok=True)

            topic_read = bag.read_messages(topic)
            logging.debug(topic)
            check_topic(topic, topic_read)

            for i, (topic, msg, t) in enumerate(tqdm(topic_read)):
                pc_array = pointcloud2_to_xyz_array(msg)  # NOTE: missing intensity values, could be extracted

                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = o3d.core.Tensor(pc_array)

                o3d.t.io.write_point_cloud(
                    os.path.join(topic_dir, "{}-{}.pcd".format(bag_name, str(i).zfill(6))),
                    pcd,
                )


def check_topic(topic, generator):
    if utils.peek(generator) is None:
        logging.warning("Topic generator is empty: {}".format(topic))


def main(args):

    logging.info("Unpacking ROS bags")
    unpack = unPackROSBag(force=args.force)
    unpack.extract_bags()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing folders.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Include debug level information in the logging.",
    )
    args = parser.parse_args()

    utils.logging_setup()

    main(args)
