import rosbag
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
from numpy_pc2 import pointcloud2_to_xyz_array
import open3d as o3d


class unPackROSBag:
    def __init__(self) -> None:
        self.root_dir = "data/external/"
        self.export_dir = "data/raw"
        self.extension = ".bag"

        self.image_topics = [
            "/synchronized_l515_image",
            "/synchronized_l515_depth_image",
        ]
        self.signal_topics = [
            "/synchronized_gyro_sample",
            "/synchronized_accel_sample",
        ]
        self.pcl_topics = ["/synchronized_velodyne"]

        self.collect_bags()

    def collect_bags(self):
        '''
        - runs when initializing the class
        - collects the bag files in the defined data directory
        '''

        self.bags_list = [
            str(Path(dir, file_))
            for dir, __, files in os.walk(self.root_dir)
            for file_ in files
            if Path(file_).suffix == self.extension
        ]

        if self.bags_list:
            print("{} bag(s) found".format(len(self.bags_list)))
        else:
            raise FileNotFoundError(
                "No files with .bag extension on the following path: {}".format(
                    self.root_dir
                )
            )

    def extract_bags(self):
        '''
        - extracts the topics of interest within three categories [image, signal and pointcloud]
        - output folders are named <bag_name>/<topic_name>/...
        '''

        for bag_path in self.bags_list:

            # create a folder with bag name in data/raw
            bag_name = Path(bag_path).stem
            os.makedirs(os.path.join(self.export_dir, bag_name), exist_ok=True)

            bag = rosbag.Bag(bag_path)
            self._write_images(bag, bag_name)
            # self._write_signal(bag, bag_name)
            self._write_pcl(bag, bag_name)

            bag.close()

    def _write_images(self, bag, bag_name):
        '''
        - converting bagmessages to opencv images
        - opencv issues: boost_cvbridge error if using cvbridge for decoding message
        - workaround based on https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
        - specific to encoding types
        '''

        for topic in self.image_topics:

            # os.join throws error if topic starts with '/'
            topic_dir = os.path.join(self.export_dir, bag_name, topic[1:])
            os.makedirs(topic_dir, exist_ok=True)

            topic_read = bag.read_messages(topic)
            print("reading topic <-- {}".format(topic))

            for i, msg in enumerate(tqdm(topic_read)):

                msg = msg.message

                if msg.encoding == "rgb8":
                    channels = 3
                    dtype = np.dtype("uint8") # hardcode 8bits
                elif msg.encoding == "16UC1":
                    channels = 1
                    dtype = np.dtype("uint8")
                else:
                    raise TypeError(
                        "image encoding problem, found {}".format(msg.encoding)
                    )

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

                cv2.imwrite(os.path.join(topic_dir, "{}.png".format(i)), image)

    def _write_signal(self, bag, bag_name):

        for topic in self.signal_topics:

            topic_read = bag.read_messages(topic)
            print("reading topic <-- {}".format(topic))


            out_csv = open(os.path.join(self.export_dir, bag_name, topic[1:] + ".csv"), "w")
            out_csv.write('# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z\n')  



            for i, msg in enumerate(tqdm(topic_read)):

                msg = msg.message

                out_csv.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (i,
                        msg.header.stamp.to_sec(),
                        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))


    def _write_pcl(self, bag, bag_name):

        for topic in self.pcl_topics:
            '''
            [x, y, z, intensity, ring, time]
            '''

            # os.join throws error if topic starts with '/'
            topic_dir = os.path.join(self.export_dir, bag_name, topic[1:])
            os.makedirs(topic_dir, exist_ok=True)

            topic_read = bag.read_messages(topic)
            print("reading topic <-- {}".format(topic))

            for i, (topic, msg, t) in enumerate(tqdm(topic_read)):
                pc_array = pointcloud2_to_xyz_array(msg) # NOTE: missing intensity values, could be extracted

                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = o3d.core.Tensor(pc_array)

                o3d.t.io.write_point_cloud(os.path.join(topic_dir,"{}.pcd".format(i)), pcd)

def main():

    unpack = unPackROSBag()
    unpack.extract_bags()


if __name__ == "__main__":
    main()
