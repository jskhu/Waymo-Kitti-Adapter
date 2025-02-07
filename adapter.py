import argparse
import os
from pathlib import Path

import math
# import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from parso import split_lines
import tensorflow as tf
import progressbar

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from adapter_lib import *

import pdb
############################Config###########################################
# path to waymo dataset "folder" (all .tfrecord files in that folder will
# be converted)
RAW_DATA_PATH = '/mnt/storage/datasets/waymo/raw_data'
IMAGESET_PATH = '/home/jordan/trail/FusionDet/data/waymo/ImageSets/train.txt'
# path to save kitti dataset
KITTI_PATH = '/mnt/storage/datasets/waymo/fusion/kitti_format_v2/training'
# location filter, use this to convert your preferred location
LOCATION_FILTER = False
LOCATION_NAME = ['location_sf']
# max indexing length
INDEX_LENGTH = 15
# as name
IMAGE_FORMAT = 'png'
# do not change
LABEL_PATH = KITTI_PATH + '/label_0'
CAM_LABEL_PATH = KITTI_PATH + '/cam_label_0'
LABEL_ALL_PATH = KITTI_PATH + '/label_all'
IMAGE_PATH = KITTI_PATH + '/image_0'
CALIB_PATH = KITTI_PATH + '/calib'
LIDAR_PATH = KITTI_PATH + '/velodyne'
IMG_CALIB_PATH = KITTI_PATH + '/img_calib'
###############################################################################


class Adapter:

    def __init__(self):
        self.__lidar_list = ['_FRONT', '_FRONT_RIGHT',
                             '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.__type_list = ['UNKNOWN', 'VEHICLE',
                            'PEDESTRIAN', 'SIGN', 'CYCLIST']
        self.__file_names = []
        self.T_front_cam_to_ref = []
        self.T_vehicle_to_front_cam = []

    def cvt(self, args, data_records, start_ind):
        """ convert dataset from Waymo to KITTI
        Args:
        return:
        """
        self.start_ind = start_ind
        self.set_file_names(RAW_DATA_PATH, data_records)
        print("Converting ..." + RAW_DATA_PATH)

        self.create_folder(args.camera_type)

        bar = progressbar.ProgressBar(maxval=len(self.__file_names) + 1,
                                      widgets=[progressbar.Percentage(), ' ',
                                               progressbar.Bar(
                                                   marker='>', left='[', right=']'), ' ',
                                               progressbar.ETA()])

        # tf.enable_eager_execution()
        file_num = 1
        frame_num = 0
        frame_name = self.start_ind
        label_exists = False
        print("start converting ...")
        bar.start()
        for file_idx, file_name in enumerate(self.__file_names):
            print('File {}/{}'.format(file_idx, len(self.__file_names)))
            dataset = tf.data.TFRecordDataset(file_name, compression_type='')
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                if (frame_num % args.keyframe) == 0:
                    if LOCATION_FILTER == True and frame.context.stats.location not in LOCATION_NAME:
                        continue
                    if args.test == False:
                        label_exists = self.save_label(frame, frame_name, args.camera_type, False, True)

                    if args.test == label_exists:
                        frame_num += 1
                        continue

                    self.save_calib(frame, frame_name)

                    self.save_label(
                        frame, frame_name, args.camera_type)

                    # Save 2d labels labelled in image, NOT projected lidar labels
                    # Does not handle args.camera_type == all
                    self.save_cam_label(frame, frame_name, args.camera_type)

                    self.save_image(frame, frame_name, args.camera_type)

                    self.save_lidar(frame, frame_name)

                    self.save_image_calib(frame, frame_name)

                    # print("image:{}\ncalib:{}\nlidar:{}\nlabel:{}\n".format(str(s1-e1),str(s2-e2),str(s3-e3),str(s4-e4)))
                    frame_name += 1

                frame_num += 1
            bar.update(file_num)
            file_num += 1
        bar.finish()
        print("\nfinished ...")
        return frame_name

    def save_image(self, frame, frame_num, cam_type):
        """ parse and save the images in png format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
        """
        for img in frame.images:
            if cam_type == 'all' or cam_type == str(img.name - 1):
                img_path = IMAGE_PATH + '/' + \
                    str(frame_num).zfill(INDEX_LENGTH) + '.' + IMAGE_FORMAT
                img = cv2.imdecode(np.frombuffer(
                    img.image, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                plt.imsave(img_path, rgb_img, format=IMAGE_FORMAT)

    def save_calib(self, frame, frame_num, kitti_format=True):
        """ parse and save the calibration data
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
        """
        fp_calib = open(CALIB_PATH + '/' +
                        str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+')

        self.T_front_cam_to_ref = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ])

        camera_calib = []
        R0_rect = ["%e" % i for i in np.eye(3).flatten()]
        Tr_velo_to_cam = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            tmp = np.array(camera.extrinsic.transform).reshape(4, 4)
            tmp = self.cart_to_homo(self.T_front_cam_to_ref) @ np.linalg.inv(tmp)
            Tr_velo_to_cam.append(["%e" % i for i in tmp[:3,:].reshape(12)])

        for cam in frame.context.camera_calibrations:
            tmp = np.zeros((3, 4))
            tmp[0, 0] = cam.intrinsic[0]
            tmp[1, 1] = cam.intrinsic[1]
            tmp[0, 2] = cam.intrinsic[2]
            tmp[1, 2] = cam.intrinsic[3]
            tmp[2, 2] = 1
            tmp = list(tmp.reshape(12))
            tmp = ["%e" % i for i in tmp]
            camera_calib.append(tmp)

        T_front_cam_to_vehicle = np.array(frame.context.camera_calibrations[0].extrinsic.transform).reshape(4, 4)
        self.T_vehicle_to_front_cam = np.linalg.inv(T_front_cam_to_vehicle)

        for i in range(5):
            calib_context += "P" + str(i) + ": " + \
                " ".join(camera_calib[i]) + '\n'
        calib_context += "R0_rect" + ": " + " ".join(R0_rect) + '\n'
        for i in range(5):
            calib_context += "Tr_velo_to_cam_" + \
                str(i) + ": " + " ".join(Tr_velo_to_cam[i]) + '\n'
        calib_context += "timestamp_micros: " + \
            str(frame.timestamp_micros) + '\n'
        calib_context += "context_name: " + str(frame.context.name) + '\n'
        fp_calib.write(calib_context)
        fp_calib.close()

    def save_lidar(self, frame, frame_num):
        """ parse and save the lidar data in psd format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
                """
        range_images, range_image_top_pose = self.parse_range_image_and_camera_projection(
            frame)

        points, intensity, elongation = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            range_image_top_pose)
        points_all = np.concatenate(points, axis=0)
        intensity_all = np.concatenate(intensity, axis=0)
        elongation_all = np.concatenate(elongation, axis=0)
        point_cloud = np.column_stack((points_all, intensity_all, elongation_all))
        pc_path = LIDAR_PATH + '/' + \
            str(frame_num).zfill(INDEX_LENGTH) + '.bin'
        point_cloud.tofile(pc_path)

    def save_label(self, frame, frame_num, cam_type, kitti_format=False, check_label_exists = False):
        """ parse and save the label data in .txt format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
                """


        # get point cloud in the frame
        range_images, range_image_top_pose = self.parse_range_image_and_camera_projection(
            frame)

        points, intensity, elongation = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            range_image_top_pose)
        points_all = tf.convert_to_tensor(
            np.concatenate(points, axis=0), dtype=np.float32)

        # preprocess bounding box data
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        Tr_velo_to_cam = []
        recorded_label = []
        label_lines = ''
        label_all_lines = ''
        """
        if kitti_format:
            for camera in frame.context.camera_calibrations:
                tmp = np.array(camera.extrinsic.transform).reshape(4, 4)
                tmp = np.linalg.inv(tmp)
                axes_transformation = np.array([[0, -1, 0, 0],
                                                [0, 0, -1, 0],
                                                [1, 0, 0, 0],
                                                [0, 0, 0, 1]])
                tmp = np.matmul(axes_transformation, tmp)
                Tr_velo_to_cam.append(tmp)
        """
        for obj in frame.laser_labels:
            # caculate bounding box
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.__lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break
            if bounding_box == None or name == None:
                continue
            box = tf.convert_to_tensor(
                [obj.box.center_x, obj.box.center_y, obj.box.center_z, obj.box.length, obj.box.width, obj.box.height, obj.box.heading], dtype=np.float32)
            box = tf.reshape(box, (1, 7))
            num_points = box_utils.compute_num_points_in_box_3d(
                points_all, box)
            num_points = num_points.numpy()[0]
            detection_difficulty = obj.detection_difficulty_level
            my_type = self.__type_list[obj.type]
            truncated = 0
            occluded = 0
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height/2

            if check_label_exists == False:
                pt_ref = self.cart_to_homo(self.T_front_cam_to_ref) @ self.T_vehicle_to_front_cam @ np.array([x,y,z,1]).reshape((4,1))
                x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi/2

            beta = math.atan2(x, z)
            alpha = (rotation_y + beta - math.pi / 2) % (2 * math.pi)

            # save the labels
            line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(round(truncated, 2),
                                                                                         occluded,
                                                                                         round(
                alpha, 2),
                round(
                bounding_box[0], 2),
                round(
                bounding_box[1], 2),
                round(
                bounding_box[2], 2),
                round(
                bounding_box[3], 2),
                round(
                height, 2),
                round(
                width, 2),
                round(
                length, 2),
                round(
                x, 2),
                round(
                y, 2),
                round(
                z, 2),
                round(
                rotation_y, 2),
                num_points,
                detection_difficulty)
            line_all = line[:-1] + ' ' + name + '\n'
            # store the label
            label_all_lines += line_all
            if (name == cam_type):
                label_lines += line
                recorded_label.append(line)

        if len(recorded_label) == 0:
            return False
        else:
            fp_label_all = open(LABEL_ALL_PATH + '/' +
                            str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+')
            fp_label = open(LABEL_PATH + '/' +
                                str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+')
            fp_label.write(label_lines)
            fp_label.close()
            fp_label_all.write(label_all_lines)
            fp_label_all.close()
            return True

    def save_cam_label(self, frame, frame_num, cam_type, kitti_format=False, check_label_exists = False):
        """ parse and save the label data in .txt format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
                """


        label_lines = ''
        recorded_label = []
        cam_type = int(cam_type)

        for labels in frame.camera_labels:
            name = labels.name - 1
            if name != cam_type:
                continue
            for label in labels.labels:
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                my_type = self.__type_list[label.type]
                line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    -1,
                    -1,
                    -10,
                    round(
                    bbox[0], 2),
                    round(
                    bbox[1], 2),
                    round(
                    bbox[2], 2),
                    round(
                    bbox[3], 2),
                    -1,
                    -1,
                    -1,
                    -1000,
                    -1000,
                    -1000,
                    -10,
                    1.0)
                label_lines += line
                recorded_label.append(line)
        fp_label = open(CAM_LABEL_PATH + '/' +
                            str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+')
        fp_label.write(label_lines)
        fp_label.close()
            
    def save_image_calib(self, frame, frame_num):
        fp_image_calib = open(IMG_CALIB_PATH + '/' +
                              str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+')
        camera_calib = []
        pose = []
        velocity = []
        timestamp = []
        shutter = []
        trigger_time = []
        readout_done_time = []
        calib_context = ''

        for camera in frame.images:
            tmp = np.array(camera.pose.transform).reshape((16,))
            pose.append(["%e" % i for i in tmp])
            tmp = np.zeros(6)
            tmp[0] = camera.velocity.v_x
            tmp[1] = camera.velocity.v_y
            tmp[2] = camera.velocity.v_z
            tmp[3] = camera.velocity.w_x
            tmp[4] = camera.velocity.w_y
            tmp[5] = camera.velocity.w_z
            velocity.append(["%e" % i for i in tmp])
            timestamp.append(camera.pose_timestamp)
            shutter.append(camera.shutter)
            trigger_time.append(camera.camera_trigger_time)
            readout_done_time.append(camera.camera_readout_done_time)

        for i in range(5):
            calib_context += "Pose_" + str(i) + ": " + \
                " ".join(pose[i]) + '\n'
        for i in range(5):
            calib_context += "Velocity_" + str(i) + ": " + \
                " ".join(velocity[i]) + '\n'
        for i in range(5):
            calib_context += "Timestamp_" + str(i) + ": " + \
                " ".join(velocity[i]) + '\n'
        for i in range(5):
            calib_context += "Shutter_" + str(i) + ": " + \
                " ".join(velocity[i]) + '\n'
        for i in range(5):
            calib_context += "Trigger_" + str(i) + ": " + \
                " ".join(velocity[i]) + '\n'
        for i in range(5):
            calib_context += "Readout_" + str(i) + ": " + \
                " ".join(velocity[i]) + '\n'
        fp_image_calib.write(calib_context)
        fp_image_calib.close()

    def set_file_names(self, raw_data_path, data_records):
        for data_record in data_records:
            self.__file_names.append(raw_data_path + '/' + data_record)

    def get_file_names(self, folder):
        for i in os.listdir(folder):
            if i.split('.')[-1] == 'tfrecord':
                self.__file_names.append(folder + '/' + i)

    def cart_to_homo(self, mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

    def create_folder(self, cam_type):
        dirs = [KITTI_PATH, CALIB_PATH, LIDAR_PATH, LABEL_ALL_PATH, IMG_CALIB_PATH, IMAGE_PATH, LABEL_PATH, CAM_LABEL_PATH]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    def extract_intensity(self, frame, range_images, lidar_num):
        """ extract the intensity from the original range image
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :param lidar_num: the number of current lidar
                :return:
                """
        intensity_0 = np.array(range_images[lidar_num][0].data).reshape(-1, 4)
        intensity_0 = intensity_0[:, 1]
        intensity_1 = np.array(range_images[lidar_num][
                               1].data).reshape(-1, 4)[:, 1]
        return intensity_0, intensity_1

    def image_show(self, data, name, layout, cmap=None):
        """Show an image."""
        plt.subplot(*layout)
        plt.imshow(tf.image.decode_jpeg(data), cmap=cmap)
        plt.title(name)
        plt.grid(False)
        plt.axis('off')

    def parse_range_image_and_camera_projection(self, frame):
        """Parse range images and camera projections given a frame.
        Args:
           frame: open dataset frame proto
        Returns:
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
        """
        self.__range_images = {}
        # camera_projections = {}
        # range_image_top_pose = None
        for laser in frame.lasers:
            if len(laser.ri_return1.range_image_compressed) > 0:
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name] = [ri]

                if laser.name == open_dataset.LaserName.TOP:
                    range_image_top_pose_str_tensor = tf.io.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                    range_image_top_pose = open_dataset.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytearray(range_image_top_pose_str_tensor.numpy()))

                # camera_projection_str_tensor = tf.io.decode_compressed(
                #     laser.ri_return1.camera_projection_compressed, 'ZLIB')
                # cp = open_dataset.MatrixInt32()
                # cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                # camera_projections[laser.name] = [cp]
            if len(laser.ri_return2.range_image_compressed) > 0:
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name].append(ri)
                #
                # camera_projection_str_tensor = tf.io.decode_compressed(
                #     laser.ri_return2.camera_projection_compressed, 'ZLIB')
                # cp = open_dataset.MatrixInt32()
                # cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                # camera_projections[laser.name].append(cp)
        return self.__range_images, range_image_top_pose

    def plot_range_image_helper(self, data, name, layout, vmin=0, vmax=1, cmap='gray'):
        """Plots range image.
        Args:
          data: range image data
          name: the image title
          layout: plt layout
          vmin: minimum value of the passed data
          vmax: maximum value of the passed data
          cmap: color map
        """
        plt.subplot(*layout)
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(name)
        plt.grid(False)
        plt.axis('off')

    def get_range_image(self, laser_name, return_index):
        """Returns range image given a laser name and its return index."""
        return self.__range_images[laser_name][return_index]

    def show_range_image(self, range_image, layout_index_start=1):
        """Shows range image.
        Args:
          range_image: the range image data from a given lidar of type MatrixFloat.
          layout_index_start: layout offset
        """
        range_image_tensor = tf.convert_to_tensor(range_image.data)
        range_image_tensor = tf.reshape(
            range_image_tensor, range_image.shape.dims)
        lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
        range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                      tf.ones_like(range_image_tensor) * 1e10)
        range_image_range = range_image_tensor[..., 0]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        self.plot_range_image_helper(range_image_range.numpy(), 'range',
                                     [8, 1, layout_index_start], vmax=75, cmap='gray')
        self.plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                                     [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
        self.plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                                     [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

    def convert_range_image_to_point_cloud(self, frame, range_images, range_image_top_pose, ri_index=0):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
          intensity: {[N, 1]} list of intensity of length 5 (number of lidars).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        # lasers = sorted(frame.lasers, key=lambda laser: laser.name)
        points = []
        # cp_points = []
        intensity = []
        elongation = []

        frame_pose = tf.convert_to_tensor(
            np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[...,
                                        0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[
            ..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min,
                                 c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(
                    beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            intensity_tensor = tf.gather_nd(range_image_tensor,
                                            tf.where(range_image_mask))
            # cp = camera_projections[c.name][0]
            # cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            # cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
            points.append(points_tensor.numpy())
            # cp_points.append(cp_points_tensor.numpy())
            intensity.append(intensity_tensor.numpy()[:, 1])
            elongation.append(intensity_tensor.numpy()[:, 2])

        return points, intensity, elongation

    def rgba(self, r):
        """Generates a color based on range.
        Args:
          r: the range value of a given point.
        Returns:
          The color for a given range
        """
        c = plt.get_cmap('jet')((r % 20.0) / 20.0)
        c = list(c)
        c[-1] = 0.5  # alpha
        return c

    def plot_image(self, camera_image):
        """Plot a cmaera image."""
        plt.figure(figsize=(20, 12))
        plt.imshow(tf.image.decode_jpeg(camera_image.image))
        plt.grid("off")

    def plot_points_on_image(self, projected_points, camera_image, rgba_func, point_size=5.0):
        """Plots points on a camera image.
        Args:
          projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
          camera_image: jpeg encoded camera image.
          rgba_func: a function that generates a color from a range value.
          point_size: the point size.
        """
        self.plot_image(camera_image)

        xs = []
        ys = []
        colors = []

        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append(rgba_func(point[2]))

        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Save Waymo dataset into Kitti format')
    parser.add_argument('--keyframe',
                        type=int,
                        default=10,
                        help='Saves every specified # of scenes. Default is 1 and the program saves every scene')
    parser.add_argument('--camera_type',
                        type=str,
                        default="0",
                        help='Select camera views to save. Input argument from 0 to 4 or all')
    parser.add_argument('--start_ind',
                        type=int,
                        default=0,
                        help='File number starts counting from this index')
    parser.add_argument('--test',
                        type=bool,
                        default=False,
                        help='if true, does not save any ground truth data')
    args = parser.parse_args()
    start_ind = args.start_ind
    with open(IMAGESET_PATH, 'r') as f:
        data_records = f.read().splitlines()
    # path, dirs, files = next(os.walk(DATA_PATH))
    adapter = Adapter()
    last_ind = adapter.cvt(args, data_records, start_ind)
    # dirs.sort()
    # for directory in dirs:
    #     adapter = Adapter()
    #     last_ind = adapter.cvt(args, directory, start_ind)
    #     start_ind = last_ind
