# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""Generates files that contains serialized Objects proto for ground truth and predictions."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

import argparse
import os
import numpy as np
import math


def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))


class Object3d(object):

    def __init__(self, line, type_):
        self.label = line.strip().split(' ')
        self.src = line
        self.cls_type = self.label[0]
        self.truncation = float(self.label[1])
        self.occlusion = float(self.label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(self.label[3])
        self.box2d = np.array((float(self.label[4]),
                               float(self.label[5]),
                               float(self.label[6]),
                               float(self.label[7])), dtype=np.float32)
        self.h = float(self.label[8])
        self.w = float(self.label[9])
        self.l = float(self.label[10])
        self.loc = np.array((float(self.label[11]), float(self.label[12]), float(self.label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(self.label[14])

        self.num_pts = -1.0
        self.score = 1.0

        if type_ == "preds":
            self.score = sigmoid(float(self.label[15]))
        elif type_ == "gt":
            self.num_pts = int(self.label[15])
        else:
            raise NotImplementedError


class Calibration(object):

    def __init__(self, calib_file):

        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[0].strip().split(' ')[1:]
        self.P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[5].strip().split(' ')[1:]
        self.R0 = np.array(obj, dtype=np.float32).reshape(3, 3)

        obj = lines[6].strip().split(' ')[1:]
        self.V2C = np.array(obj, dtype=np.float32).reshape(3, 4)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]


def get_objects_from_file(file, type_):
    with open(file, "r") as f:
        lines = f.readlines()
    objects = [Object3d(line, type_) for line in lines]
    return objects


def create_bin(input_dir, output_dir, type_, calib_dir):

    samples = sorted(os.listdir(input_dir))
    objects = metrics_pb2.Objects()

    for sample in samples:
        print("Processing sample {}".format(sample))

        sample_file = os.path.join(input_dir, sample)
        calib_file = os.path.join(calib_dir, sample)

        objects_ = get_objects_from_file(sample_file, type_)
        calib = Calibration(calib_file)

        for obj in objects_:
            o = metrics_pb2.Object()
            o.context_name = (sample)
            o.frame_timestamp_micros = -1

            # Populating box and score.
            box = label_pb2.Label.Box()
            loc_rect = np.expand_dims(obj.loc, axis=0)
            loc_lidar = calib.rect_to_lidar(loc_rect).squeeze()
            box.center_x = loc_lidar[0]
            box.center_y = loc_lidar[1]
            box.center_z = loc_lidar[2]

            box.length = obj.l
            box.width = obj.w
            box.height = obj.h

            box.heading = obj.ry
            o.object.box.CopyFrom(box)

            if type_ == "gt":
                # Add num pts
                o.object.num_lidar_points_in_box = obj.num_pts
                if obj.num_pts <= 0:
                    continue

            o.score = obj.score

            # Use correct type
            class_name = 'TYPE_' + obj.cls_type
            o.object.type = getattr(label_pb2.Label, class_name)
            objects.objects.append(o)

    output_file = os.path.join(output_dir, "{}.bin".format(type_))
    print("writing to {}".format(output_file))
    with open(output_file, 'wb') as f:
        f.write(objects.SerializeToString())


def main(args):
    create_bin(input_dir=args.preds, output_dir=args.output_dir, type_="preds", calib_dir=args.calib)
    create_bin(input_dir=args.gt, output_dir=args.output_dir, type_='gt', calib_dir=args.calib)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generaate image sets')
    parser.add_argument('--preds', type=str, default=None, help='Path to prediction text files')
    parser.add_argument('--gt', type=str, default=None, help='Path to ground truth text files')
    parser.add_argument('--calib', type=str, default=None, help='Path to calibration files')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output folder')
    args = parser.parse_args()
    main(args)
