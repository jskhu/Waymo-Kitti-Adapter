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
"""Generates a files that contains serialized Objects proto for ground truth and predictions."""

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


def get_objects_from_file(file, type_):
    with open(file, "r") as f:
        lines = f.readlines()
    objects = [Object3d(line, type_) for line in lines]
    return objects


def create_bin(input_dir, output_dir, type_):

    samples = sorted(os.listdir(input_dir))
    objects = metrics_pb2.Objects()

    for sample in samples:

        sample_file = os.path.join(input_dir, sample)
        objects_ = get_objects_from_file(sample_file, type_)

        for obj in objects_:
            o = metrics_pb2.Object()
            o.context_name = (sample)
            o.frame_timestamp_micros = -1

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = obj.loc[0]
            box.center_y = obj.loc[1]
            box.center_z = obj.loc[2]

            # Check these
            box.length = obj.l
            box.width = obj.h
            box.height = obj.w

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
    create_bin(input_dir=args.preds, output_dir=args.output_dir, type_="preds", )
    create_bin(input_dir=args.gt, output_dir=args.output_dir, type_='gt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generaate image sets')
    parser.add_argument('--preds', type=str, default=None, help='Path to prediction text files')
    parser.add_argument('--gt', type=str, default=None, help='Path to ground truth text files')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output folder')
    args = parser.parse_args()
    main(args)
