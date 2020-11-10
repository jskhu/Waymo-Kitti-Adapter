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
"""A simple example to generate a file that contains serialized Objects proto."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

import pickle
import os
import pdb


def _create_pd_file_example():
    """Creates a prediction objects file."""
    objects = metrics_pb2.Objects()

    o = metrics_pb2.Object()
    # The following 3 fields are used to uniquely identify a frame a prediction
    # is predicted at. Make sure you set them to values exactly the same as what
    # we provided in the raw data. Otherwise your prediction is considered as a
    # false negative.
    o.context_name = ('context_name for the prediction. See Frame::context::name '
                      'in  dataset.proto.')
    # The frame timestamp for the prediction. See Frame::timestamp_micros in
    # dataset.proto.
    invalid_ts = -1
    o.frame_timestamp_micros = invalid_ts
    # This is only needed for 2D detection or tracking tasks.
    # Set it to the camera name the prediction is for.
    o.camera_name = dataset_pb2.CameraName.FRONT

    # Populating box and score.
    box = label_pb2.Label.Box()
    box.center_x = 0
    box.center_y = 0
    box.center_z = 0
    box.length = 0
    box.width = 0
    box.height = 0
    box.heading = 0
    o.object.box.CopyFrom(box)
    # This must be within [0.0, 1.0]. It is better to filter those boxes with
    # small scores to speed up metrics computation.
    o.score = 1
    # For tracking, this must be set and it must be unique for each tracked
    # sequence.
    o.object.id = 'unique object tracking ID'
    # Use correct type.
    pdb.set_trace()
    o.object.type = label_pb2.Label.TYPE_PEDESTRIAN

    objects.objects.append(o)

    # Add more objects. Note that a reasonable detector should limit its maximum
    # number of boxes predicted per frame. A reasonable value is around 400. A
    # huge number of boxes can slow down metrics computation.

    # Write objects to a file.
    f = open('/tmp/your_preds.bin', 'wb')
    f.write(objects.SerializeToString())
    f.close()


def save_preds():
    pred_infos = pickle.load(open('result_v5_80.pkl', "rb"))
    gt_infos = pickle.load(open('waymo_infos_val_v5_80.pkl', "rb"))
    objects = metrics_pb2.Objects()
    scores = []
    sample_idx = 0
    for sample in pred_infos:
        calib = gt_infos[sample_idx]['calib']
        for i in range(0, sample['num_example']):
            # [x, y, z, w, l, h, rz, gt_classes]
            pred_box = sample['boxes_lidar'][i]
            name = sample['name'][i]
            o = metrics_pb2.Object()
            o.context_name = (calib['Name'])
            o.frame_timestamp_micros = calib['Timestamp']

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = pred_box[0]
            box.center_y = pred_box[1]
            box.center_z = pred_box[2]
            box.width = pred_box[3]
            box.length = pred_box[4]
            box.height = pred_box[5]
            box.heading = pred_box[6]
            o.object.box.CopyFrom(box)
            # This must be within [0.0, 1.0]. It is better to filter those boxes with
            # small scores to speed up metrics computation.
            o.score = sample['score'][i]
            scores.append(sample['score'][i])
            # Use correct type
            class_name = 'TYPE_' + name
            o.object.type = getattr(label_pb2.Label, class_name)
            objects.objects.append(o)
        sample_idx += 1
    f = open('preds.bin', 'wb')
    f.write(objects.SerializeToString())
    f.close()


def save_gt():
    gt_infos = pickle.load(open('waymo_infos_val_v5_80.pkl', "rb"))
    objects = metrics_pb2.Objects()
    for sample in gt_infos:
        calib = sample['calib']
        annos = sample['annos']
        gt_boxes = annos['gt_boxes_lidar']
        names = annos['name']
        num_boxes = len(gt_boxes)
        for i in range(0, num_boxes):
            gt_box = gt_boxes[i]  # [x, y, z, w, l, h, rz, gt_classes]
            name = names[i]
            o = metrics_pb2.Object()
            o.context_name = (calib['Name'])
            o.frame_timestamp_micros = calib['Timestamp']

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = gt_box[0]
            box.center_y = gt_box[1]
            box.center_z = gt_box[2]
            box.width = gt_box[3]
            box.length = gt_box[4]
            box.height = gt_box[5]
            box.heading = gt_box[6]
            o.score = 0.5
            o.object.box.CopyFrom(box)
            o.object.detection_difficulty_level = annos['difficulty'][i]
            # Use correct type
            class_name = 'TYPE_' + name
            o.object.type = getattr(label_pb2.Label, class_name)
            objects.objects.append(o)
    f = open('val.bin', 'wb')
    f.write(objects.SerializeToString())
    f.close()


def read_bin():
    f = open('official_val.bin', 'rb')
    content = f.read()
    official_gt = metrics_pb2.Objects()
    official_gt.ParseFromString(content)
    f.close()
    f = open('val.bin', 'rb')
    content2 = f.read()
    gt = metrics_pb2.Objects()
    gt.ParseFromString(content2)
    f = open('fake_ground_truths.bin', 'rb')
    content3 = f.read()
    objects3 = metrics_pb2.Objects()
    objects3.ParseFromString(content3)
    f = open('fake_predictions.bin', 'rb')
    content4 = f.read()
    objects4 = metrics_pb2.Objects()
    objects4.ParseFromString(content4)
    counter = 0
    counter_all = 0
    pdb.set_trace()


def main():
    #_create_pd_file_example()
    # read_bin()
    save_preds()
    save_gt()


if __name__ == '__main__':
    main()
