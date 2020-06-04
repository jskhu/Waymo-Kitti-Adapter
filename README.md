# Waymo_Kitti_Adapter
This is a tool converting [Waymo open dataset](https://waymo.com/open/) format to [Kitti dataset](http://www.cvlibs.net/datasets/kitti/) format. This tool was modified from the following repositories:
> Original Repository: https://github.com/Yao-Shao/Waymo_Kitti_Adapter
> Forked Repository: https://github.com/RocketFlash/Waymo_Kitti_Adapter
>
## Instruction
0. Download the [Waymo dataset](https://waymo.com/open/). Create a `Waymo` folder on your machine and organize the downloaded files as follows:
```
├── Waymo
│   ├── original
│   │   │──training
│   │   │   ├──training_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──training_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
│   │   │──validation
│   │   │   ├──validation_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──validation_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
│   │   │──testing
│   │   │   ├──testing_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──testing_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
```
1. Clone the [waymo open dataset repo](https://github.com/waymo-research/waymo-open-dataset) and follow the instructions on its [Quick Start Page](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) in order to build and test it.
2. Clone this repo to your computer, then copy the files from this repo to the `waymo-od` repo
```
cp Waymo-Kitti-Adapter/protocol_buffer/* waymo-od/waymo_open_dataset
cp Waymo-Kitti-Adapter/adapter.py waymo-od/
cp Waymo-Kitti-Adapter/adapter_lib.py waymo-od/
```

3. Copy adapter.py to `waymo-od` folder and set up the (our recommendation) following folder structure:
```
├── Waymo
│   ├── original
│   │   │──training
│   │   │   ├──training_0000, ...
│   │   │──validation
│   │   │   ├──validation_0000, ...
│   │   │──testing
│   │   │   ├──testing_0000...
│   ├── adapted
│   │   │──training
│   │   │──validation
│   │   │──testing
```
Create the `adapted` folder in `Waymo` directory with `training`, `validation` and `testing` folders inside. These will be locations that the adapter will save to.
4. Run `adapter.py` to save training data. Before running, open `adapter.py` and change the configurations at the top so that it suits to your own computer's path. Set `DATA_PATH` to be '{YOUR PATH}/Waymo/original/training' and `KITTI_PATH` to be '{YOUR PATH}/Waymo/adapted/training'.
```shell
python adapter.py
```
5. Run `adapter.py` to save validation data. Change `DATA_PATH` to be '{YOUR PATH}/Waymo/original/validation' and `KITTI_PATH` to be '{YOUR PATH}/Waymo/adapted/validation'.
6. Run `adapter.py` to save testing data. Change `DATA_PATH` to be '{YOUR PATH}/Waymo/original/testing' and `KITTI_PATH` to be '{YOUR PATH}/Waymo/adapted/testing'. Add the config option --test True.
```shell
python adapter.py --test True
```
4. Once completed, the folder tree will look like this:
```
...
├── Waymo
│   ├── original
│   │   │──training & testing & validation
│   ├── adapted
│   │   │──training
│   │   │   ├──calib & velodyne & label_0 & image_0
│   │   │──testing
│   │   │   ├──calib & velodyne & label_0
│   │   │──testing
│   │   │   ├──calib & velodyne & label_0 & image_0
```

## Data specification

### Cameras

Waymo dataset contains five cameras:

```
FRONT = 0;
FRONT_LEFT = 1;
FRONT_RIGHT = 2;
SIDE_LEFT = 3;
SIDE_RIGHT = 4;
```

all the names below with post-fix 0-4 is corresponding to these five cameras.

### Label

label_0 to label_4 contains label data for each camera and label_all fonder contain all the labels.

All in vehicle frame.

For each frame, here is the data specification:

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    camera_num	the camera number which the object belongs to, only exist
                     in label_all
```

### Calib

```
P0-P4 : intrinsic matrix for each camera
R0_rect : rectify matrix
Tr_velo_to_cam_0 - Tr_velo_to_cam_4 : transformation matrix from vehicle frame to camera frame
```

### Image

```
image_0 - image_4 : images for each
```

### Lidar

Point cloud in vehicle frame.

```
x y z intensity
```

For more details, see [readme.txt](https://github.com/Yao-Shao/Waymo_Kitti_Adapter/blob/master/KITTI/readme.txt) by KITTI.

## References

1. [Waymo open dataset](https://github.com/waymo-research/waymo-open-dataset)
2. [argoverse kitti adapter](https://github.com/yzhou377/argoverse-kitti-adapter)
