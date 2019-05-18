"""
Module which contains functions to sanity check the KITTI 3D Object detection data
"""
import glob
import os
import numpy as np


def structure_check(basepath, check_for_raw_data=True):
    """Checks if the given path contains the required folders, files and optionally raw data

    Args:
        basepath (string): The path which should be checked for the folder structure
        check_for_raw_data (bool): if true, the existence of the needed raw data is also checked

    Returns:
        bool: True if the given path satisfies the required folder structure, false otherwise.
        list of str: all ids for which data is complete

    """
    if check_for_raw_data:
        from kitti_reader import KITTIRawEnhancer

    expected_dirs = [
        'training/image_2',
        'training/label_2',
        'training/velodyne',
        'training/calib'
    ]
    if check_for_raw_data:
        expected_dirs.append("raw_data_downloader")

    for subdir in expected_dirs:
        if not os.path.isdir(os.path.join(basepath, subdir)):
            print("Expected subdirectory {} within {}".format(subdir, basepath))
            return False, []


    # get ids for all data as lists of strings
    img_ids = [os.path.splitext(os.path.basename(f))[0] for
               f in glob.glob(os.path.join(basepath, "training", "image_2", "*.png"))]

    lab_ids = [os.path.splitext(os.path.basename(f))[0] for
               f in glob.glob(os.path.join(basepath, "training", "label_2", "*.txt"))]

    lidar_ids = [os.path.splitext(os.path.basename(f))[0] for
                 f in glob.glob(os.path.join(basepath, "training", "velodyne", "*.bin"))]

    calib_ids = [os.path.splitext(os.path.basename(f))[0] for
                 f in glob.glob(os.path.join(basepath, "training", "calib", "*.txt"))]



    # check all against the label ids (= is there data for each label)
    missing_img = [id for id in img_ids if id not in lab_ids]
    missing_lidar = [id for id in lidar_ids if id not in lab_ids]
    missing_calib = [id for id in calib_ids if id not in lab_ids]
    missing_raw = []


    if missing_img:
        print("Missing following image ids in dataset: {}".format(missing_img))

    if missing_lidar:
        print("Missing following lidar ids in dataset: {}".format(missing_lidar))

    if missing_calib:
        print("Missing following calibration ids in dataset: {}".format(missing_calib))

    # check also, if raw data exists for the given label ids
    if check_for_raw_data:
        enhancer_raw = KITTIRawEnhancer(basepath, os.path.join(basepath, 'mapping'))
        raw_ids = [id for id in lab_ids if enhancer_raw.raw_exists_for_frame(id)]
        missing_raw = [id for id in raw_ids if id not in lab_ids]
        if missing_raw:
            print("Missing raw data for following ids in dataset: {}".format(missing_raw))

    # return false if there are any missing datasets
    if missing_img or missing_lidar or missing_calib or missing_raw:
        # return the intersection of all id lists
        if check_for_raw_data:
            return False, list(set(img_ids) & set(lab_ids) & set(lidar_ids) & set(calib_ids) & set(raw_ids))
        else:
            return False, list(set(img_ids) & set(lab_ids) & set(lidar_ids) & set(calib_ids))

    return True, lab_ids


def check_label(label):
    """
    Checks if a given label is valid
    Args:
        label: label to check

    Returns: True if label is exactly one of the 9 KITTI labels, False otherwise

    """
    return label in ['Car', 'Van', 'Truck',
                 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                 'Misc', 'DontCare']


def check_2d_bounding_box(left, right, top, bottom, img_width, img_height):
    """
    Checks if bounding box is valid. A bounding box is considered valid if
    * bounding box has positive area (zero width or height is forbidden)
    * bounding box is not inverted (right > left, bottom > top)
    * bounding box boundaries are on the image

    Args:
        left: left boundary in pixel
        right:
        top:
        bottom:
        img_width:
        img_height:

    Returns:
        True if the bounding box is considered valid, False otherwise

    """
    if left >= right:
        return False
    if top >= bottom:
        return False

    if left < 0 or top < 0:
        return False

    if right >= img_width or bottom >= img_height:
        return False

    return True


def check_truncated(val):
    """
    Checks if the truncated value is valid
    Args:
        val: value to check

    Returns:
        True if the value is between or equal to 0 or 1

    """
    return 0.0 <= val <= 1.0


def check_occlusion(val):
    """
    Checks if the occlusion value is valid. According to the documentation:
    Integer (0,1,2,3) indicating occlusion state:
    0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown

    Args:
        val: occlusion value

    Returns:
        True if occlusion value is 0,1,2 or 3. False otherwise.
    """
    return 0 <= val <= 3


def check_angle(val):
    """
    Checks if angle value is between -PI and +PI

    Args:
        val: value to check

    Returns:
        True if angle is within range. False otherwise
    """
    return -np.pi <= val <= np.pi


def check_position(x,y,z):
    """
    Checks the position of 3D bounding boxes.
    Args:
        x [m]: x value in cam coordinates (left/right)
        y [m]: y value in cam coordinates  (up/down)
        z [m]: z value in cam coordinates (back/front)

    Returns:
        True if position is within reasonable boundaries. False otherwise.

    """
    return (0.0 <= z <= 150.0) and (-10.0 <= y <= 10.0) and (-50.0 <= x <= 50.0)

def check_dimensions(height, width, length, label):
    if label == "Car":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Van":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Truck":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Pedestrian":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Person_sitting":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Cyclist":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Tram":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)
    elif label == "Misc":
        return (0.0 <= length <= 150.0) and (-10.0 <= width <= 10.0) and (-50.0 <= height <= 50.0)

    return False