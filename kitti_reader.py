"""
KITTI 3D Object detection reader module

This module contains classes to load the KITTI 3D Object detection dataset
into python data structures.
Additionally a data enhancer class is provided to enhance the provided dataset
with information from 3rd party data sources.

"""

import copy
import csv
import glob
import numpy as np
import os
import pykitti
from PIL import Image


class KITTIFrameReader:
    """
    A class to read the kitti dataset into a python dictionary
    """

    def __init__(self, basepath):
        """Initializes KITTIFrameReader.

        The basepath is expected to provide following tree of data:
            basepath
                |-training
                    |-image_2
                        |-000000.png
                        |-...
                    |-label_2
                        |-000000.txt
        Args:
            basepath (str):


        """
        if KITTIFrameReader._validate(basepath):
            self.basepath = basepath
        else:
            raise ValueError("Basepath does not contain the expected data")

    def read_all_frames(self):
        """Function which reads all KITTI frames

        This function reads all frames of the KITTI 3D Object detection dataset.


        Returns:
            :obj:`list` of :obj:`dict`: list of dicts containing information about the frame.
                                        The dictionary has following structure:
                                        "image": `dict`
                                            'id' (str): id of the frame within the dataset
                                            'path' (str): path to the image of the frame
                                            'width' (int): width of the image
                                            'height' (int): height of the image

                                        "detections": `list` of `dict`
                                            'label' (str): Describes the type of object: 'Car', 'Van', 'Truck',
                                                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                                    'Misc' or 'DontCare'
                                            'left','right','top', 'bottom' (int): 2D bounding box of object in the
                                                    image:
                                                    contains left, top, right, bottom pixel coordinates
                                            'truncated' (float): Float from 0 (non-truncated) to 1 (truncated), where
                                                    truncated refers to the object leaving image boundaries
                                            'occluded' (int): Integer (0,1,2,3) indicating occlusion state:
                                                    0 = fully visible, 1 = partly occluded, 2 = largely occluded,
                                                    3 = unknown
                                            'alpha' (float): Observation angle of object, ranging [-pi..pi]
                                            'height', 'width', 'length' (float): 3D object dimensions: height, width,
                                                    length (in meters)
                                            'pos_x', 'pos_y', 'pos_z' (float): 3D object location x,y,z in camera
                                                    coordinates (in meters)
                                            'rot_y': Rotation ry around Y-axis in camera coordinates [-pi..pi]

        """
        image_ids = self._get_image_ids()

        if not len(image_ids):
            raise RuntimeError('There is no data in the dataset (no image ids)')

        first_image_id = image_ids[0]
        image_ext = self._find_image_ext(first_image_id)

        frames = [self.read_frame(image_name, image_ext) for image_name in image_ids]

        return frames

    def read_frame(self, frame_id, image_ext='png'):
        frame = dict()
        frame['image'], frame['detections'] = self._get_image_detection(frame_id, image_ext)
        frame['calibration'] = self._get_calibration(frame_id)
        frame['velodyne'] = {'path': os.path.join(self.basepath, 'training', 'velodyne', frame_id + '.bin')}
        return frame

    def _get_image_detection(self, frame_id, image_ext='png'):
        """Extract all information for a given frame

        Args:
            frame_id (str): Id of the frame wich should be loaded as string (e.g. '000000').
            image_ext (str): Extension of the image files.

        Returns:
            dict: Dictionary containing information about the image of the frame
                  (for details see read_all_frames)
            list of dict: Dictionaries containing information about the detections within the frame
                          (for details see read_all_frames)

        """
        detections_path = "{}/training/label_2/{}.txt".format(self.basepath, frame_id)
        detections = self._get_detections(detections_path)
        image_path = "{}/training/image_2/{}.{}".format(self.basepath, frame_id, image_ext)
        image_width, image_height = self._image_dimensions(image_path)
        return {
                'id': frame_id,
                'path': image_path,
                'width': image_width,
                'height': image_height
               }, \
               detections

    def _get_image_ids(self):
        """Get all available ids

        Provide all ids of the labels within the class' instance basepath.

        Returns:
            list of str: A list of all available ids as string

        """

        path = "{}/training/label_2/".format(self.basepath)
        return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(path, "*.txt"))]

    def _find_image_ext(self, image_id):
        """Finds the image extension of a given frame id

        Args:
            image_id (str): The id of the frame as string (e.g. '000000').

        Returns:
            str: The image extension, e.g. 'png' or 'jpg'

        """
        for image_ext in ['png', 'jpg']:
            if os.path.exists("{}/training/image_2/{}.{}".format(self.basepath, image_id, image_ext)):
                return image_ext

        raise Exception("could not find jpg or png for {} at {}/training/image_2".format(image_id, self.basepath))

    @staticmethod
    def _get_detections(detections_path):
        """Parses a KITTI 3D Object detection label file

        Args:
            detections_path (str): Path to the label file

        Returns:
            dict: A dictionary containing all keys of the label file

        """
        detections = []
        with open(detections_path) as f:
            f_csv = csv.reader(f, delimiter=' ')
            for row in f_csv:
                x1, y1, x2, y2 = map(float, row[4:8])
                truncated = float(row[1])
                occluded = int(row[2])
                alpha = float(row[3])
                height, width, length = map(float, row[8:11])
                pos_x, pos_y, pos_z = map(float, row[11:14])
                rot_y = float(row[14])
                label = row[0]
                detections.append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2,
                    'truncated': truncated,
                    'occluded': occluded,
                    'alpha': alpha,
                    'height': height,
                    'width': width,
                    'length': length,
                    'pos_x': pos_x,
                    'pos_y': pos_y,
                    'pos_z': pos_z,
                    'rot_y': rot_y

                })

        return detections

    @staticmethod
    def _image_dimensions(path):
        """Provides the image dimensions for a given image path

        Args:
            path (str): Path to an image file.

        Returns:
            int: width of the image in px
            int: height of the image in px

        """
        with Image.open(path) as image:
            return image.width, image.height

    @staticmethod
    def _validate(basepath):
        """Checks if the given path contains the required folder structure

        Args:
            basepath (string): The path which should be checked for the folder structure

        Returns:
            bool: True if the given path satisfies the required folder structure, false otherwise.

        """
        expected_dirs = [
            'training/image_2',
            'training/label_2',
            'training/velodyne',
            'training/calib'
        ]
        for subdir in expected_dirs:
            if not os.path.isdir(os.path.join(basepath, subdir)):
                print( "Expected subdirectory {} within {}".format(subdir, basepath))
                return False

        return True

    def _get_calibration(self, frame_id):

        # extract calibration data from txt file
        fn = os.path.join(self.basepath, "training", "calib", frame_id + '.txt')
        cal_raw = {}
        with open(fn, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    cal_raw[key] = np.array([float(x) for x in value.split()])

        cdata = dict()
        # expand to a 4x4 matrix
        cdata['Tr_velo_to_cam'] = np.zeros((4, 4))
        cdata['Tr_velo_to_cam'][3, 3] = 1
        cdata['Tr_velo_to_cam'][:3, :4] = cal_raw['Tr_velo_to_cam'].reshape(3, 4)

        # expand to a 4x4 matrix
        cdata['Tr_imu_to_velo'] = np.zeros((4, 4))
        cdata['Tr_imu_to_velo'][3, 3] = 1
        cdata['Tr_imu_to_velo'][:3, :4] = cal_raw['Tr_imu_to_velo'].reshape(3, 4)

        # expand to a 4x4 matrix
        cdata['R0_rect'] = np.zeros((4, 4))
        cdata['R0_rect'][3, 3] = 1
        cdata['R0_rect'][:3, :3] = cal_raw['R0_rect'].reshape(3, 3)

        # reshape to a 3x4 matrix
        for k in ['P0', 'P1', 'P2', 'P3']:
            cdata[k] = cal_raw[k].reshape(3,4)

        return cdata

    @staticmethod
    def get_velodyne(frame):
        return np.fromfile(frame['velodyne']['path'], dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def create_new_frame_from_detections(org_frame, detections):
        """
        Creates a new frame dictionary only containing the specified detections
        Args:
            org_frame: frame dictionary
            detections: list of detection dictionaries

        Returns:
            a copy of org_frame, but only with the detections specified in detections

        """
        new_frame = copy.deepcopy(org_frame)
        new_frame['detections'] = detections

        return new_frame


class KITTIRawEnhancer():
    """
    A class to enhance the 3D Object Detection dataset with raw data from the KITTI dataset
    """
    def __init__(self, basepath, mappingpath):
        """Initializes KITTIFrameReader.

                The basepath is expected to provide following tree of data, which is created by the
                raw_data_downloader.sh script when downloading the KITTI raw data:
                    basepath
                    |-raw_data_downloader
                        |-2011_09_26
                            |-2011_09_26_drive_0001_sync
                                |-oxts
                                |-...
                    |-...

                The mappingpath is the path to the KITTI mapping folder of the object development kit from KITTI
                (https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)
                    mappingpath
                    |-train_mapping.txt
                    |-train_rand.txt


                Args:
                    basepath (str): path to the folder which contains the raw_data_downloader folder
                    mappingpath (str): path to the folder which contains the mapping files between the dataset frames
                                       and the raw dataset frames (can be obtained from the object development kit)


                """
        self.basepath = os.path.join(basepath, 'raw_data_downloader')
        self.mappingpath = mappingpath
        self.id_mapping = self._get_id_mapping()

    def _get_id_mapping(self):
        """
        A function which creates a ordered list of dicts to provide easy access to the information which is needed
        to locate the correct frame in the raw data by just using the object detection frame id.

        Returns:
             list of dict: A list of dictionaries containing date, drive and frame respectively for the dataset frame
                           at the given location. E.g. date of the 314th frame (0 based) in the dataset:
                           ret[314]['date']
        """
        raw_id = []
        with open(os.path.join(self.mappingpath, 'train_mapping.txt')) as f:
            f_csv = csv.reader(f, delimiter=' ')
            for i, row in enumerate(f_csv):
                raw_info = {
                    'date': row[0],
                    'drive': row[1].split("_")[4],
                    'frame': int(row[2])
                }
                raw_id.append(raw_info)

        with open(os.path.join(self.mappingpath, 'train_rand.txt')) as f:
            ids = f.read().strip().split(',')
            dataset_id = [raw_id[int(id) - 1] for id in ids]

        return dataset_id

    def raw_exists_for_frame(self, frame):
        """
        A helper function which checks, if the raw dataset exists for a given frame

        Args:
            frame (dict): the frame dictionary for which the existence of raw data should be checked

        Returns:
            bool: True if raw data exists, False otherwise
        """
        raw_info = self.id_mapping[int(frame['image']['id'])]

        drive = raw_info['date'] + '_drive_' + raw_info['drive'] + '_sync'
        raw_data_path = os.path.join(self.basepath, raw_info['date'], drive)

        return os.path.isdir(raw_data_path)


    def enhance_oxts(self, frame):
        """
        Add oxts data (IMU) to the given frame

        Args:
            frame (dict): the frame which should be enhanced

        Returns:
            dict: the same frame dictionary, but now with an additional key "oxts", which contains a list of
                  pykitti.oxts objects.
        """
        raw_info = self.id_mapping[int(frame['image']['id'])]
        ds = pykitti.raw(self.basepath, raw_info['date'], raw_info['drive'], frames=[raw_info['frame']])

        frame['oxts'] = ds.oxts

        return frame
