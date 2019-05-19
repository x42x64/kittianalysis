"""
Module with a collection of helpers to plot data from the Kitti dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d

import kitti_reader
from PIL import Image
from matplotlib.collections import PatchCollection

colormap = dict()
colormap['Car'] = 'red'
colormap['Van'] = 'orange'
colormap['Truck'] = 'yellow'
colormap['Pedestrian'] = 'blue'
colormap['Person_sitting'] = 'cyan'
colormap['Cyclist'] = 'green'
colormap['Tram'] = 'purple'
colormap['Misc'] = 'brown'
colormap['DontCare'] = 'white'

class CoveragePlot():
    """
    A class which creates a translucent birds-eye-view plot for detections
    """
    def __init__(self, data, ax, alpha=0.01):
        """
        The constructor creates the birds-eye-view.

        Args:
            data (list of dict): a list of detection dictionaries
            ax (object): the axes object to plot on
            alpha (float): the opaqueness of each rectangle to draw

        """
        patches = []
        for detection in data:
            patches.append(CoveragePlot._rotated_rectangle(detection['pos_x'],
                                                 detection['pos_z'],
                                                 detection['width'],
                                                 detection['length'],
                                                 detection['rot_y'],
                                                 alpha=alpha,
                                                 color=[0, 0, 0]))


        p = PatchCollection(patches,alpha=alpha)

        ax.add_collection(p)

    @staticmethod
    def _rotated_rectangle(pos_x, pos_y, width, length, angle, alpha, color):
        """
        Helper function to create a rectangle rotated around its center

        Args:
            pos_x (float): Position in x direction
            pos_y (float): Position in y direction
            width (float): Width of the rectangle
            length (float): Length of the rectangle
            angle (float): orientation of the rectangle in rad
            alpha (float): opaqueness of the rectangle
            color (list of float): list of 3 values to describe the RGB color of the rectangle

        Returns:
            object: a matplotlib.patches.Polygon which represents a rectangle rotated around its center at the specified
                    location
        """
        # create a 2D array with the edge points of the rectangle
        p = np.array(
                    [[-length/2.0, -length/2.0, length/2.0, length/2.0],
                     [-width/2.0, width/2.0, width/2.0, -width/2.0]]
                     )

        # rotation matrix
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

        # rotate the points
        transformed_p = np.matmul(rot, p) + np.array([[pos_x], [pos_y]])

        # return a Polygon object
        return matplotlib.patches.Polygon(transformed_p.transpose(), alpha=alpha, color=color )


def plot_2d_bbox(ax, detection, fill=False):
    """
    Plots the 2D bounding box of a detection onto a given axis
    Args:
        ax: axis to plot on
        detection: detection dictionary
        fill: if True, the bounding box is filled with the semi transparent label color

    """
    rect = patches.Rectangle((detection['left'], detection['top']),
                      detection['right'] - detection['left'], detection['bottom'] - detection['top'], linewidth=1,
                      edgecolor=colormap[detection['label']], facecolor='none')

    if fill:
        rect.set_facecolor(colormap[detection['label']])
        rect.set_alpha(0.3)

    ax.add_patch(rect)


def create_3d_bbox(detection):
    """

    Args:
        detection: detection dictionary

    Returns:
        A 3x8 numpy array representing all 8 edges of the bounding box

    """
    w = detection['width']
    h = detection['height']
    l = detection['length']
    x = detection['pos_x']
    y = detection['pos_y']
    z = detection['pos_z']
    ry = detection['rot_y']

    # compute rotational matrix around yaw axis
    R = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, +np.cos(ry)]])

    # 3D bounding box corners
    x_corners = np.array([0., l, l, l, l, 0., 0., 0.])  # -l/2
    y_corners = np.array([0., 0., h, h, 0., 0., h, h])  # -h
    z_corners = np.array([0., 0., 0., w, w, w, w, 0.])  # --w/2

    x_corners += -l / 2.0
    y_corners += -h
    z_corners += -w / 2.0

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x, y, z]).reshape((3, 1))

    return corners_3d


def plot_3d_bbox_in_image(ax, detection, calib, fill=False):
    """
    Projects and draws a 3D bounding box onto a given axes.
    Assumption: all corners of the box are in front of the image plane.

    Args:
        ax: axes to plot on
        detection: detection dictionary
        calib: calibration dictionary
        fill: if True, the faces of the box are filled with a semi-transparent color

    """

    # 3d corners of bounding box
    corners_3d = create_3d_bbox(detection)

    # 3d corners of bbox projected into image
    corners_3d_ext = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2d = calib['P2'].dot(corners_3d_ext)
    corners_2d = corners_2d / corners_2d[2]
    corners_2d = corners_2d[:2,:]

    faces = [
        [0,1,4,5],
        [1,2,3,4],
        [2,3,6,7],
        [6,7,0,5],
        [3,4,5,6],
        [0,1,2,7]
    ]

    for f in faces:
        verts = corners_2d[:, f]

        p = patches.Polygon(verts.T, color=colormap[detection['label']], fill=fill, alpha=0.2, linewidth=3)
        ax.add_patch(p)


def plot_lidar_in_image(ax, points, calib, img_width, img_height, point_size=8):
    """
    Function to plot lidar pointclouds into an image on an axes.

    Args:
        ax: axes object to draw on
        points: nx4 numpy array with laser points in x,y,z,reflectivity
        calib:  dictionary with calibration information (keys Tr_velo_to_cam, R0_rect and P2 are needed)
        img_width: width of the image to plot on
        img_height: height of the image to plot on
        point_size: The marker size in points**2


    """
    # camera calibration matrix (extrinsic calibration is done in a separate step to filter point)
    trafo = calib['P2'].dot(calib['R0_rect'])

    points_3d = points[:, :3].T # the 4th row is reflectivity
    points_3d_ext = np.vstack((points_3d, np.ones((points_3d.shape[-1]))))

    # get point cloud in cam coordinate system to filter all points behind the camera
    points_3d_cam = calib['Tr_velo_to_cam'].dot(points_3d_ext)
    idx = points_3d_cam[2,:] > 0.0

    reflectivity = points[idx,3]
    points_3d_cam_ext_front = points_3d_cam[:,idx]

    # transform into px coordinates
    points_2d_ext = trafo.dot(points_3d_cam_ext_front)
    points_2d = points_2d_ext / points_2d_ext[2]
    points_2d = points_2d[:2,:]

    # filter points which are not on the image
    idx = (points_2d[0,:] > 0.0) & (points_2d[0,:] < img_width) & (points_2d[1,:] > 0.0) & (points_2d[1,:] < img_height)

    ax.scatter(points_2d[0,idx], points_2d[1,idx], s=point_size, c=reflectivity[idx], alpha=0.7)

def plot_lidar_in_3d(ax, points, point_size=8, subsample=1, alpha=0.6):
    """
    Plots the subsampled lidar points on a 3d axes.
    Args:
        ax: axes to plot on
        points: nx4 numpy array containing x,y,z,intensity values
        point_size: the marker size in points**2
        subsample: every nth point is used for plotting. Use subsample=1 to plot every point.
        alpha: float for the transparency value (alpha)

    """
    points = points[0::subsample, :]
    ax.scatter(points[:,0], points[:,1], points[:,2], s=point_size, c=points[:,3], alpha=alpha)

def plot_info(ax, detection):
    """
    Puts the label next to the bounding box

    Args:
        ax: axes to plot on
        detection: detection dictionary

    Returns:

    """
    txt = detection['label']
    ax.text(detection['left'], detection['top'], txt, size=10,
             ha="left", va="top",
             bbox=dict(boxstyle="square",
                       edgecolor='none',
                       facecolor=colormap[detection['label']],
                       alpha=0.7
                       )
             )


def plot_frame_2d(ax, frame, show_image=True, show_2d=True, show_3d=False, show_lidar=False, show_info=True):
    """
    A helper function to draw bounding boxes of the detections onto the image

    Args:
        ax (object): axis to plot on
        img_path (str): path to the image file
        detections (list of dict): detection dictionary describing the detection

    Returns:
        object: axes with the plot
    """

    if show_image:
        # Display the image
        im = Image.open(frame['image']['path'])
        ax.imshow(im)

    if show_lidar:
        lidar = kitti_reader.KITTIFrameReader.get_velodyne(frame)
        plot_lidar_in_image(ax, lidar, frame['calibration'], frame['image']['width'], frame['image']['height'])
        if not show_image:
            ax.invert_yaxis()
            ax.set_aspect('equal')

    # Create a Rectangle patch
    for d in frame['detections']:
        if show_2d:
            plot_2d_bbox(ax, d, True)
        if show_info:
            plot_info(ax, d)
        if show_3d:
            if d['label'] != 'DontCare':
                plot_3d_bbox_in_image(ax, d, frame['calibration'], fill=show_image)

    return ax

def plot_3d_bbox_in_3d(ax, detection, calib, fill=False):
    """
    Plots a 3d bounding box into a 3d axes.

    Args:
        ax: axes to plot on
        detection: detection dictionary
        calib: calibration dictionary
        fill: True if the faces of the 3d bbox should be filled

    Returns:

    """
    # 3d corners of bounding box
    corners_3d_cam = create_3d_bbox(detection)

    # 3d corners transformed to velodyne coordinate system
    corners_3d_ext = np.vstack((corners_3d_cam, np.ones((corners_3d_cam.shape[-1]))))
    corners_3d_velo = np.linalg.inv(calib['Tr_velo_to_cam']).dot(corners_3d_ext)
    corners_3d_velo = corners_3d_velo[:3, :] / corners_3d_velo[3]

    faces = [
        [0, 1, 4, 5],
        [1, 2, 3, 4],
        [2, 3, 6, 7],
        [6, 7, 0, 5],
        [3, 4, 5, 6],
        [0, 1, 2, 7]
    ]

    for f in faces:
        pc = art3d.Poly3DCollection([list(zip(corners_3d_velo[0, f],corners_3d_velo[1, f],corners_3d_velo[2, f]))])
        pc.set_alpha(0.2)
        pc.set_facecolor(colormap[detection['label']])

        ax.add_collection3d(pc)


def plot_frame_3d(ax, frame, show_3d=True, show_lidar=True, show_info=True, subsample=4):
    """
    Function to visualize a complete frame on a 3d axes.

    Args:
        ax: 3d axes to plot on
        frame: frame which should be visualized
        show_3d: if True, the bounding boxes are plotted
        show_lidar: if True, the lidar points are plotted
        show_info: if True, labels for the bounding boxes are displayed Todo: not implemented

    """

    if show_lidar:
        lidar = kitti_reader.KITTIFrameReader.get_velodyne(frame)
        plot_lidar_in_3d(ax, lidar, point_size=2, subsample=subsample)

    for d in frame['detections']:
        if show_3d:
            if d['label'] != 'DontCare':
                plot_3d_bbox_in_3d(ax, d, frame['calibration'])

    ax.auto_scale_xyz([-80, 80], [-80, 80], [-80, 80])



def crop_detection(img_path, detection):
    """
    A helper function to extract the image patch with just the detection

    Args:
        img_path (str): path to the image file
        detection (dict): detection dictionary describing the detection

    Returns:
        object: a cropped PIL image
    """
    img = Image.open(img_path)
    area = (detection['left'], detection['top'], detection['right'], detection['bottom'])

    return img.crop(area)
