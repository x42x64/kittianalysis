"""
Module with a collection of helpers to plot data from the Kitti dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from PIL import Image
from matplotlib.collections import PatchCollection


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


def plot_detection(ax, img_path, detections, color):
    """
    A helper function to draw bounding boxes of the detections onto the image

    Args:
        ax (object): axis to plot on
        img_path (str): path to the image file
        detections (list of dict): detection dictionary describing the detection
        color: color specification

    Returns:
        object: axes with the plot
    """

    im = Image.open(img_path)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for d in detections:
        rect = patches.Rectangle((d['left'], d['top']), d['right']-d['left'], d['bottom']-d['top'], linewidth=1, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    return ax


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
