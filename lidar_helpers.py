"""
A module containing a collection of helper functions for point cloud processing
"""

import numpy as np

def is_point_in_tetrahydron(tetra, p):
    """
    checks if point p is inside the tetrahydron. See http://steve.hollasch.net/cgindex/geometry/ptintet.html

    Args:
        tetra: 4x3 numpy matrix
        p: 3x1 numpy vector

    Returns:
        True if point is within or on the tetrahydron, False otherwise

    """

    M0 = np.hstack((tetra, np.ones((4,1))))
    M1 = M0.copy()
    M2 = M0.copy()
    M3 = M0.copy()
    M4 = M0.copy()


    M1[0,:3] = p.T
    M2[1,:3] = p.T
    M3[2,:3] = p.T
    M4[3,:3] = p.T

    D0 = np.linalg.det(M0)

    if D0 != 0.0:
        D1 = np.linalg.det(M1)
        D2 = np.linalg.det(M2)
        D3 = np.linalg.det(M3)
        D4 = np.linalg.det(M4)

        determinants = np.array([D0, D1, D2, D3, D4])

        # all determinants have the same sign or is 0.0
        return np.all(np.logical_or(np.sign(determinants)==np.sign(D0),determinants == 0.0))


    return False

def is_point_in_tetrahydron2(tetra, p):
    """
    Optimized version of is_point_in_tetrahydron

    Args:
        tetra: 4x3 numpy matrix
        p: 3x1 numpy vector

    Returns:
        True if point is within or on the tetrahydron, False otherwise

    """

    M0 = np.hstack((tetra, np.ones((4,1))))

    D0 = np.linalg.det(M0)
    sD0 = np.sign(D0)

    if D0 != 0.0:
        M1 = M0.copy()
        M1[0, :3] = p.T
        D1 = np.linalg.det(M1)

        if sD0 == np.sign(D1) or D1 == 0.0:
            M1 = M0.copy()
            M1[1, :3] = p.T
            D1 = np.linalg.det(M1)

            if sD0 == np.sign(D1) or D1 == 0.0:
                M1 = M0.copy()
                M1[2, :3] = p.T
                D1 = np.linalg.det(M1)

                if sD0 == np.sign(D1) or D1 == 0.0:
                    M1 = M0.copy()
                    M1[3, :3] = p.T
                    D1 = np.linalg.det(M1)

                    if sD0 == np.sign(D1) or D1 == 0.0:
                        return True

    return False


def is_point_in_bbox(bbox, p):
    """
    Checks if point p is within the boundingbox bbox
    Args:
        bbox: 3x8 array describing the corners of the bounding box
        p: 3d vector of the point which should be checked.

    Returns:

    """

    tetra = []
    tetra.append(bbox[:,[1,3,4,5]])
    tetra.append(bbox[:,[3,5,6,7]])
    tetra.append(bbox[:,[0,1,5,7]])
    tetra.append(bbox[:,[1,2,3,7]])
    tetra.append(bbox[:,[0,2,4,6]])

    for t in tetra:
        if is_point_in_tetrahydron2(t.T, p):
            return True

    return False


def get_object_points(points, bbox):

    object_points = []

    x_min = bbox[0,:].min()
    x_max = bbox[0,:].max()

    y_min = bbox[1, :].min()
    y_max = bbox[1, :].max()

    z_min = bbox[2, :].min()
    z_max = bbox[2, :].max()

    for i in range(points.shape[0]):
        if x_min <= points[i, 0] <= x_max and y_min <= points[i, 1] <= y_max and z_min <= points[i, 2] <= z_max and \
            is_point_in_bbox(bbox, points[i, :3]):
            object_points.append(points[i, :])

    if object_points:
        return np.stack(object_points)
    else:
        return None