import logging
import math
from typing import List, Tuple, Union

import numpy as np
from shapely.geometry import Polygon


def vertices_to_boxes(vertices: np.ndarray) -> np.ndarray:
    """Convert vertices to boxes.

    Args:
        vertices (np.ndarray): A Nx4x2 matrix containing the vertices to be converted to boxes.

    Returns:
        np.ndarray: A Nx4 matrix containing the converted boxes.
    """
    boxes = []
    for v in vertices:
        polygon = Polygon(v)
        boxes.append(list(polygon.bounds))

    return np.asarray(boxes)


def vertices_to_rotated_boxes(vertices: np.ndarray) -> np.ndarray:
    """Convert vertices to rotated boxes.

    VOC defined bounding box with orientation angle:
        [xmin, ymin, xmax, ymax, angle]

    The box coordinates `[xmin, ...]` describe the box when rotated to an upright orientation which
    is defined when the larger side of the box (the height vector `h`) is parallel to the positive y axis
    and the shorter side of the box (the width vector `w`) is parallel to he positive x axis.

    e.g.,
        tan(h) = 0 degrees

    The angle describes the rotation angle required to rotate the box coordinates from an
    upright orientation to its true orientation.
    This angle is defined in the range (-90, 90] degrees.

    Args:
        vertices (np.ndarray): A Nx4x2 matrix containing the vertices to be converted to boxes.

    Returns:
        np.ndarray: A Nx5 matrix containing the converted boxes.
    """
    boxes = []
    for v in vertices:
        xc, yc = Polygon(v).centroid.xy
        yc = yc[0]
        xc = xc[0]

        pxl_points_sorted = sort_points(v)

        tl = pxl_points_sorted[0]
        tr = pxl_points_sorted[1]
        bl = pxl_points_sorted[3]

        angle, h, w = rotated_box_dims(tl, tr, bl)

        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2

        logging.debug(f"After angle = {angle}")
        logging.debug(f"xc, yc, w, h = [{xc}, {yc}, {w}, {h}]")
        logging.debug(
            f"xmin, ymin, xmax, ymax, angle = [{xmin}, {ymin}, {xmax}, {ymax}, {angle}]"
        )

        boxes.append([xmin, ymin, xmax, ymax, angle])

    return np.asarray(boxes)


def rotated_box_dims(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray) -> Tuple[float]:
    """Compute the angle, height, and width dimensions of a rotated box given the
    top left, top right, and bottom left vertices. The angle is the angle between
    the height vector and the top-down y axis in image space. The height vector is
    obtained from one of the following conditions:

    If the magnitude of the vector between the top left and top right vertices is larger
    than the magnitude of the vector between the top left and bottom left vertices,

        e.g., `|| tl - tr || > || tl - bl ||`,
        h = || tl - tr ||

        1. If the top left coordinate is parallel to or higher than the top right coordinate,

            origin point = top right
            reference point = top left

            e.g.:

                        tl o
                           ^
                            \
            tl o<--o tr,     \
                              o tr
        2. If the top left coordinate is lower than the top right coordinate,

            origin point = top left
            reference point = top right

            e.g.:

                  o tr
                  ^
                 /
                /
            tl o

    Otherwise, the height vector is between the top left and bottom left vertices,

    3. In this case, the height vector is always oriented from the bottom left to top left vertices,

    e.g.:

    tl o               o tl
       ^               ^
        \             /                                                                 # noqa: W605
         \     ,     /                                                                  # noqa: W605
          o bl   bl o

    Args:
        tl (np.ndarray): _description_
        tr (np.ndarray): _description_
        bl (np.ndarray): _description_

    Returns:
        Tuple[float]: _description_
    """
    v1 = np.linalg.norm(tl - tr)
    v2 = np.linalg.norm(tl - bl)

    if (v1 > v2) and (tl[1] <= tr[1]):
        logging.debug(f"point1 = top right: {tr}")
        logging.debug(f"point2 = top left: {tl}")
        angle = calculate_angle(tr, tl)
        return angle, v1, v2
    elif (v1 > v2) and (tl[1] > tr[1]):
        logging.debug(f"point1 = top left: {tl}")
        logging.debug(f"point2 = top right: {tr}")
        angle = calculate_angle(tl, tr)
        return angle, v1, v2
    else:
        logging.debug(f"point1 = bottom left: {bl}")
        logging.debug(f"point2 = top left: {tl}")
        angle = calculate_angle(bl, tl)
        return angle, v2, v1


def calculate_angle(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute the orientation angle of a rotated rectangle. `angle` is the
    angle required to rotate the box from an upright orientation to its true orientation.

    Adapted from detectron2.structures.rotated_boxes
    angle in range: (-90, 90] degrees

    1. When angle in (-90, 90]:
        box_rotated is obtained by rotating box w.r.t its center by :math:`|angle|` degrees CCW;
    2. When angle > 90:
        box_rotated is obtained by rotating box w.r.t its center by :math:`|angle| % 90` degrees CCW;
    3. When angle < -90
        box_rotated is obtained by rotating box w.r.t its center by :math:`-|angle| % 90` degrees CCW.

    Args:
        point1 (np.ndarray): x, y coordinates of origin rotation point
        point2 (np.ndarray): x, y coordinates of reference rotation point

    Returns:
        float: orientation angle
    """
    x1, y1 = point1
    x2, y2 = point2
    angle = np.degrees(np.arctan2(x1 - x2, y1 - y2))
    logging.debug(f"Before angle = {angle}")
    if (-90 < angle) and (angle <= 90):
        return angle
    elif angle > 90:
        return angle % 90
    else:
        return -(angle % 90)


def boxes_to_vertices(boxes: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    num_instances = len(boxes)

    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)

    vertices = []
    for i in range(num_instances):
        v = extract_vertices(boxes[i])
        vertices.append(v)
    return np.asarray(vertices)


def rotated_boxes_to_vertices(
    boxes: Union[np.ndarray, List[List[float]]],
    box_mode: str = "XYWHA_ABS",
    classes: List[str] = [],
) -> np.ndarray:
    """
    Convert rotated boxes to vertices

    :param boxes: The boxes to convert
    :param box_mode: The format used for the box, either 'XYWHA_ABS' or 'XYXYA_ABS'
    :param classes: The classes for each box. Need to sort this as well.

    :return: Returns the vertices. if classes is provided returns both vertices and classes
    """
    num_instances = len(boxes)

    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)

    # Display in largest to smallest order to reduce occlusion.
    areas = boxes[:, 2] * boxes[:, 3]

    sorted_idxs = np.argsort(-areas).tolist()
    # Re-order overlapped instances in descending order.
    boxes = boxes[sorted_idxs]
    # Re-order classes as well
    if len(classes) and not isinstance(classes, np.ndarray):
        classes = np.asarray(classes)
        classes = classes[sorted_idxs]
        classes = classes.tolist()

    vertices = []
    for i in range(num_instances):
        v = extract_rotated_vertices(boxes[i], box_mode)
        vertices.append(v)
    if len(classes):
        return np.asarray(vertices), classes
    else:
        return np.asarray(vertices)


def extract_vertices(box: np.ndarray) -> List[List[float]]:
    xmin, ymin, xmax, ymax = box
    vertices = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return vertices


def extract_rotated_vertices(
    box: np.ndarray, box_mode: str = "XYWHA_ABS"
) -> List[List[float]]:
    if box_mode == "XYWHA_ABS":
        xc, yc, w, h, angle = box
    elif box_mode == "XYXYA_ABS":
        xmin, ymin, xmax, ymax, angle = box
        w = xmax - xmin
        h = ymax - ymin
        xc = xmin + w / 2
        yc = ymin + h / 2
    else:
        raise ValueError(
            f"Box mode {box_mode} is not supported.\
            Must either be `XYWHA_ABS` or `XYXYA_ABS`."
        )

    # angle is the number of degrees the box is rotated CCW w.r.t. the 0-degree box
    theta = angle * math.pi / 180.0
    c = math.cos(theta)
    s = math.sin(theta)
    deltas = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]

    vertices = []
    for k in range(4):
        x_delta = deltas[k][0]
        y_delta = deltas[k][1]
        x = y_delta * s + x_delta * c + xc
        y = y_delta * c - x_delta * s + yc
        vertex = (x, y)
        vertices.append(vertex)
    return vertices


def sort_points(points: np.ndarray) -> np.ndarray:
    """Sort points into top left, top right, bottom right, and bottom left box
    coordinates.

    left points:
        top left: left point with min y coordinate
        bottom left: left point with max y coordinate

    right points:
        top right: right point with min y coordinate
        bottom right: right point with max y coordinate

    Args:
        points (np.ndarray): A 4x2 matrix of arbitrarily ordered box coordinates.
            (e.g., [[x1, y1], [x3, y3], [x2, y2], [x4, y4]])

    Returns:
        np.ndarray: A 4x2 matrix of sorted box coordinates.
    """
    points_x_sorted = points[np.argsort(points[:, 0])]

    if points_x_sorted[1, 0] == points_x_sorted[2, 0]:
        center_points = points_x_sorted[1:3].copy()
        bottom_center = center_points[np.argmax(center_points[:, 1])]
        top_center = center_points[np.argmin(center_points[:, 1])]
        points_x_sorted[1] = bottom_center
        points_x_sorted[2] = top_center

    left_points = points_x_sorted[:2]
    right_points = points_x_sorted[2:]

    bottom_left = left_points[np.argmax(left_points[:, 1])]
    top_left = left_points[np.argmin(left_points[:, 1])]

    bottom_right = right_points[np.argmax(right_points[:, 1])]
    top_right = right_points[np.argmin(right_points[:, 1])]

    logging.debug(f"top left = {top_left}")
    logging.debug(f"top right = {top_right}")
    logging.debug(f"bottom right = {bottom_right}")
    logging.debug(f"bottom left = {bottom_left}")

    return np.stack((top_left, top_right, bottom_right, bottom_left))
