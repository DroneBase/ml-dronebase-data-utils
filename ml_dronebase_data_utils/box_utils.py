import logging
import math
from typing import List, Union

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

        top_left = pxl_points_sorted[0]
        top_right = pxl_points_sorted[1]
        bottom_right = pxl_points_sorted[2]
        bottom_left = pxl_points_sorted[3]

        logging.info(f"top left = {top_left}")
        logging.info(f"top right = {top_right}")
        logging.info(f"bottom right = {bottom_right}")
        logging.info(f"bottom left = {bottom_left}")

        vec1_mag = np.linalg.norm(top_left - top_right)
        vec2_mag = np.linalg.norm(top_left - bottom_left)

        if vec1_mag > vec2_mag:
            h, w = vec1_mag, vec2_mag
            if top_left[1] > top_right[1]:
                x1, y1 = top_left
                x2, y2 = top_right
                logging.info(f"point1 = top left: {top_left}")
                logging.info(f"point2 = top right: {top_right}")
            else:
                x1, y1 = top_right
                x2, y2 = top_left
                logging.info(f"point1 = top right: {top_right}")
                logging.info(f"point2 = top left: {top_left}")
        else:
            h, w = vec2_mag, vec1_mag
            if top_left[0] >= bottom_left[0]:
                x1, y1 = bottom_left
                x2, y2 = top_left
                logging.info(f"point1 = bottom left: {bottom_left}")
                logging.info(f"point2 = top left: {top_left}")
            else:
                x1, y1 = top_left
                x2, y2 = bottom_left
                logging.info(f"point1 = top left: {top_left}")
                logging.info(f"point2 = bottom left: {bottom_left}")

        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2

        angle = calculate_angle(x1, y1, x2, y2)
        logging.info(f"After angle = {angle}")
        logging.info(f"xc, yc, w, h = [{xc}, {yc}, {w}, {h}]")
        logging.info(
            f"xmin, ymin, xmax, ymax, angle = [{xmin}, {ymin}, {xmax}, {ymax}, {angle}]"
        )

        boxes.append([xmin, ymin, xmax, ymax, angle])

    return np.asarray(boxes)


def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
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
        x1 (float): x coordinate of rotation point
        y1 (float): y coordinate of rotation point
        x2 (float): x coordinate of reference point
        y2 (float): y coordinate of reference point

    Returns:
        float: orientation angle
    """
    angle = np.degrees(np.arctan2(x1 - x2, y1 - y2))
    logging.info(f"Before angle = {angle}")
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

"""
Convert rotated boxes to vertices

:param boxes: The boxes to convert
:param box_mode: The format used for the box, either 'XYWHA_ABS' or 'XYXYA_ABS'
:param classes: The classes for each box. Need to sort this as well.

:return: Returns the vertices. if classes is provided returns both vertices and classes
"""
def rotated_boxes_to_vertices(
    boxes: Union[np.ndarray, List[List[float]]], box_mode: str = "XYWHA_ABS", classes: List[str] = []
) -> np.ndarray:
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
        logging.info(f"center points before = {center_points}")
        logging.info(f"center argmax index = {np.argmax(center_points[:, 1])}")
        logging.info(f"center argmax = {center_points[np.argmax(center_points[:, 1])]}")
        logging.info(f"point1 before = {points_x_sorted[1]}")
        bottom_center = center_points[np.argmax(center_points[:, 1])]
        logging.info(f"point1 after = {points_x_sorted[1]}")
        logging.info(f"center points after = {center_points}")
        logging.info(f"center argmin index = {np.argmin(center_points[:, 1])}")
        logging.info(f"center argmin = {center_points[np.argmin(center_points[:, 1])]}")
        logging.info(f"point2 before = {points_x_sorted[2]}")
        top_center = center_points[np.argmin(center_points[:, 1])]
        logging.info(f"point2 after = {points_x_sorted[2]}")
        points_x_sorted[1] = bottom_center
        points_x_sorted[2] = top_center
    logging.info(f"points x sorted = {points_x_sorted}")
    left_points = points_x_sorted[:2]
    right_points = points_x_sorted[2:]
    logging.info(f"left points = {left_points}")
    logging.info(f"right points = {right_points}")

    logging.info(f"left argmin = {np.argmin(left_points[:, 1])}")
    logging.info(f"left argmax = {np.argmax(left_points[:, 1])}")
    logging.info(f"right argmin = {np.argmin(right_points[:, 1])}")
    logging.info(f"right argmax = {np.argmax(right_points[:, 1])}")

    bottom_left = left_points[np.argmax(left_points[:, 1])]
    top_left = left_points[np.argmin(left_points[:, 1])]

    bottom_right = right_points[np.argmax(right_points[:, 1])]
    top_right = right_points[np.argmin(right_points[:, 1])]

    return np.stack((top_left, top_right, bottom_right, bottom_left))
