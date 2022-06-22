import logging
import math
from typing import List, Union

import numpy as np
from shapely.geometry import Polygon


def vertices_to_boxes(vertices: np.ndarray) -> np.ndarray:
    boxes = []
    for v in vertices:
        xmin = min(v[:, 0])
        ymin = min(v[:, 1])
        xmax = max(v[:, 0])
        ymax = max(v[:, 1])

        boxes.append([xmin, ymin, xmax, ymax])

    return np.asarray(boxes)


def vertices_to_rotated_boxes(vertices: np.ndarray) -> np.ndarray:
    """Convert vertices to rotated boxes.

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

        logging.debug(f"top left = {top_left}")
        logging.debug(f"top right = {top_right}")
        logging.debug(f"bottom right = {bottom_right}")
        logging.debug(f"bottom left = {bottom_left}")

        vec1_mag = np.linalg.norm(bottom_right - top_right)
        vec2_mag = np.linalg.norm(bottom_right - bottom_left)

        if vec1_mag < vec2_mag:
            if bottom_right[0] <= top_right[0]:
                x2, y2 = top_right
                x1, y1 = bottom_right
                logging.debug(f"point1 = bottom right: {bottom_right}")
                logging.debug(f"point2 = top right: {top_right}")
            else:
                x2, y2 = bottom_right
                x1, y1 = top_right
                logging.debug(f"point1 = top right: {top_right}")
                logging.debug(f"point2 = bottom right: {bottom_right}")
            w, h = vec1_mag, vec2_mag
        else:
            x2, y2 = bottom_right
            x1, y1 = bottom_left
            logging.debug(f"point1 = bottom left: {bottom_left}")
            logging.debug(f"point2 = bottom right: {bottom_right}")
            w, h = vec2_mag, vec1_mag

        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2

        angle = calculate_angle(x1, y1, x2, y2)
        logging.debug(f"After angle = {angle}")
        logging.debug(f"xc, yc, w, h = [{xc}, {yc}, {w}, {h}]")
        logging.debug(
            f"xmin, ymin, xmax, ymax, angle = [{xmin}, {ymin}, {xmax}, {ymax}, {angle}]"
        )

        boxes.append([xmin, ymin, xmax, ymax, angle])

    return np.asarray(boxes)


def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute the orientation angle of a rotated rectangle. `angle` is the
    angle required to rotate the box from a straight position to its original rotated position.

    Adapted from detectron2.structures.rotated_boxes
    angle in range: (-90, 90] degrees

    1. When angle == 0:
        box_rotated == box
    2. When angle > 0:
        box_rotated is obtained by rotating box w.r.t its center by :math:`|angle| % 90` degrees CCW;
    3. When angle < 0
        box_rotated is obtained by rotating box w.r.t its center by :math:`|angle| % 90` degrees CW.

    Args:
        x1 (float): x coordinate of rotation point
        y1 (float): y coordinate of rotation point
        x2 (float): x coordinate of reference point
        y2 (float): y coordinate of reference point

    Returns:
        float: orientation angle
    """
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    logging.debug(f"Before angle = {angle}")
    if (-90 <= angle) and (angle <= 90):
        return angle
    elif angle > 90:
        return angle % 90
    else:
        return -(angle % 90)


def boxes_to_vertices(boxes: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    num_instances = len(boxes)

    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)

    # Display in largest to smallest order to reduce occlusion.
    areas = boxes[:, 2] * boxes[:, 3]

    sorted_idxs = np.argsort(-areas).tolist()
    # Re-order overlapped instances in descending order.
    boxes = boxes[sorted_idxs]

    vertices = []
    for i in range(num_instances):
        v = extract_vertices(boxes[i])
        vertices.append(v)
    return np.asarray(vertices)


def rotated_boxes_to_vertices(
    boxes: Union[np.ndarray, List[List[float]]], box_mode: str = "XYWHA_ABS"
) -> np.ndarray:
    num_instances = len(boxes)

    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)

    # Display in largest to smallest order to reduce occlusion.
    areas = boxes[:, 2] * boxes[:, 3]

    sorted_idxs = np.argsort(-areas).tolist()
    # Re-order overlapped instances in descending order.
    boxes = boxes[sorted_idxs]

    vertices = []
    for i in range(num_instances):
        v = extract_rotated_vertices(boxes[i], box_mode)
        vertices.append(v)
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
        x = x_delta * c - y_delta * s + xc
        y = x_delta * s + y_delta * c + yc
        vertex = (x, y)
        vertices.append(vertex)
    return vertices


def sort_points(points: np.ndarray) -> np.ndarray:
    """Sort points into top left, top right, bottom right, and bottom left box
    coordinates.

    left points:
        top left: left point with max y coordinate
        bottom left: left point with min y coordinate

    right points:
        top right: right point with max y coordinate
        bottom right: right point with min y coordinate

    Args:
        points (np.ndarray): A 4x2 matrix of arbitrarily ordered box coordinates.
            (e.g., [[x1, y1], [x3, y3], [x2, y2], [x4, y4]])

    Returns:
        np.ndarray: A 4x2 matrix of sorted box coordinates.
    """
    points_x_sorted = points[np.argsort(points[:, 0])]

    if points_x_sorted[1, 0] == points_x_sorted[2, 0]:
        center_points = points_x_sorted[1:3]
        points_x_sorted[1] = center_points[np.argmin(center_points[:, 1])]
        points_x_sorted[2] = center_points[np.argmax(center_points[:, 1])]
    left_points = points_x_sorted[:2]
    right_points = points_x_sorted[2:]

    bottom_left = left_points[np.argmin(left_points[:, 1])]
    top_left = left_points[np.argmax(left_points[:, 1])]

    bottom_right = right_points[np.argmin(right_points[:, 1])]
    top_right = right_points[np.argmax(right_points[:, 1])]

    return np.stack((top_left, top_right, bottom_right, bottom_left))
