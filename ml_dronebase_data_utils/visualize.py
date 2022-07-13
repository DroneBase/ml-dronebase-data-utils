from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from .box_utils import rotated_boxes_to_vertices


def draw_boxes(
    image: Union[Image.Image, np.ndarray],
    boxes: Optional[Union[np.ndarray, List[List[float]]]],
    outline: str = "red",
) -> Image.Image:
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=outline)
    return image


def draw_rotated_boxes(
    image: Union[Image.Image, np.ndarray],
    boxes: Optional[Union[np.ndarray, List[List[float]]]],
    box_mode: str = "XYWHA_ABS",
    outline: str = "red",
    width: int = 3,
) -> Image.Image:
    polygons = rotated_boxes_to_vertices(boxes, box_mode)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    for poly in polygons:
        draw.polygon(
            xy=(
                (poly[0, 0], poly[0, 1]),
                (poly[1, 0], poly[1, 1]),
                (poly[2, 0], poly[2, 1]),
                (poly[3, 0], poly[3, 1]),
                (poly[0, 0], poly[0, 1]),
            ),
            outline=outline,
            width=width,
        )
    return image


def draw_lines(
    image: Union[Image.Image, np.ndarray],
    lines: Union[np.ndarray, List, List[Tuple[float]]],
    outline: str = "red",
) -> Image.Image:
    if not isinstance(lines, np.ndarray):
        lines = np.asarray(lines)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.chord(line, outline=outline)
    return image
