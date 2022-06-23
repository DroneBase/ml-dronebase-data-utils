import os

import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from rasterio.io import DatasetReader
from tqdm import tqdm

from .box_utils import vertices_to_boxes, vertices_to_rotated_boxes
from .pascal_voc import PascalVOCWriter
from .s3 import upload_file


def geo_to_voc(ortho_path: str, geo_path: str, save_path: str):
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    writer = PascalVOCWriter(ortho_path, ortho.width, ortho.height)

    vertices = get_pixel_vertices(ortho, gdf)
    boxes = vertices_to_boxes(vertices)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        writer.addObject("panel", xmin, ymin, xmax, ymax)

    if "s3://" in save_path:
        writer.save("annot.xml")
        upload_file("annot.xml", save_path)
        os.remove("annot.xml")
    else:
        writer.save(save_path)


def geo_to_rotated_voc(ortho_path: str, geo_path: str, save_path: str):
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    writer = PascalVOCWriter(ortho_path, ortho.width, ortho.height)

    vertices = get_pixel_vertices(ortho, gdf)
    boxes = vertices_to_rotated_boxes(vertices)

    for box in boxes:
        xmin, ymin, xmax, ymax, angle = box
        writer.addObject("panel", xmin, ymin, xmax, ymax, angle)

    if "s3://" in save_path:
        writer.save("annot.xml")
        upload_file("annot.xml", save_path)
        os.remove("annot.xml")
    else:
        writer.save(save_path)


def get_pixel_vertices(ortho: DatasetReader, gdf: GeoDataFrame) -> np.ndarray:
    gdf = gdf.to_crs(ortho.crs)
    polys = gdf.geometry.explode().tolist()
    geo_vertices = [p.exterior.coords for p in polys]

    pxl_vertices = []
    for _, geo_v in enumerate(tqdm(geo_vertices, total=len(geo_vertices))):
        pxl_v = [ortho.index(c[0], c[1]) for c in geo_v[:2]]
        pxl_v = np.asarray(pxl_v)
        # swap axes (y, x) -> (x, y)
        pxl_v[:, [1, 0]] = pxl_v[:, [0, 1]]
        pxl_vertices.append(pxl_v[:4])

    return np.asarray(pxl_vertices)
