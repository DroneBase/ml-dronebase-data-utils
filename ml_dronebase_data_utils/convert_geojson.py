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
    """Convert a geojson file containing a set of vertices in geographical coordinates
    to an xml file containing VOC style bounding boxes in image coordinates.

    Args:
        ortho_path (str): The path/url to the orthomosaic used to map geographic coordinates
            to image coordinates. It must have a valid coordinate reference system (CRS) to
            support coordinate mapping.
        geo_path (str): The path/url to the geojson file to be converted to. The `geometry` field
            in the GeoDataFrame is assumed to be Multipolygons.
        save_path (str): The path/url to save the converted xml file.
    """
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
    """Convert a geojson file containing a set of vertices in geographical coordinates
    to an xml file containing oriented VOC style bounding boxes in image coordinates.

    Args:
        ortho_path (str): The path/url to the orthomosaic used to map geographic coordinates
            to image coordinates. It must have a valid coordinate reference system (CRS) to
            support coordinate mapping.
        geo_path (str): The path/url to the geojson file to be converted to. The `geometry` field
            in the GeoDataFrame is assumed to contain Multipolygons.
        save_path (str): The path/url to save the converted xml file.
    """
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
    """Convert the set of geographical vertices to image vertices.

    Args:
        ortho (DatasetReader): The orthomosaic file used to index geographical coordinates
            to image coordinates.
        gdf (GeoDataFrame): The dataframe containing the set of geographical vertices.
            The `geometry` field is assumed to contain Multipolygons.

    Returns:
        np.ndarray: A matrix of vertices in image coordinates with shape Nx4x2.
    """
    gdf = gdf.to_crs(ortho.crs)
    polys = gdf.geometry.explode().tolist()
    geo_vertices = [p.exterior.coords for p in polys]

    import logging

    pxl_vertices = []
    for geo_v in tqdm(
        geo_vertices, total=len(geo_vertices), desc="Extracting Pixel Vertices"
    ):
        geo_v_ = [[x, y] for x, y in geo_v]
        geo_v_ = np.asarray(geo_v_)
        geo_v_[:, [1, 0]] = geo_v_[:, [0, 1]]
        logging.info(f"get_pixel_vertices:: geo vertices = {geo_v_}")
        pxl_v = [ortho.index(x, y) for x, y in geo_v]
        pxl_v = np.asarray(pxl_v)
        # swap axes (y, x) -> (x, y)
        pxl_v[:, [1, 0]] = pxl_v[:, [0, 1]]
        logging.info(f"get_pixel_vertices:: pixel vertices = {pxl_v}")
        pxl_vertices.append(pxl_v[:4])

    return np.asarray(pxl_vertices)
