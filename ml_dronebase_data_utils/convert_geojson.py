import os

import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from rasterio.io import DatasetReader
from tqdm import tqdm
from typing import Optional, List, Dict

from .box_utils import vertices_to_boxes, vertices_to_rotated_boxes
from .pascal_voc import PascalVOCWriter
from .s3 import upload_file


def geo_to_voc(ortho_path: str, geo_path: str, save_path: str, class_attribute: Optional[str] = None, class_mapping: Optional[Dict[int,str]] = None,default_class:str = 'panel',skip_classes: List[int] =[], rotated: bool = False):
    '''
    Convert data on geojson format to pascal voc data.

    :param ortho_path: Path to the ortho. Can be a local/s3 location
    :param geo_path: Path to the geojson. Can be a local/s3 location
    :param save_path: Path where the xml file would be saved. Can be a local/s3 location.
    :param class_attribute: The geojson attribute to be used as the class, e.g. id represents the defect id for current panel
    :param class_mapping: The class mapping to use. This would map the value of class_attribute to some class
    :param default_class: The default class to use. Useful when only a single class is being used. Defaults to panel for backwards compatibility.
    :param skip_classes: The classes to be skipped while the conversion. This should be values from the class_attribute field in the geojson.
    :param rotated: Specify if to use rotated bounding boxes, defaults to false.
    '''
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    writer = PascalVOCWriter(ortho_path, ortho.width, ortho.height)

    vertices = get_pixel_vertices(ortho, gdf)
    if rotated:
        boxes = vertices_to_rotated_boxes(vertices)
    else:
        boxes = vertices_to_boxes(vertices)

    if class_attribute is not None:
        names = gdf[class_attribute]
    else:
        names = [default_class]*len(boxes)

    for box, name in zip(boxes, names):
        # Dumb Logic
        try:
            # int(name) might be too restrictive in some scenarios, adapt if required
            name = int(name)
        except ValueError:
            pass
        if name in skip_classes:
            continue
        if class_mapping is not None:
            # If mapping is found, use the default name instead of default.
            name = class_mapping.get(name,name)
        if rotated:
            xmin, ymin, xmax, ymax, angle = box
            writer.addObject(name, xmin, ymin, xmax, ymax, angle)
        else:
            xmin, ymin, xmax, ymax = box
            writer.addObject(name, xmin, ymin, xmax, ymax)

    if "s3://" in save_path:
        writer.save("annot.xml")
        upload_file("annot.xml", save_path, exist_ok=False)
        os.remove("annot.xml")
    else:
        writer.save(save_path)


def geo_to_rotated_voc(ortho_path: str, geo_path: str, save_path: str, class_attribute: Optional[str] = None, class_mapping: Optional[Dict[int,str]] = None,default_class:str = 'panel',skip_classes: List[int] =[]):
    # Migrated rotated logic to geo_to_voc and only keeping it for compatibility
    geo_to_voc(ortho_path,geo_path,save_path,class_attribute,class_mapping,default_class,skip_classes,rotated=True)


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
