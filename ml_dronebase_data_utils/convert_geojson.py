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
    gdf = gdf.to_crs(ortho.crs)
    polys = gdf.geometry.explode().tolist()
    geo_vertices = [p.exterior.coords for p in polys]

    pxl_vertices = []
    for geo_v in tqdm(geo_vertices, total=len(geo_vertices)):
        pxl_v = [ortho.index(c[0], c[1]) for c in geo_v]
        pxl_v = np.asarray(pxl_v)
        # swap axes (y, x) -> (x, y)
        pxl_v[:, [1, 0]] = pxl_v[:, [0, 1]]
        pxl_vertices.append(pxl_v[:4])

    return np.asarray(pxl_vertices)

def run_geojson_conversion(**kwargs):
    '''
    Convert geojson format to pascal voc format

    Keyword Arguments

    ortho_path -> The ortho path, can be local/s3
    geojson -> The geojson path
    save_path -> The save path
    class_attribute -> The class attribute to use from the geojson for class labels
    class_mapping -> A plain txt file containing class mappings
    skip_classes -> Classes to skip, specify multiple
    rotated -> Use rotated bounding box, defaults to false

    '''
    import tempfile
    from pathlib import Path

    ortho_path = kwargs.get('ortho_path',None)
    geojson = kwargs.get('geojson',None)
    save_path = kwargs.get('save_path',None)

    if ortho_path is None or geojson is None or save_path is None:
        print("You must specify ortho_path, anno_path and save_path")
        return 1
    
    class_attribute = kwargs.get('class_attribute',None)
    class_mapping = kwargs.get('class_mapping',None)
    default_class = kwargs.get('default_class','panel')
    skip_classes = kwargs.get('skip_classes',[])
    rotated = kwargs.get('rotated',False)

    #Call the function
    geo_to_voc(
        ortho_path,
        geojson,
        save_path,
        class_attribute,
        class_mapping,
        default_class,
        skip_classes,
        rotated
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert geojson to voc format data")

    parser.add_argument('--ortho-path',required=True,help="The ortho path, can be local/s3")
    parser.add_argument('--geojson',required=True,help="The geojson path")
    parser.add_argument('--save-path',required=True,help="The save path")
    parser.add_argument('--class-attribute',type=str,help="The class attribute to use from the geojson for class labels")
    parser.add_argument('--class-mapping',help="A plain txt file containing class mappings")
    parser.add_argument('--skip-classes',type=int,nargs='+',help="Classes to skip, specify multiple")
    parser.add_argument('--rotated',action='store_true',default=False,help="Use rotated bounding box, defaults to false")

    args = vars(parser.parse_args())

    # Read class mapping if provided
    class_mapping = args.get('class_mapping',None)
    if class_mapping is not None:
        mapping = {}
        with open('class_mapping') as cm:
            for line in cm:
                key,value = line.split(':',maxsplit=2)
                key = key.strip()
                value = value.strip('\n').strip()
                mapping[key] = value
        args['class_mapping'] = mapping

    run_geojson_conversion(args)
