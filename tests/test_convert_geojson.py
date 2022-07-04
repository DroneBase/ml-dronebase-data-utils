import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon, box

from ml_dronebase_data_utils.box_utils import (
    boxes_to_vertices,
    rotated_boxes_to_vertices,
    sort_points,
    vertices_to_boxes,
    vertices_to_rotated_boxes,
)
from ml_dronebase_data_utils.convert_geojson import get_pixel_vertices


@pytest.fixture
def ortho_path():
    return "s3://db-insights-solar-data/insights/2021/SMA Longroad/LR_NJ090032_20210420/NJ090032 RGB.tiff"


@pytest.fixture
def geo_path():
    return "s3://db-insights-solar-data/insights/2021/SMA Longroad/LR_NJ090032_20210420/Panels NJ090032.geojson"


def test_get_boxes(ortho_path, geo_path):
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    vertices = get_pixel_vertices(ortho, gdf)
    boxes = vertices_to_boxes(vertices)

    vertices = np.array([sort_points(v) for v in vertices])
    vertices_reconstructed = boxes_to_vertices(boxes)
    vertices_reconstructed = np.array([sort_points(v) for v in vertices_reconstructed])

    polygons = np.array(
        [box(minx=b[0], miny=b[1], maxx=b[2], maxy=b[3]) for b in boxes]
    )
    polygons_reconstructed = np.array([Polygon(v) for v in vertices_reconstructed])

    centroids = np.array([p.centroid.xy for p in polygons]).squeeze(2)
    indices_sorted = np.lexsort((centroids[:, 1], centroids[:, 0]))
    polygons = polygons[indices_sorted]

    centroids_reconstructed = np.array(
        [p.centroid.xy for p in polygons_reconstructed]
    ).squeeze(2)
    indices_sorted = np.lexsort(
        (centroids_reconstructed[:, 1], centroids_reconstructed[:, 0])
    )
    polygons_reconstructed = polygons_reconstructed[indices_sorted]

    ious = np.array(
        [
            p.intersection(p_r).area / p.union(p_r).area
            for p, p_r in zip(polygons, polygons_reconstructed)
        ]
    )
    in_bounds = ious == 1.0
    assert in_bounds.all(), "Reconstructed vertices do not match the original vertices."


def test_get_rotated_boxes(ortho_path, geo_path):
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    vertices = get_pixel_vertices(ortho, gdf)
    boxes = vertices_to_rotated_boxes(vertices)
    angles = boxes[:, -1]
    in_range = (-90 < angles) & (angles <= 90)
    assert in_range.all(), "Computed angle not in range."

    vertices = [sort_points(v) for v in vertices]
    vertices_reconstructed = rotated_boxes_to_vertices(boxes, box_mode="XYXYA_ABS")
    vertices_reconstructed = [sort_points(v) for v in vertices_reconstructed]

    polygons = np.array([Polygon(v) for v in vertices])
    polygons_reconstructed = np.array([Polygon(v) for v in vertices_reconstructed])

    centroids = np.array([p.centroid.xy for p in polygons]).squeeze(2)
    indices_sorted = np.lexsort((centroids[:, 1], centroids[:, 0]))
    polygons = polygons[indices_sorted]

    centroids_reconstructed = np.array(
        [p.centroid.xy for p in polygons_reconstructed]
    ).squeeze(2)
    indices_sorted = np.lexsort(
        (centroids_reconstructed[:, 1], centroids_reconstructed[:, 0])
    )
    polygons_reconstructed = polygons_reconstructed[indices_sorted]

    ious = np.array(
        [
            p.intersection(p_r).area / p.union(p_r).area
            for p, p_r in zip(polygons, polygons_reconstructed)
        ]
    )
    in_bounds = ious > 0.95
    assert (
        in_bounds.all()
    ), "Reconstructed vertices does not match the original vertices."
