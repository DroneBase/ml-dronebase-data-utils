import geopandas as gpd
import numpy as np
import pytest
import rasterio

from ml_dronebase_data_utils.box_utils import (
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


@pytest.fixture
def save_path():
    return "s3://ml-detectron-test-dataset/data/solar-panel-dataset/val/annotations/NJ090032.xml"


@pytest.fixture
def vertices_flat():
    return np.array([[[30, 20], [30, 60], [80, 20], [80, 60]]])


@pytest.fixture
def vertices_straight():
    return np.array([[[100, 20], [100, 80], [120, 20], [120, 80]]])


@pytest.fixture
def vertices_clockwise():
    return np.array([[[10, 90], [30, 70], [50, 130], [70, 110]]])


@pytest.fixture
def vertices_counter_clockwise():
    return np.array([[[80, 160], [110, 100], [130, 140], [100, 200]]])


def test_flat_vertices_to_rotated_boxes(vertices_flat):
    boxes = vertices_to_rotated_boxes(vertices_flat)
    assert boxes[0][-1] == 90.0


def test_straight_vertices_to_rotated_boxes(vertices_straight):
    boxes = vertices_to_rotated_boxes(vertices_straight)
    assert boxes[0][-1] == 0.0


def test_clockwise_vertices_to_rotated_boxes(vertices_clockwise):
    boxes = vertices_to_rotated_boxes(vertices_clockwise)
    assert boxes[0][-1] == -45.0


def test_counter_clockwise_vertices_to_rotated_boxes(vertices_counter_clockwise):
    boxes = vertices_to_rotated_boxes(vertices_counter_clockwise)
    assert np.round(boxes[0][-1], decimals=2) == 63.43


def test_get_boxes(ortho_path, geo_path):
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    vertices = get_pixel_vertices(ortho, gdf)
    boxes = vertices_to_boxes(vertices)
    assert boxes is not None


def test_get_rotated_boxes(ortho_path, geo_path):
    ortho = rasterio.open(ortho_path)
    gdf = gpd.read_file(geo_path)

    vertices = get_pixel_vertices(ortho, gdf)
    boxes = vertices_to_rotated_boxes(vertices)
    assert boxes is not None
