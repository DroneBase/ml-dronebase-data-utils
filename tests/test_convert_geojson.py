import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon, box

from ml_dronebase_data_utils.box_utils import (
    boxes_to_vertices,
    rotated_box_dims,
    rotated_boxes_to_vertices,
    sort_points,
    vertices_to_boxes,
    vertices_to_rotated_boxes,
)
from ml_dronebase_data_utils.convert_geojson import get_pixel_vertices


@pytest.fixture
def ortho_path():
    # return "/Users/conorwallace/Downloads/CA060026_20220317.tiff"
    return "s3://db-insights-solar-data/insights/2021/SMA Longroad/LR_NJ090032_20210420/NJ090032 RGB.tiff"


@pytest.fixture
def geo_path():
    # return "/Users/conorwallace/Downloads/CA060026_20220317.geojson"
    return "s3://db-insights-solar-data/insights/2021/SMA Longroad/LR_NJ090032_20210420/Panels NJ090032.geojson"


@pytest.fixture
def points_a() -> np.ndarray:
    """Returns a set of vertices of a box rotated 90 degrees CCW from the y-axis.

    Returns:
        np.ndarray: A 4x2 matrix of vertices that have been shuffled to test handling an arbitrary set of points.
    """
    points = np.array([[70, 84], [130, 84], [130, 114], [70, 114]])
    np.random.shuffle(points)
    return points


@pytest.fixture
def points_b() -> np.ndarray:
    """Returns a set of vertices of a box rotated 60 degrees CCW from the y-axis.

    Returns:
        np.ndarray: A 4x2 matrix of vertices that have been shuffled to test handling an arbitrary set of points.
    """
    points = np.array([[81, 71], [133, 101], [118, 127], [66, 97]])
    np.random.shuffle(points)
    return points


@pytest.fixture
def points_c() -> np.ndarray:
    """Returns a set of vertices of a box rotated 27 degrees CCW from the y-axis.
    This set of vertices tests the edge case where the two center points have the same x coordinate,
    thus transitioning to the 2nd height vector variation.

    Returns:
        np.ndarray: A 4x2 matrix of vertices that have been shuffled to test handling an arbitrary set of points.
    """
    points = np.array([[72, 79], [99, 66], [126, 119], [99, 132]])
    np.random.shuffle(points)
    return points


@pytest.fixture
def points_d() -> np.ndarray:
    """Returns a set of vertices of a box rotated 0 degrees CCW from the y-axis.

    Returns:
        np.ndarray: A 4x2 matrix of vertices that have been shuffled to test handling an arbitrary set of points.
    """
    points = np.array([[84, 69], [114, 69], [114, 129], [84, 129]])
    np.random.shuffle(points)
    return points


@pytest.fixture
def points_e() -> np.ndarray:
    """Returns a set of vertices of a box rotated -27 degrees CCW from the y-axis.
    This set of vertices tests the edge case where the two center points have the same x coordinate,
    thus transitioning to the 3rd height vector variation.

    Returns:
        np.ndarray: A 4x2 matrix of vertices that have been shuffled to test handling an arbitrary set of points.
    """
    points = np.array([[99, 66], [126, 79], [99, 132], [72, 119]])
    np.random.shuffle(points)
    return points


@pytest.fixture
def points_f() -> np.ndarray:
    """Returns a set of vertices of a box rotated -60 degrees CCW from the y-axis.

    Returns:
        np.ndarray: A 4x2 matrix of vertices that have been shuffled to test handling an arbitrary set of points.
    """
    points = np.array([[66, 101], [118, 71], [132, 97], [81, 127]])
    np.random.shuffle(points)
    return points


def test_sort_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
    points_c: np.ndarray,
    points_d: np.ndarray,
    points_e: np.ndarray,
    points_f: np.ndarray,
):
    points_a_sorted = sort_points(points_a)
    target_points_a = np.array([[70, 84], [130, 84], [130, 114], [70, 114]])
    assert np.all(
        points_a_sorted == target_points_a
    ), "Assertion test a) failed.\
        Sorted points don't match the expected result."

    tl_a, tr_a, bl_a = points_a_sorted[0], points_a_sorted[1], points_a_sorted[3]
    angle_a, _, _ = rotated_box_dims(tl_a, tr_a, bl_a)

    assert (
        round(angle_a) == 90
    ), f"Assertion test a) failed.\
        Rotation angle returned {round(angle_a)} degrees, expected 90 degrees."

    points_b_sorted = sort_points(points_b)
    target_points_b = np.array([[81, 71], [133, 101], [118, 127], [66, 97]])
    assert np.all(
        points_b_sorted == target_points_b
    ), "Assertion test b) failed.\
        Sorted points don't match the expected result."

    tl_b, tr_b, bl_b = points_b_sorted[0], points_b_sorted[1], points_b_sorted[3]
    angle_b, _, _ = rotated_box_dims(tl_b, tr_b, bl_b)

    assert (
        round(angle_b) == 60
    ), f"Assertion test b) failed.\
        Rotation angle returned {round(angle_b)} degrees, expected 60 degrees."

    points_c_sorted = sort_points(points_c)
    target_points_c = np.array([[72, 79], [99, 66], [126, 119], [99, 132]])
    assert np.all(
        points_c_sorted == target_points_c
    ), "Assertion test c) failed.\
        Sorted points don't match the expected result."

    tl_c, tr_c, bl_c = points_c_sorted[0], points_c_sorted[1], points_c_sorted[3]
    angle_c, _, _ = rotated_box_dims(tl_c, tr_c, bl_c)

    assert (
        round(angle_c) == 27
    ), f"Assertion test c) failed.\
        Rotation angle returned {round(angle_c)} degrees, expected 27 degrees."

    points_d_sorted = sort_points(points_d)
    target_points_d = np.array([[84, 69], [114, 69], [114, 129], [84, 129]])
    assert np.all(
        points_d_sorted == target_points_d
    ), "Assertion test d) failed.\
        Sorted points don't match the expected result."

    tl_d, tr_d, bl_d = points_d_sorted[0], points_d_sorted[1], points_d_sorted[3]
    angle_d, _, _ = rotated_box_dims(tl_d, tr_d, bl_d)

    assert (
        round(angle_d) == 0
    ), f"Assertion test d) failed.\
        Rotation angle returned {round(angle_d)} degrees, expected 0 degrees."

    points_e_sorted = sort_points(points_e)
    target_points_e = np.array([[72, 119], [99, 66], [126, 79], [99, 132]])
    assert np.all(
        points_e_sorted == target_points_e
    ), "Assertion test e) failed.\
        Sorted points don't match the expected result."

    tl_e, tr_e, bl_e = points_e_sorted[0], points_e_sorted[1], points_e_sorted[3]
    angle_e, _, _ = rotated_box_dims(tl_e, tr_e, bl_e)

    assert (
        round(angle_e) == -27
    ), f"Assertion test e) failed.\
        Rotation angle returned {round(angle_e)} degrees, expected -27 degrees."

    points_f_sorted = sort_points(points_f)
    target_points_f = np.array([[66, 101], [118, 71], [132, 97], [81, 127]])
    assert np.all(
        points_f_sorted == target_points_f
    ), "Assertion test f) failed.\
        Sorted points don't match the expected result."

    tl_f, tr_f, bl_f = points_f_sorted[0], points_f_sorted[1], points_f_sorted[3]
    angle_f, _, _ = rotated_box_dims(tl_f, tr_f, bl_f)

    assert (
        round(angle_f) == -60
    ), f"Assertion test f) failed.\
        Rotation angle returned {round(angle_f)} degrees, expected -60 degrees."


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
