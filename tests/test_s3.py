import glob
import os

from ml_dronebase_data_utils.s3 import list_prefix, sync_dir


# def test_imports():
#     from ml_dronebase_data_utils.s3 import download_dir  # noqa: F401
#     from ml_dronebase_data_utils.s3 import download_file  # noqa: F401
#     from ml_dronebase_data_utils.s3 import is_json  # noqa: F401
#     from ml_dronebase_data_utils.s3 import sync_dir  # noqa: F401
#     from ml_dronebase_data_utils.s3 import upload_dir  # noqa: F401
#     from ml_dronebase_data_utils.s3 import upload_file  # noqa: F401


# def test_list_prefix_files():
#     data_file = "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/images/NJ090032 RGB.tiff"
#     data_url = "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/images/"
#     data_files = list_prefix(data_url, filter_files=True, filter_prefixes=False)
#     assert data_files[0] == data_file


def test_list_prefix_prefixes():
    data_prefix = (
        "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/images/"
    )
    data_url = "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/"
    data_prefixes = list_prefix(data_url, filter_files=False, filter_prefixes=True)
    print(data_prefixes)
    # assert data_prefixes[0] == data_prefix

    data_url = "s3://ml-commercial-anomaly/data/all-labeled-data/val/"
    data_prefixes = list_prefix(data_url, filter_files=False, filter_prefixes=True)
    print(data_prefixes)


# def test_sync():
#     data_url = (
#         "s3://ml-detectron-test-dataset/data/solar-panel-dataset/train/annotations/"
#     )
#     data_dir = "solar-panel-dataset-v2/train/annotations/"
#     sync_dir(from_dir=data_url, to_dir=data_dir)
#     data_files = glob.glob(os.path.join(data_dir, "*"))
#     assert len(data_files) > 0
