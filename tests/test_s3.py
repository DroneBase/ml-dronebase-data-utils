from ml_dronebase_data_utils.s3 import list_prefix


def test_imports():
    from ml_dronebase_data_utils.s3 import download_dir  # noqa: F401
    from ml_dronebase_data_utils.s3 import download_file  # noqa: F401
    from ml_dronebase_data_utils.s3 import is_json  # noqa: F401
    from ml_dronebase_data_utils.s3 import sync_dir  # noqa: F401
    from ml_dronebase_data_utils.s3 import upload_dir  # noqa: F401
    from ml_dronebase_data_utils.s3 import upload_file  # noqa: F401


def test_list_prefix_files():
    data_file = "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/images/NJ090032 RGB.tiff"
    data_url = "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/images/"
    data_files = list_prefix(data_url, filter_files=True, filter_prefixes=False)
    assert data_files[0] == data_file


def test_list_prefix_prefixes():
    data_prefix = (
        "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/images/"
    )
    data_url = "s3://ml-detectron-test-dataset/data/solar-panel-dataset/predict/"
    data_prefixes = list_prefix(data_url, filter_files=False, filter_prefixes=True)
    assert data_prefixes[0] == data_prefix
