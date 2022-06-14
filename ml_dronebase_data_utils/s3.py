import json
import os
import pathlib
import warnings
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def is_json(myjson: str) -> bool:
    """Checks if the string is a json file.

    Args:
        myjson (str): Filename or path to potential json file.

    Returns:
        bool: Whether myjson was a json file.
    """
    try:
        json.loads(myjson)
    except ValueError:
        return False

    return True


def list_prefix(
    s3_url: str,
    filter_files: Optional[bool] = False,
    filter_prefixes: Optional[bool] = False,
) -> List[str]:
    """List urls to all files and prefixes within the path of the given url.

    Args:
        s3_url (str): The s3 url to list from.
        filter_files (Optional[bool]): If true, output will only contain urls
        to the files within the given url. Defaults to False.
        filter_prefixes (Optional[bool]): If true, output will only contain urls
        to the prefixes within the given url. Defaults to False.

    Returns:
        List[str]: The files and/or prefixes within the given url.
    """
    s3 = boto3.resource("s3")
    bucket_name, prefix = _parse_url(s3_url)
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=prefix)
    files = [
        os.path.join("s3://", bucket_name, obj.key)
        for obj in objects
        if obj.key != prefix
    ]

    assert not (filter_files and filter_prefixes), "Can't filter files and prefixes"

    if filter_files:
        files = [file for file in files if file[-1] != "/"]
    elif filter_prefixes:
        files = [file for file in files if file[-1] == "/"]
    return files


def upload_file(local_path: str, s3_url: str, exist_ok: bool = True):
    """Upload file to s3 bucket.

    Args:
        local_path (str): Path to the file to upload.
        s3_url (str): s3 url to upload file to.
        exist_ok (bool, optional): Decides whether or not to ignore existing file. Defaults to True.
    """
    bucket_name, prefix = _parse_url(s3_url)
    client = boto3.client("s3")

    filename = os.path.basename(local_path)

    if not pathlib.Path(prefix).suffix:
        new_prefix = os.path.join(prefix, filename)
    else:
        if pathlib.Path(prefix).suffix == pathlib.Path(filename).suffix:
            new_prefix = prefix
        else:
            new_prefix = os.path.join(os.path.dirname(prefix), filename)
            warnings.warn(
                "Mismatched file extensions, converting prefix to local file format."
            )

    if exist_ok:
        try:
            client.head_object(Bucket=bucket_name, Key=new_prefix)
        except ClientError:
            client.upload_file(local_path, bucket_name, new_prefix)
    else:
        client.upload_file(local_path, bucket_name, new_prefix)


def upload_dir(local_path: str, s3_url: str, exist_ok: bool = True):
    """Upload data from a local directory to an S3 bucket.

    Args:
        local_path (str): Local directory to upload from.
        s3_url (str): S3 url to upload files in directory to.
        exist_ok (bool): Decides whether or not to ignore existing files. Default True.
    """
    bucket_name, prefix = _parse_url(s3_url)
    client = boto3.client("s3")

    files = os.listdir(local_path)
    num_files = len(files)
    for _, filename in enumerate(tqdm(files, total=num_files)):
        file_path = os.path.join(local_path, filename)
        s3_path = os.path.join(prefix, filename)

        if exist_ok:
            try:
                client.head_object(Bucket=bucket_name, Key=s3_path)
            except ClientError:
                client.upload_file(file_path, bucket_name, s3_path)
        else:
            client.upload_file(file_path, bucket_name, s3_path)


def download_file(s3_url: str, local_path: str, size_limit: Optional[int] = None):
    """Download file from S3 bucket to local directory.

    Args:
        s3_url (str): S3 url to the file to download.
        local_path (str): Local directory to store file.
        size_limit (int, optional): Limits the file size accepted to size_limit bytes.
        Default None.
    """
    bucket_name, prefix = _parse_url(s3_url)
    s3 = boto3.client("s3")
    if size_limit is not None:
        response = s3.head_object(Bucket=bucket_name, Key=prefix)
        file_size = int(response["ContentLength"])
        if file_size > size_limit:
            raise ValueError(
                "image size {} exceeds size_limit {}".format(file_size, size_limit)
            )

    s3.download_file(bucket_name, prefix, local_path)


def download_dir(
    s3_url: str, local_path: Optional[str] = None, size_limit: Optional[int] = None
):
    """Download the contents of a folder directory.

    Args:
        s3_url (str): S3 url to bucket directory to download.
        local_path (str, optional): Local directory to store files in.
        size_limit (int, optional): Limits the file size accepted to size_limit bytes.
        Default None.
    """
    bucket_name, prefix = _parse_url(s3_url)
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=prefix)
    num_objects = sum(1 for _ in objects.all())
    for _, obj in enumerate(tqdm(objects, total=num_objects)):
        target = (
            obj.key
            if local_path is None
            else os.path.join(local_path, os.path.relpath(obj.key, prefix))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        if size_limit is not None:
            if obj.size > size_limit:
                continue
        bucket.download_file(obj.key, target)


def sync_dir(from_dir: str, to_dir: str) -> None:
    """Download the contents of a directory in parallel using aws s3 sync.

    Args:
        from_dir (str): S3 url or local path that is the master dir.
        to_dir (str): S3 url or local path that will be synced to from_dir (master dir).
    """
    if "s3://" not in to_dir:
        os.makedirs(to_dir)
    os.system("aws s3 sync {} {}".format(from_dir, to_dir))


def split_dataset(
    data_url: str,
    train_split: Optional[float] = 0.8,
    labels_url: Optional[str] = None,
    val_split: Optional[float] = None,
):
    """train_test_split for files hosted in s3

    Args:
        data_url (str): s3 url location of data to be split
        train_split (Optional[float], optional): percentage of the dataset reserved for training. Defaults to 0.8.
        labels_url (Optional[str], optional): s3 url location of data labels to be split. Defaults to None.
        val_split (Optional[float], optional): percentage of remaining dataset split into val and test (e.g., if
        train_split = 0.6, val_split = 0.5, the splits will be 60% train, 20% val, and 20% test). Defaults to None.
    """
    if labels_url is not None:
        _split_labeled_dataset(data_url, labels_url, train_split, val_split)
    else:
        _split_unlabeled_dataset(data_url, train_split, val_split)


def move_files(bucket: str, prefix: str, files: List[str]):
    """Move files from within s3.

    Args:
        bucket (str): bucket name from within which to move files.
        prefix (str): prefix to move the files to.
        files (List[str]): list of files being moved.
    """
    s3 = boto3.resource("s3")
    for file in files:
        copy_source = {"Bucket": bucket, "Key": file}
        new_prefix = os.path.join(prefix, os.path.basename(file))
        s3.meta.client.copy(copy_source, bucket, new_prefix)


def _parse_url(url: str) -> Tuple[str, str]:
    url_parsed = urlparse(url, allow_fragments=False)
    bucket = url_parsed.netloc
    prefix = url_parsed.path.lstrip("/")
    return bucket, prefix


def _make_split_prefix(prefix: str, split: str) -> str:
    if prefix[-1] == "/":
        prefix = prefix[:-1]
    child_prefix = os.path.basename(prefix)
    parent_prefix = os.path.dirname(prefix)
    split_prefix = os.path.join(parent_prefix, split, child_prefix)
    return split_prefix


def delete_missing_pairs(input_data, pair_data):
    """Delete items from input data that does not have a corresponding item in pair data
       This assumes that the filenames without extensions to be the same in both input lists

    Args:
        input_data (list): list of file paths to check
        pair_data (list): list of file paths to find a match

    Returns:
        list: list of files with missing pairs deleted
    """
    to_del = []
    for idx, pth in enumerate(input_data):
        name = pathlib.Path(pth).stem
        combined = "\t".join(pair_data)
        if name not in combined:
            to_del.append(idx)
    for index in sorted(to_del, reverse=True):
        del input_data[index]
    return input_data


def _split_labeled_dataset(
    data_url: str,
    labels_url: str,
    train_split: int = 0.8,
    val_split: Optional[float] = None,
    delete_missing_pairs: Optional[
        bool
    ] = False,  # Assumes filenames without extension to be the same
):
    data_bucket_name, data_prefix = _parse_url(data_url)
    labels_bucket_name, labels_prefix = _parse_url(labels_url)

    s3 = boto3.resource("s3")
    data_bucket = s3.Bucket(data_bucket_name)
    labels_bucket = s3.Bucket(labels_bucket_name)
    data = [
        x.key
        for x in data_bucket.objects.filter(Prefix=data_prefix)
        if x.key[-1] != "/"
    ]
    labels = [
        x.key
        for x in labels_bucket.objects.filter(Prefix=labels_prefix)
        if x.key[-1] != "/"
    ]

    if delete_missing_pairs:
        # Delete images without labels
        data = delete_missing_pairs(data, labels)

        # Delete labels without images
        labels = delete_missing_pairs(labels, data)

    x_train, x_val, y_train, y_val = train_test_split(
        data, labels, train_size=train_split
    )
    split_prefix = _make_split_prefix(data_prefix, "train")
    move_files(data_bucket_name, split_prefix, x_train)
    split_prefix = _make_split_prefix(labels_prefix, "train")
    move_files(labels_bucket_name, split_prefix, y_train)

    if val_split is not None:
        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, train_size=val_split
        )
        split_prefix = _make_split_prefix(data_prefix, "val")
        move_files(data_bucket_name, split_prefix, x_val)
        split_prefix = _make_split_prefix(labels_prefix, "val")
        move_files(labels_bucket_name, split_prefix, y_val)

        split_prefix = _make_split_prefix(data_prefix, "test")
        move_files(data_bucket_name, split_prefix, x_test)
        split_prefix = _make_split_prefix(labels_prefix, "test")
        move_files(labels_bucket_name, split_prefix, y_test)
    else:
        split_prefix = _make_split_prefix(data_prefix, "val")
        move_files(data_bucket_name, split_prefix, x_val)
        split_prefix = _make_split_prefix(labels_prefix, "val")
        move_files(labels_bucket_name, split_prefix, y_val)


def _split_unlabeled_dataset(
    data_url: str, train_split: int = 0.8, val_split: Optional[float] = None
):
    data_bucket_name, data_prefix = _parse_url(data_url)

    s3 = boto3.resource("s3")
    data_bucket = s3.Bucket(data_bucket_name)
    data = [
        x.key
        for x in data_bucket.objects.filter(Prefix=data_prefix)
        if x.key[-1] != "/"
    ]

    x_train, x_val = train_test_split(data, train_size=train_split)
    split_prefix = _make_split_prefix(data_prefix, "train")
    move_files(data_bucket_name, split_prefix, x_train)

    if val_split is not None:
        x_val, x_test = train_test_split(x_val, train_size=val_split)
        split_prefix = _make_split_prefix(data_prefix, "val")
        move_files(data_bucket_name, split_prefix, x_val)

        split_prefix = _make_split_prefix(data_prefix, "test")
        move_files(data_bucket_name, split_prefix, x_test)
    else:
        split_prefix = _make_split_prefix(data_prefix, "val")
        move_files(data_bucket_name, split_prefix, x_val)
