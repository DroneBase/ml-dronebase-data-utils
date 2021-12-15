import json
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError
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


def upload_dir(local_directory: str, bucket_name: str, prefix: str) -> None:
    """Upload data from a local directory to an S3 bucket.

    Args:
        local_directory (str): Local directory to upload from.
        bucket_name (str): S3 bucket name.
        prefix ([type]): Relative path from bucket to save data.
    """
    client = boto3.client("s3")

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(prefix, relative_path)

            try:
                client.head_object(Bucket=bucket_name, Key=s3_path)
            except ClientError:
                client.upload_file(local_path, bucket_name, s3_path)


def download_s3_file(bucket_name: str, prefix: str, local_path: str) -> None:
    """Download file from S3 bucket to local directory.

    Args:
        bucket_name (str): S3 bucket name.
        prefix ([type]): Relative path from bucket to requested file.
        local_path ([type]): Local directory to store file.
    """
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, prefix, local_path)


def download_s3_folder(
    bucket_name: str, prefix: str, local_directory: Optional[str] = None
) -> None:
    """Download the contents of a folder directory.

    Args:
        bucket_name (str): S3 bucket name.
        prefix (str): Relative path from bucket to requested files.
        local_directory (str, optional): Local directory to store files in.
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=prefix)
    num_objects = sum(1 for _ in objects.all())
    for i, obj in enumerate(tqdm(objects, total=num_objects)):
        target = (
            obj.key
            if local_directory is None
            else os.path.join(local_directory, os.path.relpath(obj.key, prefix))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)
