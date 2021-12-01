import json
import os
import boto3


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True


def upload_dir(local_directory, bucket, destination):
    client = boto3.client('s3')

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in files:

            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)

            try:
                client.head_object(Bucket=bucket, Key=s3_path)
            except ValueError:
                client.upload_file(local_path, bucket, s3_path)


def download_s3_file(bucket_name, prefix, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, prefix, local_path)


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
