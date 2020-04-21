import pandas as pd
import boto3
import json

from botocore.exceptions import ClientError

s3 = boto3.resource('s3',
                    aws_access_key_id='AKIAI3PWAHDDX5GCDDFA',  # ## todo: remove these in prod
                    aws_secret_access_key='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')
s3_client = boto3.client('s3',
                         aws_access_key_id='AKIAI3PWAHDDX5GCDDFA',  # ## todo: remove these in prod
                         aws_secret_access_key='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')


def read_json_data(s3_bucket, json_path):
    obj = s3.Object(s3_bucket, json_path)
    body = obj.get()['Body'].read()
    return json.loads(body)


def upload_file(local_file_name, bucket, target_file=None):
    """Upload a file to an S3 bucket

    :param local_file_name: File to upload
    :param bucket: Bucket to upload to
    :param target_file: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if target_file is None:
        object_name = local_file_name
    # Upload the file
    try:
        response = s3_client.upload_file(local_file_name, bucket, target_file)
    except ClientError as e:
        print(e)
        return False
    return True

# for chunk in pd.read_csv('s3://fis-ada-bucket/MODEL_SELECTION/DATASETS/iris.data', chunksize=100):
#     print(chunk)
# upload_file('requirements.txt', 'ada-engine', 'MODEL_SELECTION/RESULT_REPORTS/requirements.txt')
