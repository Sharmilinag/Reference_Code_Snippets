import boto3
from datetime import datetime

_aws_id = ''
_aws_secret = ''

BOTO_SESS = boto3.Session(aws_access_key_id=_aws_id,  # ## todo: remove these in prod
                          aws_secret_access_key=_aws_secret)

BOTO_SM = boto3.client('sagemaker',
                       aws_access_key_id=_aws_id,  # ## todo: remove these in prod
                       aws_secret_access_key=_aws_secret
                       )
BOTO_S3 = s3 = boto3.resource('s3',
                              aws_access_key_id=_aws_id,  # ## todo: remove these in prod
                              aws_secret_access_key=_aws_secret)

#   ##  Time-stamps
now = datetime.now()  # current date and time
TIMESTAMP = now.strftime("%Y%m%d-%H%M%S")

#   ##  s3
S3_BUCKET = 'ada-engine'
MODEL_PATH = 'sagemaker/DEMO-xgboost-dm'
DATA_PATH = 'MODEL_SELECTION/DATASETS'
SKLEARN_MODEL_PATH = 'sagemaker/sklearn_models'
S3_TRAIN_PATH = 's3://{}/{}/train'.format(S3_BUCKET, DATA_PATH)
S3_VALIDATION_PATH = 's3://{}/{}/validation'.format(S3_BUCKET, DATA_PATH)

#   ##  sage-maker
ROLE = 'arn:aws:iam::<account-number>:role/<role-name>'
REGION = 'us-east-1'
SM_XG_CONTAINERS = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
                    'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
                    'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
                    'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
