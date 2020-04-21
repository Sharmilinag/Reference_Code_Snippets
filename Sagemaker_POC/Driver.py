import Utils as utils
import Constants as cons
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import sys


def main():
    role = cons.ROLE
    print(role)

    #   ##  read the data
    data = pd.read_csv('s3://ada-engine/MODEL_SELECTION/DATASETS/bank_clean.csv', index_col=0)
    print(data.shape)
    # print(data.columns.values)
    y = data['y_yes']
    X = data.drop(['y_no', 'y_yes'], axis=1)
    print(X.shape, y.shape)

    #   ##  train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # sys.exit(0)
    #   ##  Prepare s3 inputs for sage-maker
    # Input Mode (Pipe or File)
    input_mode = "File"
    s3_train_path = 's3://{}/{}/train'.format(cons.S3_BUCKET, cons.DATA_PATH)
    print(s3_train_path)
    s3_validation_path = 's3://{}/{}/validation'.format(cons.S3_BUCKET, cons.DATA_PATH)
    s3_input_train = sagemaker.s3_input(s3_data=s3_train_path, content_type='csv')
    s3_input_validation = sagemaker.s3_input(s3_data=s3_validation_path, content_type='csv')

    #   ##  sage-maker configs
    sess = sagemaker.Session(boto_session=cons.BOTO_SESS, default_bucket=cons.S3_BUCKET)
    xgb = sagemaker.estimator.Estimator(cons.SM_XG_CONTAINERS[cons.REGION], role, train_instance_count=1,
                                        train_instance_type='ml.m5.large',
                                        output_path='s3://{}/{}/output'.format(cons.S3_BUCKET, cons.MODEL_PATH),
                                        sagemaker_session=sess, input_mode=input_mode)
    xgb.set_hyperparameters(
        eval_metric='auc',
        objective='binary:logistic',
        num_round=100,
        rate_drop=0.3,
        tweedie_variance_power=1.4
    )
    objective_metric_name = 'validation:auc'

    # #   ## Train the XgBoost Model (NO HYPER PARAMETER TUNING)
    xgb.fit({'train': s3_input_train})

    #   ##  HYPER PARAMETER TUNING
    # #   ##  set ranges for hyper parameters
    # hyperparameter_ranges = {
    #     'alpha': ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
    #     'lambda': ContinuousParameter(0.01, 10, scaling_type="Logarithmic")
    # }
    #
    # #   ##  hyper-parameter tuning jobs
    # tuner_log = HyperparameterTuner(
    #     xgb,
    #     objective_metric_name,
    #     hyperparameter_ranges,
    #     max_jobs=20,
    #     max_parallel_jobs=10,
    #     strategy='Random'
    # )
    #
    # tuner_log.fit({'train': s3_input_train, 'validation': s3_input_validation}, include_cls_metadata=False)


def check_best_hyper_parameter(job_name):
    #   ##  Analyze tuning job results - after tuning job is completed check jobs have finished
    status_log = cons.BOTO_SM.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=job_name)['HyperParameterTuningJobStatus']
    # HyperParameterTuningJobName=tuner_log.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
    assert status_log == 'Completed', "First must be completed, was {}".format(status_log)
    # df_log = sagemaker.HyperparameterTuningJobAnalytics(tuner_log.latest_tuning_job.job_name).dataframe()
    df_log = sagemaker.HyperparameterTuningJobAnalytics(job_name).dataframe()
    print(df_log)
    df_log.to_csv('hyp.csv')


if __name__ == "__main__":
    main()
    # hp_job_name = 'xgboost-200328-0130'
    # check_best_hyper_parameter(hp_job_name)
