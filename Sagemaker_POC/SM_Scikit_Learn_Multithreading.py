import sagemaker
import Constants as cons
from sagemaker.sklearn.estimator import SKLearn
from Scikit_Learn_Script_Dependencies.Scikit_Learn_Hyp_Tuning_Constants import CLASSIFIER_ALGORITHMS
import time
from multiprocessing import Pool

#   ##  Concurrent Thread Count
THREAD_CNT = 5


def single_process_main():
    start = time.time()
    ###############################################################################################

    #   ##  Prepare the data in the format expected by the "Scikit_Learn_Script.py"
    #   ##  This Data is prepared and available at "s3://ada-engine/MODEL_SELECTION/DATASETS/train/train.csv"

    # S3 prefix
    # prefix = 'sagemaker'

    # Get a SageMaker-compatible role used by this Driver Instance.
    role = cons.ROLE
    print(role)

    # SageMaker configs
    sess = sagemaker.Session(boto_session=cons.BOTO_SESS, default_bucket=cons.S3_BUCKET)
    s3_train_path = 's3://{}/{}/train'.format(cons.S3_BUCKET, cons.DATA_PATH)
    s3_model_path = 's3://{}/{}/{}'.format(cons.S3_BUCKET, 'sagemaker', 'sklearn_models')

    for algorithm in CLASSIFIER_ALGORITHMS.keys():
        #   ##  SageMaker Scikit_Learn object
        sklearn = SKLearn(
            entry_point='Scikit_Learn_Script_Hyp_Tuning.py',
            train_instance_type="ml.m5.large",
            source_dir='Scikit_Learn_Script_Dependencies',
            role=role,
            sagemaker_session=sess,
            hyperparameters={'max_leaf_nodes': 30, 'algorithm': algorithm}
            # , 'validation_data': cons.S3_VALIDATION_PATH}
        )

        #   ##  Set Job Name (over-riding the default job name)
        sm_job_name = 'sklearn-{0}-{1}'.format(algorithm.replace('_', '-'), cons.TIMESTAMP)

        # Train the Model
        sklearn.fit({'train': cons.S3_TRAIN_PATH, 'test': cons.S3_VALIDATION_PATH}, job_name=sm_job_name)

        #   ##  Move the model to appropriate model path
        # copy_source = {'Bucket': cons.S3_BUCKET, 'Key': sm_job_name + '/output/model.tar.gz'}
        # model_key = '{}/{}/output/model.tar.gz'.format(cons.SKLEARN_MODEL_PATH, sm_job_name)
        # cons.BOTO_S3.meta.client.copy(copy_source, cons.S3_BUCKET, model_key)
        #   ##  Delete job run artifacts generated from S3
        # bucket = cons.BOTO_S3.Bucket(cons.S3_BUCKET)
        # for obj in bucket.objects.filter(Prefix='{}/'.format(sm_job_name)):
        #     cons.BOTO_S3.Object(bucket.name, obj.key).delete()

    ###############################################################################################################
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\n\nTIME ELAPSED ::: {:0>2} hrs, {:0>2} mins, {:05.2f} secs".format(int(hours), int(minutes), seconds))


def spawn_sagemaker_job(algorithm):
    #   ##  SageMaker session
    SM_SESS = sagemaker.Session(boto_session=cons.BOTO_SESS, default_bucket=cons.S3_BUCKET)
    print('===========================', algorithm)
    #   ##  SageMaker Scikit_Learn object
    sklearn = SKLearn(
        entry_point='Scikit_Learn_Script_Hyp_Tuning.py',
        train_instance_type="ml.m5.large",
        source_dir='Scikit_Learn_Script_Dependencies',
        role=cons.ROLE,
        sagemaker_session=SM_SESS,
        hyperparameters={'max_leaf_nodes': 30, 'algorithm': algorithm}
    )
    #   ##  Set Job Name (over-riding the default job name)
    sm_job_name = 'sklearn-{0}-{1}'.format(algorithm.replace('_', '-'), cons.TIMESTAMP)
    # Train the Model
    sklearn.fit({'train': cons.S3_TRAIN_PATH, 'test': cons.S3_VALIDATION_PATH}, job_name=sm_job_name)
    return sm_job_name


def multi_process_main():
    #   ##  Prepare the data in the format expected by the "Scikit_Learn_Script.py"
    #   ##  This Data is prepared and available at "s3://ada-engine/MODEL_SELECTION/DATASETS/train/train.csv"

    # S3 prefix
    # prefix = 'sagemaker'

    # Get a SageMaker-compatible role used by this Driver Instance.
    role = cons.ROLE
    print(role)

    #   ##  Algorithm choices
    algos = CLASSIFIER_ALGORITHMS.keys()

    #   ##  multi-pooling
    pool = Pool(processes=THREAD_CNT)
    results = pool.map(spawn_sagemaker_job, algos)
    print('==================================================================================')
    print(results)


if __name__ == "__main__":
    start = time.time()
    # single_process_main()  # ##  single process approach
    multi_process_main()  # ##  multi process approach
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\n\nTIME ELAPSED ::: {:0>2} hrs, {:0>2} mins, {:05.2f} secs".format(int(hours), int(minutes), seconds))
