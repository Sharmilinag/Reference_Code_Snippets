# import tarfile
# tf = tarfile.open("s3://ada-engine/sklearn-Ada-Boost-20200415-105952/output/output.tar.gz")
# print(tf.extractall())

import pandas as pd
df = pd.read_csv('s3://ada-engine/sklearn-Ada-Boost-20200415-105952/output/output.tar.gz')
print(df)


# import Constants as cons
# from multiprocessing import Pool
#
# sm_job_name = 'sklearn-test-job20200414-084907'
# # bucket = cons.BOTO_S3.Bucket(cons.S3_BUCKET)
# # for obj in bucket.objects.filter(Prefix='{}/'.format(sm_job_name)):
# #     cons.BOTO_S3.Object(bucket.name, obj.key).delete()
#
# def spawn_sagemaker_job(num):
#     print(num)
#
# def procs():
#     nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#     pool = Pool(processes=5)
#     results = pool.map(spawn_sagemaker_job, nums)
#     print(results)
#
# if __name__ == "__main__":
#     procs()




