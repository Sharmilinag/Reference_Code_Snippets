import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import sys

'''
Ref: 
https://www.statisticshowto.datasciencecentral.com/goodness-of-fit-test/
https://www.kdnuggets.com/2019/05/sample-huge-dataset-machine-learning.html
http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html
'''


def sub_sample(data_full_path, tar_col, sample_percentage, p_val_threshold, col_config_file):
    #   ##  Parse column config if not None
    if col_config_file is not None:
        with open(col_config_file) as json_file:
             col_config = json.load(json_file)
    #   ##  Read the data and convert to numpy array for computation
    chunk_size = 200000
    data_set_cols = None
    data_array = None
    log_file = open('gen_log', 'a')
    log_file.write('=================================\n')
    log_file.close()
    for i, chunk in enumerate(pd.read_csv(data_full_path, chunksize=chunk_size)):
        log_file = open('gen_log', 'a')
        log_file.write('Rows Processed :: {0}\n'.format(str(i * chunk_size)))
        log_file.close()
        tmp_array = chunk.to_numpy()
        if i == 0:
            #   ##  save column names
            data_set_cols = list(chunk.columns.values)
            #   ##  store into the numpy array
            data_array = tmp_array
            #   ##  if column config is defined, check integrity of column names in the config file
            #   ##      making sure that all columns in the data-set are defined in the column config file

        else:
            #   ##  stack it on top of the existing numpy array
            data_array = np.vstack([data_array, tmp_array])

    #   ##  determine column index and data-types (Categorical or Continuous)
    #   ##  if Column Config File isn't defined then run heuristics to determine the type
    data_col_indx_dtypes = {}
    if col_config_file is None:
        total_len = data_array.shape[0]
        ''''
        NOTE: a column is determined to be "Categorical" if the following condition is met:
                count(unique(col)) <= 0.3% of the total values in the column
              Otherwise, its considered as "Continuous" column
        '''
        for indx, col in enumerate(data_set_cols):
            if (len(np.unique(data_array[:, indx])) * 1.0 / total_len) > 0.003:
                data_col_indx_dtypes[indx] = 'Continuous'
            else:
                data_col_indx_dtypes[indx] = 'Categorical'
    else:
        #   ##  if Column config is defined, the data-type is read in directly from the config
        for key, val in col_config.items():
            for col in val:
                data_col_indx_dtypes[data_set_cols.index(col)] = key

    #   ##  prepare data for sub-sampling
    target_col_index = data_set_cols.index(tar_col)
    y = data_array[:, target_col_index]
    X = np.delete(data_array, target_col_index, 1)

    #   ##  create the sub sample
    sub_sample_stats = None
    trial_cnt = 1
    # ##  Maximum trials to create a sub-sample is 200 (error raised if sub-sample doesn't meet statistical
    #                                                      significance after 200 trials)
    while trial_cnt <= 200:

        #   ##  Create the Sub-Sample
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sample_percentage, random_state=trial_cnt)

        #   ##  iterate over all columns and determine that the sub-sample is statistically significant
        '''The histograms of categorical variables can be compared using "Pearson’s chi-square test (Goodness of Fit 
        Test)", while the cumulative distribution functions of the numerical variables can be compared using the 
        "Kolmogorov-Smirnov test". 
        
        Both statistical tests work under the null hypothesis that the sample has the same distribution of the 
        population. Since a sample is made by many columns and we want all of them to be significative, we can reject 
        the null hypothesis if the p-value of at least one of the tests is lower than the usual 5% confidence level. 
        In other words, we want every column to pass the significance test in order to accept the sample as valid. '''
        sub_sample_stats = []
        is_low_p_val = False
        for indx, col in enumerate(data_set_cols):
            if indx == target_col_index:
                continue
            print('========================', col)
            if data_col_indx_dtypes[indx] == 'Continuous':
                test_stats, p_val = stats.ks_2samp(X_test[:, indx], X[:, indx])
            else:  # ##  when variable is Categorical
                #   ##  Prepare data for Chi Squared Test (Goodness of Fit Test)
                #   ##  1. Prepare the Count Tables
                population_table = pd.crosstab(index=X[:, indx], columns="count")
                sample_table = pd.crosstab(index=X_test[:, indx], columns="count")
                #   ##  2. Calculate the ratios and observed and expected values for the Test
                observed = sample_table
                population_ratios = population_table / len(X)  # Get population ratios
                expected = population_ratios * len(X_test)  # Get expected counts
                #   ##  3. Calculate the Chi Squared Test (Goodness of Fit Test)
                try:
                    test_stats, p_val = stats.chisquare(f_obs=observed,  # Array of observed counts
                                                        f_exp=expected)  # Array of expected counts
                    test_stats = test_stats[0]
                    p_val = p_val[0]
                except ValueError:
                    test_stats, p_val = stats.ks_2samp(X_test[:, indx], X[:, indx])
            # ##  check whether p-val is above p-val threshold (else null hypothesis is rejected)
            '''
            if p-val is below p-val threshold; the sampling for the current column doesn't have the same probability 
            distribution as this column from the population. Hence sub-sample wouldn't have the same distribution 
            as the Population 
            '''
            if p_val < p_val_threshold:
                is_low_p_val = True
                break
            # ##  Collect the Stats and P-Value for logs
            row = {
                "Columns": col,
                "Test_Statistics": test_stats,
                "P_Value": p_val
            }
            sub_sample_stats.append(row)

        trial_cnt = trial_cnt + 1
        #   ##  check if the previous loop had been exited because a low p-val was detected; if so repeat trial
        if is_low_p_val:
            if trial_cnt == 200:
                sub_sample_stats_df = pd.DataFrame(sub_sample_stats)
                sub_sample_stats_df.to_csv('goodness_of_fit_results_ERROR.csv')     # todo make filename as constant
                raise ValueError('The "{0}" Dataset could not be Sub-Sampled based on the p-value threshold of {1}'
                                 .format(data_full_path, p_val_threshold))
            continue

    sub_sample_stats_df = pd.DataFrame(sub_sample_stats)
    print(sub_sample_stats_df)
    sub_sample_stats_df.to_csv('goodness_of_fit_results.csv')   # todo make filename as constant

    #   ##  write out the small data-set


def parse_arguments():
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal. Sample Size Percentage expressed in "
                                             "decimals" % (x,))
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]. Sample Size Percentage expressed in decimals"
                                             % (x,))
        return x

    parser = argparse.ArgumentParser(description='Sub Sampling a Huge Dataset')
    parser.add_argument('--s3_bucket', help='S3 bucket name for the artifacts')
    parser.add_argument('--data', help='Dataset Path on S3')
    parser.add_argument('--sample_percentage', type=restricted_float,
                        help='Sample Size Percentage expressed in decimals')  # eg: 0.4, 0.3 etc
    parser.add_argument('--target_col', help='Target Column for Prediction')
    parser.add_argument('--p_val_threshold', default=0.05, type=restricted_float,
                        help='Specify a threshold (0.0 to 1.0) for P-Value cut-off to Accept Statistical Significance '
                             'of Sample')
    parser.add_argument('--col_config', default=None,
                        help='Path to JSON formatted config file specifying Continuous and '
                             'Categorical columns')
    results = parser.parse_args()
    print(type(results.data), results.data)
    print(type(results.s3_bucket), results.s3_bucket)
    print(type(results.target_col), results.target_col)
    print(type(results.sample_percentage), results.sample_percentage)

    return results.s3_bucket, results.data, results.target_col, results.sample_percentage, \
           results.p_val_threshold, results.col_config


if __name__ == "__main__":
    s3_bucket, data_path, tar_col, sample_percentage, p_val_threshold, col_config = parse_arguments()
    sub_sample('s3://{0}/{1}'.format(s3_bucket, data_path), tar_col, sample_percentage, p_val_threshold, col_config)
