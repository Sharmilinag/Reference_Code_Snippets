import numpy as np
import pandas as pd
import Constants as cons
from tabulate import tabulate
import Utils as utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import os
from sys import getsizeof
import sys
import argparse
import time
import pickle

# DATA_FILE = 'iris.data'  # todo: parameterize
# CATEGORICAL_COLUMNS = ['class_name']
# NUMERIC_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# TARGET_COLUMN = 'class_name'

# todo: take separate test data

'''
Module Restrictions:
    1. No missing values allowed in the data-set 
    2. Can't handle MULTI-LABEL Classification
    3. Target Column (feature) should be a Single Feature Only
    4. Target Class imbalance is NOT handled 
    5. The data needs to be one-hot-encoded
    6. No Drop Column Allowed (all columns expect the specified Target column will be treated as features)
'''


def best_model_classification():
    #   ##  Read the data and convert to numpy array for computation
    chunk_size = 200000
    data_set_cols = None
    data_array = None
    log_file = open('gen_log', 'a')
    log_file.write('=================================\n')
    log_file.close()
    for i, chunk in enumerate(pd.read_csv(cons.DATA_FILE, chunksize=chunk_size)):
        log_file = open('gen_log', 'a')
        log_file.write('Rows Processed :: {0}\n'.format(str(i * chunk_size)))
        log_file.close()
        tmp_array = chunk.to_numpy()
        if i == 0:
            data_set_cols = list(chunk.columns.values)
            data_array = tmp_array
        else:
            data_array = np.vstack([data_array, tmp_array])

    pickle_out = open("data.pickle", "wb")
    pickle.dump(data_array, pickle_out)
    pickle_out.close()

    #   ##  Get X and y
    target_col_index = data_set_cols.index(cons.TARGET_COLUMN)
    y = data_array[:, target_col_index]
    X = np.delete(data_array, target_col_index, 1)

    #   ##  Train and Test Splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)

    print('\nTraining Sample Count: {0}'.format(len(X_train)))
    print('Test Sample Count: {0}'.format(len(X_test)))
    print('Total {0} Classes Detected in Training Data'.format(len(np.unique(y_train))))
    print('Total {0} Classes Detected in Test Data \n'.format(len(np.unique(y_test))))
    result_file = open(cons.OUTPUT_FILE, 'a')
    result_file.write('\nTraining Sample Count: {0}\n'.format(len(X_train)))
    result_file.write('Test Sample Count: {0}\n'.format(len(X_test)))
    result_file.write('Total {0} Classes Detected in Training Data\n'.format(len(np.unique(y_train))))
    result_file.write('Total {0} Classes Detected in Test Data\n'.format(len(np.unique(y_test))))
    print('=====================================')

    #   ##  calculate class distribution and write out on report
    #   ##  1. Train Data Class Distribution report
    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    lst = []
    total = len(y_train) * 1.0
    for ele, cnt in zip(unique_elements, counts_elements):
        row = {'Class': ele,
               'Sample_Count': cnt,
               'Percentage': str(format(cnt / total, '.2f')) + '%'
               }
        lst.append(row)
    result_file.write('\nTrain Data Class Distribution\n')
    result_file.write(tabulate(lst, headers="keys", tablefmt="pretty"))
    result_file.write('\n')
    #   ##  2. Test Data Class Distribution report
    unique_elements, counts_elements = np.unique(y_test, return_counts=True)
    lst = []
    total = len(y_test) * 1.0
    for ele, cnt in zip(unique_elements, counts_elements):
        row = {'Class': ele,
               'Count': cnt,
               'Percentage': str(format(cnt / total, '.2f')) + '%'
               }
        lst.append(row)
    result_file.write('\nTest Data Class Distribution\n')
    result_file.write(tabulate(lst, headers="keys", tablefmt="pretty"))

    #   ##  Label encode the Target column
    label_encoder_target_train = preprocessing.LabelEncoder()
    y_train = label_encoder_target_train.fit_transform(y_train)
    print(label_encoder_target_train.classes_)
    label_encoder_target_test = preprocessing.LabelEncoder()
    y_test = label_encoder_target_test.fit_transform(y_test)
    print(label_encoder_target_test.classes_)

    #   ##  Get unique values for Test Class
    y_test_class = np.unique(y_test)

    #   ##  populate the parameter set for Gaussian_Process_Classifier and determine classification type
    if len(y_train) > 2:
        cons.CLASSIFIER_PARAMETERS['Gaussian_Process'] = cons.param_range_gpmc
        cons.CLASSIFIER_PARAMETERS['XGB_Classifier']['objective'] = 'multi:softmax'
        tsk_type = 'MULTI-CLASS CLASSIFICATION'
    elif len(y_train) == 2:
        cons.CLASSIFIER_PARAMETERS['Gaussian_Process'] = cons.param_range_gpbc
        cons.CLASSIFIER_PARAMETERS['XGB_Classifier']['objective'] = 'binary:logistic'
        tsk_type = 'BINARY CLASSIFICATION'
    else:
        result_file.write('\nERROR !! Only One Single Class was found in Training Data Target Column')
        raise ValueError('Only One Single Class was found in Training Data Target Column')
    #   ##  write out task type in the report
    result_file.write('\n\nTask Type: {0}\n'.format(tsk_type))

    result_file.close()

    #   ##  Scala data_set
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)

    #   ## Train and Evaluate Various Models based on Various Algorithms using sklearn
    # Prepare the list of estimators with tuned hyper parameters (used during Ensemble modeling)
    ensemble_clfs_hard_voting = []
    ensemble_clfs_soft_voting = []
    result_file = open(cons.OUTPUT_FILE, 'a')
    result_file.write('\n\nAlgorithm Comparison\n\n')
    result_file.close()
    clf_performance = []
    for clf_name, clf in cons.CLASSIFIER_ALGOS.items():
        # clf = RandomForestClassifier()
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # clf_row = {
        #     'classifier': clf_name,
        #     'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_test, y_pred),
        #     'weighted_precision': sklearn.metrics.precision_score(y_test, y_pred, average='weighted',
        #                                                           labels=y_test_class),
        #     'weighted_recall': sklearn.metrics.recall_score(y_test, y_pred, average='weighted', labels=y_test_class),
        #     'weighted_f1': sklearn.metrics.f1_score(y_test, y_pred, average='weighted', labels=y_test_class),
        #     'parameters': 'DEFAULT PARAMETERS'
        # }
        # print('Base Model Scores:   ')
        # print(clf_row)
        # print('\n=====================================\n')
        # clf_performance.append(clf_row)
        # sys.exit(0)
        # result_file = open(cons.OUTPUT_FILE, 'a')
        # result_file.write('\n' + clf_name + '\n')
        #   ##  Grid Search for Hyper Parameter Tuning
        # Set the scoring parameter
        print(clf_name)
        print(cons.CLASSIFIER_PARAMETERS[clf_name])
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')
        # Instantiate the grid search model
        grid_search = RandomizedSearchCV(estimator=clf, param_distributions=cons.CLASSIFIER_PARAMETERS[clf_name],
                                         cv=3, n_jobs=-1, verbose=0, scoring=scorer)
        # Fit the grid search to the data
        start = time.time()
        grid_search.fit(X_train, y_train)  # Train the Best Model
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\n\nTRAIN TIME ELAPSED ::: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        # print('Best Parameters : ')
        # print(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        #   ##  record model with tuned hyper parameters
        ensemble_clfs_hard_voting.append((clf_name, best_model))
        if clf_name not in ['Passive_Aggressive_Classifier',
                            'Ridge_Classifier']:
            ensemble_clfs_soft_voting.append((clf_name, best_model))
        #   ##  record model prediction performance
        clf_row = {
            'classifier': clf_name,
            'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_test, y_pred),
            'weighted_precision': sklearn.metrics.precision_score(y_test, y_pred, average='weighted',
                                                                  labels=y_test_class),
            'weighted_recall': sklearn.metrics.recall_score(y_test, y_pred, average='weighted', labels=y_test_class),
            'weighted_f1': sklearn.metrics.f1_score(y_test, y_pred, average='weighted', labels=y_test_class),
            'training_time': "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
            'parameters': str(grid_search.best_params_)
        }
        # print('Parameter Tuned Model Scores :   ')
        print(clf_row)
        clf_performance.append(clf_row)
        # result_file.write("\n")
        # result_file.write(str(clf_row))
        # result_file.close()
        print('=====================================')

    #   ##  Building Ensemble Voting Models
    voting_params = ['hard', 'soft']
    i = 1
    for voting_param in voting_params:
        if voting_param == 'hard':
            clf = sklearn.ensemble.VotingClassifier(estimators=ensemble_clfs_hard_voting, voting=voting_param)
        elif voting_param == 'soft':
            clf = sklearn.ensemble.VotingClassifier(estimators=ensemble_clfs_soft_voting, voting=voting_param)
        else:
            continue
        start = time.time()
        clf.fit(X_train, y_train)  # Train the Model
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\n\nTRAIN TIME ELAPSED ::: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        y_pred = clf.predict(X_test)
        #   ##  record model prediction performance
        clf_row = {
            'classifier': 'Stacked_Ensemble_Model_{0}'.format(i),
            'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_test, y_pred),
            'weighted_precision': sklearn.metrics.precision_score(y_test, y_pred, average='weighted',
                                                                  labels=y_test_class),
            'weighted_recall': sklearn.metrics.recall_score(y_test, y_pred, average='weighted', labels=y_test_class),
            'weighted_f1': sklearn.metrics.f1_score(y_test, y_pred, average='weighted', labels=y_test_class),
            'training_time': "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
            'parameters': '[voting={0}, clf={1}]'.format(voting_param, clf)
        }
        print('Stacked Ensemble Model {0}'.format(i))
        print(clf_row)
        clf_performance.append(clf_row)
        print('===========================================')
        i += 1

    #   ##  Print out the Performance table
    result_file = open(cons.OUTPUT_FILE, 'a')
    # clf_perf_df = pd.DataFrame(clf_performance)
    result_file.write(tabulate(clf_performance, headers="keys"))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Get Best Model for the Data')
    parser.add_argument('--s3_bucket', help='S3 bucket name for the artifacts'),
    parser.add_argument('--data', help='Dataset Path on S3')
    parser.add_argument('--target_col', help='Target Column for Prediction')
    parser.add_argument('--task_type', choices=['classification', 'regresion', 'clustering'],
                        help='Task Type : classification, regression etc'),
    parser.add_argument('--test_data', const=None, nargs='?', help='Test Dataset (hold-out-set) Full Path')

    results = parser.parse_args()
    print(type(results.data), results.data)
    print(type(results.target_col), results.target_col)
    print(type(results.task_type), results.task_type)
    print(type(results.test_data), results.test_data)
    print(type(results.test_data), results.s3_bucket)
    return results.s3_bucket, results.target_col, results.data, results.test_data, results.task_type


def init_result_file():
    # Initialize the Results file
    file_header_str = "HYPER PARAMETER TUNED TRAINED MODEL COMPARISON REPORT"
    file_header_highlighter = ''
    for _ in file_header_str:
        file_header_highlighter += '='
    result_file = open(cons.OUTPUT_FILE, 'w')
    result_file.write('\n\t{0}\n'.format(file_header_highlighter))
    result_file.write('\n\t{0}'.format(file_header_str))
    result_file.write('\n\t{0}\n'.format(file_header_highlighter))
    result_file.write('\nData File: {0}\n'.format(cons.DATA_FILE))
    result_file.close()


if __name__ == "__main__":
    s3_bucket, target_col, d_file, t_d_file, task_type = parse_arguments()
    cons.populate_run_time_parameters(s3_bucket, target_col, d_file, t_d_file)
    init_result_file()
    if task_type == 'classification':
        start = time.time()
        best_model_classification()
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\n\nTIME ELAPSED ::: {:0>2} hrs, {:0>2} mins, {:05.2f} secs".format(int(hours), int(minutes), seconds))
        result_file = open(cons.OUTPUT_FILE, 'a')
        result_file.write("\n\n\tTOTAL REPORT GENERATION TIME ::: {:0>2}:{:0>2}:{:05.2f}"
                          .format(int(hours), int(minutes), seconds)
                          )
        result_file.close()
    #   ##  upload resultant report to S3
    utils.upload_file(cons.OUTPUT_FILE, s3_bucket, 'MODEL_SELECTION/RESULT_REPORTS/' + cons.OUTPUT_FILE)

# python Driver.py --s3_bucket fis-ada-bucket --data MODEL_SELECTION/DATASETS/iris.data
#  --config MODEL_SELECTION/DATASET_CONFIGS/dataset_config.json --task_type classification
