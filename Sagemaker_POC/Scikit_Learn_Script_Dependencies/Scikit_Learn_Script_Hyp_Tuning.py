import argparse
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import Scikit_Learn_Hyp_Tuning_Constants as hyp_cons
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import time
import json
import numpy as np


def format_data(data_path):
    # Take the set of files and read them all into a single pandas data_frame
    input_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that this channel was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(data_path))
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    data = pd.concat(raw_data)
    # Assuming labels are in the first column
    data_y = data.ix[:, 0]
    data_X = data.ix[:, 1:]
    return data_X, data_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper-parameters are described here. In this simple example we are just including one hyper-parameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    #   ##  Algorithm Choice
    parser.add_argument('--algorithm', type=str, required=True)
      ##  Algorithm Choice
    # parser.add_argument('--validation_data', type=str, required=True)
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    print('==================', args.max_leaf_nodes)
    print('==================', args.model_dir)
    print('==================', args.algorithm)

    #   ##  read data
    X_train, y_train, = format_data(args.train)
    print(X_train.shape, y_train.shape)
    X_test, y_test = format_data(args.test)
    print(X_test.shape, y_test.shape)
    y_test_class = np.unique(y_test)
    print(y_test_class)

    # We determine the number of leaf nodes using the hyper-parameter above
    max_leaf_nodes = args.max_leaf_nodes

    # Now use scikit-learn's base classifier to train the model
    base_clf = hyp_cons.CLASSIFIER_ALGORITHMS[args.algorithm]
    random_search = RandomizedSearchCV(base_clf,
                                       param_distributions=hyp_cons.CLASSIFIER_PARAMETERS[args.algorithm],
                                       cv=5,
                                       n_jobs=-1,
                                       random_state=0)

    #   ##  Train the model with the Best Params
    start = time.time()
    random_search.fit(X_train, y_train)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    #   ##  Evaluate the best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    #   ##  record model prediction performance
    clf_row = {
        'classifier': args.algorithm,
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'weighted_precision': precision_score(y_test, y_pred, average='weighted',
                                              labels=y_test_class),
        'weighted_recall': recall_score(y_test, y_pred, average='weighted', labels=y_test_class),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted', labels=y_test_class),
        'training_time': "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
        'parameters': str(random_search.best_params_)
    }

    # Save the decision tree model
    joblib.dump(best_model, os.path.join(args.model_dir, "model.joblib"))

    with open(os.path.join(args.output_data_dir, "best_params_details.json"), 'w') as json_file:
        json.dump(clf_row, json_file)
