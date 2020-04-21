import argparse
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def format_data():
    # Take the set of files and read them all into a single pandas data_frame
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)
    # Assuming labels are in the first column
    train_y = train_data.ix[:, 0]
    train_X = train_data.ix[:, 1:]
    return train_X, train_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper-parameters are described here. In this simple example we are just including one hyper-parameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    print('==================', args.max_leaf_nodes)
    print('==================', args.model_dir)

    #   ##  read data
    X_train, y_train, = format_data()
    print(X_train.shape, y_train.shape)

    # We determine the number of leaf nodes using the hyper-parameter above
    max_leaf_nodes = args.max_leaf_nodes

    # Now use scikit-learn's decision tree classifier to train the model
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(X_train, y_train)

    # Save the decision tree model
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
