import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier)
import Utils as utils
from xgboost.sklearn import XGBClassifier

'''
Refs:
# https://optunity.readthedocs.io/en/latest/notebooks/notebooks/sklearn-automated-classification.html
# https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
# https://scikit-learn.org/stable/modules/grid_search.html
# https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
# https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
'''

LOCAL_DATA_PATH = 'data'
OUTPUT_FILE = 'Model_Comparison_Report.txt'

#   ##  Run-time parameters
DATA_COLUMN_CONFIG = None  # ## holds which columns are numeric, categorical, target etc
DATA_FILE = None
TEST_DATA_FILE = None
TARGET_COLUMN = None


def populate_run_time_parameters(s3_bucket, target_col, d_file, t_d_file):
    global DATA_COLUMN_CONFIG, DATA_FILE, TEST_DATA_FILE, TARGET_COLUMN
    DATA_FILE = 's3://' + s3_bucket + '/' + d_file
    TARGET_COLUMN = target_col
    if t_d_file is None:
        TEST_DATA_FILE = None
    else:
        TEST_DATA_FILE = 's3://' + s3_bucket + '/' + t_d_file


#   ##  Key name in CLASSIFIER_ALGOS should match key name in CLASSIFIER_PARAMETERS

#   ##  Classification Estimators
CLASSIFIER_ALGOS = {
    # 'Gaussian_Process': GaussianProcessClassifier(),
    'Logistic_Regression': LogisticRegression(),
    'Passive_Aggressive_Classifier': PassiveAggressiveClassifier(),
    'Ridge_Classifier': RidgeClassifier(),
    'Decision_Tree': DecisionTreeClassifier(),
    'Random_Forest': RandomForestClassifier(),
    'Nearest_Neighbors': KNeighborsClassifier(),
    'Quadratic_Discriminant_Analysis': QuadraticDiscriminantAnalysis(),
    'Gaussian_Naive_Bayes': GaussianNB(),
    # 'Multi_Layer_Perceptron': MLPClassifier(),
    'Gradient_Boosting_Classifier': GradientBoostingClassifier(),
    'Ada_Boost': AdaBoostClassifier()#,
    #'XGB_Classifier': XGBClassifier()
}

#   ##  Parameter range for XGBoost
param_range_xgb = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                   "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                   "min_child_weight": [1, 3, 5, 7],
                   "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                   "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
                   'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
                   'seed': [8],
                   'nthread': [10]
                   }

#   ##  Parameter range for Logistic_Regression
param_range_lr = {'penalty': ['l1', 'l2', 'none'],
                  'C': [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
                  'class_weight': ['balanced', None],
                  'random_state': [8],
                  'solver': ['lbfgs', 'saga']
                  }

#   ##  Parameter range for Passive_Aggressive_Classifier
param_range_pac = {'C': [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
                   'random_state': [8],
                   'n_jobs': [1]
                   }

#   ##  Parameter range for Ridge_Classifier
param_range_rc = {'alpha': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'class_weight': ['balanced', None],
                  'random_state': [8],
                  'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                  }

#   ##  Parameter range for Random Forest
param_range_rf = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=50, num=5)],
                  'criterion': ['gini', 'entropy'],
                  # 'max_features': ['auto', 'sqrt'],
                  'max_depth': [int(x) for x in np.linspace(10, 50, num=5)],
                  # 'min_samples_split': [0.1, 0.2, 0.3, 0.8, 0.9],
                  # 'min_samples_leaf': [1, 2, 4],
                  'bootstrap': [True, False],
                  'class_weight': ['balanced', None]
                  }
param_range_rf['max_depth'].append(None)

#   ##  Parameters for K-Nearest Neighbors
param_range_nn = {'n_neighbors': [int(x) for x in np.linspace(start=4, stop=25, num=10)],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'leaf_size': [5, 10, 20, 30, 40, 50],
                  'p': [1, 2],  # ##  Power parameter for the Minkowski metric (p=1-->euclidean, p=2-->manhattan)
                  'n_jobs': [1]
                  }
param_range_nn['n_neighbors'].append(3)

# QuadraticDiscriminantAnalysis (priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
param_range_qda = {'priors': [None],
                   'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   'store_covariance': [True, False],
                   'tol': [0.0001]
                   }

# GaussianNB(priors=None, var_smoothing=1e-09)
param_range_gnb = {'priors': [None],
                   'var_smoothing': [1e-09]
                   }

# GradientBoostingClassifier
param_range_gb = {'n_estimators': [10, 25, 50, 100, 125, 150, 500],
                  'learning_rate': [0.001, 0.05, 0.01, 0.05, 0.1, 0.3, 1],
                  'subsample': [0.2, 0.5, 0.7, 1.0],
                  'max_depth': [3, 7, 9],
                  'random_state': [8]
                  }

# AdaBoostClassifier
param_range_adab = {'base_estimator': [LogisticRegression(),
                                       GaussianNB(),
                                       DecisionTreeClassifier(),
                                       PassiveAggressiveClassifier(),
                                       RidgeClassifier(),
                                       RandomForestClassifier()
                                       ],
                    'n_estimators': [10, 25, 50, 100, 125, 150],
                    'learning_rate': [0.001, 0.05, 0.01, 0.05, 0.1, 0.3, 1],
                    'algorithm': ['SAMME', 'SAMME.R'],
                    'random_state': [8]
                    }

# MLPClassifier
param_range_mlp = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                   'activation': ['identity', 'logistic', 'tanh', 'relu'],
                   'solver': ['lbfgs', 'sgd', 'adam'],  ## adam variations
                   'alpha': list(10.0 ** -np.arange(1, 10)),
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'warm_start': [True, False],
                   'early_stopping': [True, False],
                   'validation_fraction': [0.1, 0.25, 0.5, 0.75, 0.9],
                   'shuffle': [True, False],
                   'random_state': [8]
                   }

# DecisionTreeClassifier
param_range_dtc = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best'],  # skipping 'random' - use only in case of overfitting
                   'max_depth': [2, 5, 10, 20, 50, None],
                   'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 1 to 40 for CART algo
                   'min_samples_leaf': [1, 2, 4, 10, 15, 20],  # 1 to 20 for the CART algorithm
                   'max_features': ['auto', 'sqrt', 'log2', None],
                   'class_weight': ['balanced', None]
                   }

# GaussianProcessClassifier
# binary classification
param_range_gpbc = {'kernel': [1.0 * RBF(1.0),
                               1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                               1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
                               1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                                    length_scale_bounds=(0.1, 10.0),
                                                    periodicity_bounds=(1.0, 10.0)),
                               ConstantKernel(0.1, (0.01, 10.0))
                               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
                               1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                                            nu=1.5)],  # "1.0 * RBF(1.0)" is used as default
                    'optimizer': ['fmin_l_bfgs_b'],  # 'fmin_l_bfgs_b' as default
                    'random_state': [8],
                    'n_jobs': [1]
                    }

# GaussianProcessClassifier
# multiclass classification
param_range_gpmc = {'kernel': [1.0 * RBF(1.0),
                               1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                               1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
                               1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                                    length_scale_bounds=(0.1, 10.0),
                                                    periodicity_bounds=(1.0, 10.0)),
                               1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                                            nu=1.5)],  # "1.0 * RBF(1.0)" is used as default
                    'optimizer': ['fmin_l_bfgs_b'],  # 'fmin_l_bfgs_b' as default
                    'multi_class': ['one_vs_rest'],  # no prediction probability for one_vs_one
                    'random_state': [8],
                    'n_jobs': [1]
                    }

CLASSIFIER_PARAMETERS = {
    'Logistic_Regression': param_range_lr,
    'Passive_Aggressive_Classifier': param_range_pac,
    'Ridge_Classifier': param_range_rc,
    'Decision_Tree': param_range_dtc,
    'Random_Forest': param_range_rf,
    'Nearest_Neighbors': param_range_nn,
    'Quadratic_Discriminant_Analysis': param_range_qda,
    'Gaussian_Naive_Bayes': param_range_gnb,
    # 'Multi_Layer_Perceptron': param_range_mlp,
    'Gaussian_Process': None,  # this is populated at run-time (depending on whether its a binary or multi-class prob)
    'Gradient_Boosting_Classifier': param_range_gb,
    'Ada_Boost': param_range_adab,
    'XGB_Classifier': param_range_xgb
}
