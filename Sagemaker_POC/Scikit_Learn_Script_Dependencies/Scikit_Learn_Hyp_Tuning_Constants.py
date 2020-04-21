from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier)
import numpy as np

#   ##  Classification Estimators
CLASSIFIER_ALGORITHMS = {
    'Passive_Aggressive_Classifier': PassiveAggressiveClassifier(),
    'Ridge_Classifier': RidgeClassifier(),
    'Decision_Tree': DecisionTreeClassifier(),
    'Random_Forest': RandomForestClassifier(),
    'Nearest_Neighbors': KNeighborsClassifier(),
    'Quadratic_Discriminant_Analysis': QuadraticDiscriminantAnalysis(),
    'Gradient_Boosting_Classifier': GradientBoostingClassifier(),
    'Ada_Boost': AdaBoostClassifier()
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
param_range_rf = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=20)],
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [int(x) for x in np.linspace(10, 50, num=5)],
                  'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'min_samples_leaf': [1, 2, 4],
                  'bootstrap': [True, False],
                  'class_weight': ['balanced', None]
                  }
param_range_rf['max_depth'].append(None)

#   ##  Parameters for DecisionTreeClassifier
param_range_dtc = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best'],  # skipping 'random' - use only in case of overfitting
                   'max_depth': [2, 5, 10, 20, 50, None],
                   'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 1 to 40 for CART algo
                   'min_samples_leaf': [1, 2, 4, 10, 15, 20],  # 1 to 20 for the CART algorithm
                   'max_features': ['auto', 'sqrt', 'log2', None],
                   'class_weight': ['balanced', None]
                   }

#   ##  Parameters for K-Nearest Neighbors
param_range_nn = {'n_neighbors': [int(x) for x in np.linspace(start=1, stop=25, num=10)],
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
                                       RidgeClassifier(),
                                       RandomForestClassifier()
                                       ],
                    'n_estimators': [10, 25, 50, 100, 125, 150],
                    'learning_rate': [0.001, 0.05, 0.01, 0.05, 0.1, 0.3, 1],
                    'algorithm': ['SAMME'],
                    'random_state': [8]
                    }

#   ##  Classification Parameters
CLASSIFIER_PARAMETERS = {
    'Passive_Aggressive_Classifier': param_range_pac,
    'Ridge_Classifier': param_range_rc,
    'Decision_Tree': param_range_dtc,
    'Random_Forest': param_range_rf,
    'Nearest_Neighbors': param_range_nn,
    'Quadratic_Discriminant_Analysis': param_range_qda,
    'Gradient_Boosting_Classifier': param_range_gb,
    'Ada_Boost': param_range_adab
}