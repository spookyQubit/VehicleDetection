import featureExtractor
import numpy as np
import os
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics.scorer import accuracy_scorer
from sklearn.svm import SVC
import pickle

"""
Implementation of functions which are necessary to
create a classifier which can distinguish between vehicles and non-vehicles.
"""


def get_file_names(root_directory, extension=(".png")):
    """
    Given a root directory, traverse all sub-directories
    and return a list of all files
    which ends with elements in extension
    """

    file_names = []
    for root, dirs, files in os.walk(root_directory):
        for file_name in files:
            if file_name.endswith(extension):
                file_names.append(os.path.join(root, file_name))
    return file_names


def my_accuracy_scorer(*args):
    score = accuracy_scorer(*args)
    print('score is {}'.format(score))
    return score


if __name__ == "__main__":

    """
    Get image file paths
    """
    # Get file names for vehicles and non-vehicles
    vehicles_root_dir = "../vehicles"
    non_vehicles_root_dir = "../non-vehicles"

    vehicles = get_file_names(vehicles_root_dir)
    non_vehicles = get_file_names(non_vehicles_root_dir)

    """
    Extract features
    """
    # Define params
    color_hist_params = {"nbins": 32, "bin_range": (0, 256)}
    spatial_pixel_value_params = {"resize_shape": (32, 32)}
    hog_params = {"channels": "ALL", "num_orientations": 9,
                  "pix_per_cell": 8, "cells_per_block": 2,
                  "vis": False}
    color_space = "YCrCb"

    # Get the vehicle/non-vehicle features
    vehicle_sample_size = len(vehicles)
    non_vehicle_sample_size = len(non_vehicles)
    X = featureExtractor.get_features_from_list_of_images(vehicles[:vehicle_sample_size] + non_vehicles[:non_vehicle_sample_size],
                                                          color_hist_params,
                                                          spatial_pixel_value_params,
                                                          hog_params,
                                                          color_space)
    # Get the car labels
    y = np.hstack((np.ones(vehicle_sample_size),
                   np.zeros(non_vehicle_sample_size)))

    # Shuffle the data points
    X, y = shuffle(X, y)

    """
    Preporcess the data to be fit into the classifier
    and build a pipeline.
    """

    # Make a test/train split
    # Split up data into randomized training and test sets
    rand_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=rand_state)
    """
    pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
    C_param_range = [0.01, 1.0]
    gamma_param_range = ['auto']
    param_grid = [{'clf__C': C_param_range, 'clf__kernel': ['linear']},
                  {'clf__C': C_param_range, 'clf__gamma': gamma_param_range, 'clf__kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=my_accuracy_scorer, cv=3, verbose=1, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    clf = SVC(C=1.0, kernel="rbf")
    clf.fit(scaler.transform(X_train), y_train)
    print('Test accuracy: %.3f' % clf.score(scaler.transform(X_test), y_test))

    """
    Save the params and the model
    """
    outputFile = "params_model.p"
    if outputFile is not None:
        estimator = clf.fit(scaler.transform(X), y)
        data = {
            "estimator": estimator,
            "standard_scaler": scaler,
            "color_space": color_space,
            "color_hist_params": color_hist_params,
            "spatial_pixel_value_params": spatial_pixel_value_params,
            "hog_params": hog_params
        }
        with open(outputFile, "wb") as f:
            pickle.dump(data, f)




