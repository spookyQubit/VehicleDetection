import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import os

"""
Implementation of functions which are necessary to extract features from an image.
"""


def get_color_hist_features(img, nbins=32, bin_range=(0, 256)):
    """
    Given an image of shape (w, h, c=3),
    return the concatenated histogram of all color channels.
    So, the length of the returned array will be 3 * nbins
    """

    # Make sure that the image has three color channels
    assert (img.shape[2] == 3)

    ch0_hist = np.histogram(img[:, :, 0], bins=nbins)
    ch1_hist = np.histogram(img[:, :, 1], bins=nbins)
    ch2_hist = np.histogram(img[:, :, 2], bins=nbins)

    color_hist_features = np.concatenate((ch0_hist[0],
                                          ch1_hist[0],
                                          ch2_hist[0]))
    return color_hist_features


def get_spatial_pixel_value_features(img, resize_shape=(32, 32)):
    """
    Rescale the input image to resize_shape and
    subsequently return the array of pixel values
    """
    image = np.copy(img)
    return np.resize(image, resize_shape).ravel()


def get_hog_features(img, num_orientations,
                     pix_per_cell, cells_per_block,
                     vis=False, feature_vec=True):
    """
    Return the hog features of a 2D image.
    """
    assert (len(img.shape) == 2)

    if vis:
        features, hog_image = hog(image=img,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  orientations=num_orientations,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(image=img,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       orientations=num_orientations,
                       visualise=vis, feature_vector=feature_vec)
    return features


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_single_image_feature(img,
                             color_hist_params,
                             spatial_pixel_value_params,
                             hog_params,
                             color_space="YCrCb"):
    """
    :param img:
    :param color_hist_params:
    :param spatial_pixel_value_params:
    :param hog_params:
    :param color_space:
    :return: Return the feature of a single image
    """
    image_features = []

    # Convert color according to color space
    feature_image = np.copy(img)
    if color_space != "RGB":
        feature_image = convert_color(img, "RGB2" + color_space)

    # Get color hist features
    color_hist_features = get_color_hist_features(feature_image, color_hist_params["nbins"],
                                                  color_hist_params["bin_range"])
    # Get spatial pixel value features
    spatial_pixel_value_features = get_spatial_pixel_value_features(feature_image,
                                                                    spatial_pixel_value_params["resize_shape"])

    # Get hog features
    # Do not extract features with visualization on
    assert (hog_params["vis"] == False)
    hog_features = []
    if (hog_params["channels"] == "ALL") & (len(img.shape) == 3):
        for ch in range(img.shape[2]):
            hf = get_hog_features(feature_image[:, :, 1],
                                  hog_params["num_orientations"],
                                  hog_params["pix_per_cell"],
                                  hog_params["cells_per_block"],
                                  hog_params["vis"])
            hog_features.extend(hf)
    elif len(img.shape) == 3:
        hog_features = get_hog_features(feature_image[:, :, hog_params["channel"]],
                                        hog_params["num_orientations"],
                                        hog_params["pix_per_cell"],
                                        hog_params["cells_per_block"],
                                        hog_params["vis"])
    else:
        hog_features = get_hog_features(feature_image,
                                        hog_params["num_orientations"],
                                        hog_params["pix_per_cell"],
                                        hog_params["cells_per_block"],
                                        hog_params["vis"])

    image_features = np.concatenate((color_hist_features,
                                     spatial_pixel_value_features,
                                     hog_features))
    return image_features


def get_features_from_list_of_images(image_files_paths,
                                     color_hist_params,
                                     spatial_hist_params,
                                     hog_params,
                                     color_space="RGB"):
    """
    Define a function to extract features from a list of images
    Have this function call get_single_image_feature
    """

    features = []
    for img_file in image_files_paths:
        image = mpimg.imread(img_file)

        feature_image = np.copy(image)
        file_feature = get_single_image_feature(feature_image,
                                                color_hist_params,
                                                spatial_hist_params,
                                                hog_params,
                                                color_space)
        features.append(file_feature)

    # Returned array has dim: (num_samples x num_features)
    return np.vstack(features)
