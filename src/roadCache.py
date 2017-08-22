import itertools
import numpy as np
from scipy.ndimage.measurements import label


def merge_detected_windows(img, box_list, threshold=2):
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    for box in box_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    heatmap[heatmap <= threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)

    # label connected heatmaps
    labels = label(heatmap)
    return merge_labeled_bboxes(labels)


def merge_labeled_bboxes(labels):
    mergedBoxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        mergedBoxes.append(bbox)
    return mergedBoxes


class WindowHistory:
    def __init__(self, history_size, threshold):
        self.history = []
        self.history_size = history_size
        self.threshold = threshold

    def addBoxes(self, boxList):
        '''
        Add the boxes from this image to the history queue
        :param boxList:
        :return:
        '''
        if len(self.history) == self.history_size:
            # drop the oldest box list and append the latest
            self.history = self.history[1:]
            # append the latest list of boxes
            self.history.append(boxList)
        elif len(self.history) < self.history_size:
            self.history.append(boxList)
        else:
            raise AssertionError("History cannot be more than 10")

    def getWindows(self, img):
        '''
        Get the windows for the current image
        :param img:
        :return:
        '''
        if len(self.history) < self.history_size:
            # we don't have sufficient history, so we return nothing
            return []
        else:
            # we merge windows from all the previous detections and return the merge windows
            boxes = list(itertools.chain.from_iterable(self.history))
            return merge_detected_windows(img, box_list=boxes, threshold=self.threshold)
