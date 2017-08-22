from moviepy.editor import VideoFileClip
import cv2
import pickle
import numpy as np
from featureExtractor import convert_color
from featureExtractor import get_hog_features
from featureExtractor import get_spatial_pixel_value_features
from featureExtractor import get_color_hist_features
from roadCache import WindowHistory
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws rectangles on img at locations defined by bboxes
    img:  an image
    bboxes: list of tuples (bboxes)
            which mark the diagonaly opposite corners of a rectangle.
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def find_cars(img,
              ystart, ystop, scale,
              clf,
              scaler,
              orient,
              pix_per_cell, cell_per_block,
              spatial_size,
              hist_bins, color_space='YCrCb'):
    """
    Given an image, and the parameters to sub-sample the image with boxes,
    the function returns a list of bounding boxes which are predicted to have a vehicle.
    """

    bboxes = []

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2' + color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = get_spatial_pixel_value_features(subimg, resize_shape=(spatial_size, spatial_size))
            hist_features = get_color_hist_features(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = np.hstack((hist_features, spatial_features, hog_features)).reshape(1, -1)

            test_prediction = clf.predict(scaler.transform(test_features))

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                bboxes.append(bbox)

    return bboxes


class CarFinder():
    def __init__(self,
                 ystart, ystop, scales,
                 params_model_file, histSize=10):
        self.ystart = ystart
        self.ystop = ystop
        self.scales = scales
        self.threshold = 2
        self.histSize = histSize
        self.window_history = WindowHistory(self.histSize,
                                            0.5 * self.threshold * self.histSize * len(self.scales))
        self._initialize_model_params(params_model_file)

    def _initialize_model_params(self, params_model_file):
        with open(params_model_file, "rb") as f:
            data = pickle.load(f)
            self.clf = data["estimator"]
            self.scaler = data["standard_scaler"]
            self.orient = data["hog_params"]["num_orientations"]
            self.pix_per_cell = data["hog_params"]["pix_per_cell"]
            self.cell_per_block = data["hog_params"]["cells_per_block"]
            self.spatial_size = data["spatial_pixel_value_params"]["resize_shape"][0]
            self.hist_bins = data["color_hist_params"]["nbins"]
            self.color_space = data["color_space"]
            print(self.__dict__)

    def process_image(self, img):
        #bgrImg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # WHY??
        bgrImg = np.copy(img)

        bboxes = []
        for scale in self.scales:
            boxes = find_cars(bgrImg,
                              self.ystart, self.ystop, scale,
                              self.clf,
                              self.scaler,
                              self.orient,
                              self.pix_per_cell, self.cell_per_block,
                              self.spatial_size,
                              self.hist_bins, self.color_space)
            bboxes.extend(boxes)

        self.window_history.addBoxes(bboxes)
        image = np.copy(img)
        merged_boxes = self.window_history.getWindows(image)
        boxed_image = draw_boxes(image, merged_boxes)

        # image = np.copy(img)
        # for window in windows:
        #    cv2.rectangle(image, window[0], window[1], (0, 255, 255), 6)
        return boxed_image

    def process_video(self, inputVid, outputVid):
        clip1 = VideoFileClip(inputVid)
        laneClip = clip1.fl_image(self.process_image)
        laneClip.write_videofile(outputVid, audio=False)


if __name__ == "__main__":

    params_model_file = "params_model.p"
    ystart = 350
    ystop = 650
    scales = [1.5]

    processing_video = False
    if processing_video:
        cf = CarFinder(ystart, ystop, scales, params_model_file, histSize=10)
        input_video_file = "../project_video.mp4"
        output_video_file = "../project_video_output.mp4"
        cf.process_video(input_video_file, output_video_file)
    else:
        cf = CarFinder(ystart, ystop, scales, params_model_file, histSize=1)
        input_image_file = "../test_images/test4.jpg"
        img = mpimg.imread(input_image_file)
        boxed_image = cf.process_image(img)
        plt.imshow(boxed_image)
        plt.show()
        cv2.imwrite("../output_images/test4_cars_detected.jpg", img)
