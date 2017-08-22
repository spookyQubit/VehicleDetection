### Vehicle detection

The goal of this project is to to use computer vison and machine learning techniques to detect cars in a video stream. The video is taken from a front facing camera mounted on a moving vehicle. 

The steps followed to achieve this goal are:

* Feature Extraction: From a set of training images, we first extract the features needed to build a classifier to distinguish a vehicle from a non-vehicle. 

* Classifier: Gird search is used to find the best classifier. sklearn's GridSearcgCV is used with cv=3. The parameter spaces which we search for include: 1) Linear/RBF SVC, 2) C, and 3) gamma. The best classier is found to be an RBF SVC with C=1 and gamma='auto'. 

* Sliding window: After building a classifer, a sliding window technique is used to feed in image sections (windows) to the classifier. The classifer then predicts whether the section contains a vehicle or not.  

* Heat map: Once all the windows in an image are classified as having a car or not, a voating scheme is uses to filter out possible false positives.

* Video pipeline: The classifier/sliding window pipeline is applied to each image in a video and an output video is created with bounding boxes around segments in the image where vehicles are detected.


[//]: # (Image References)

[image1]: ./output_images/test4_cars_detected.jpg "Vehicle detected image"

---

### Feature Extraction
The code to extract features from a given image can be found in the functions implemented in [featureExtractor.py](#link). The three features which are considered in the project are 
* spatial pixel binning: Pixel intensities of a resized image to (32x32) is used as a feature. 
* histogram of colors: A concatenated version of histograms of all colors is used as a feature.   
* [Histogram of Oriented Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (HOG): Pixels per cell used is 8x8. 2x2 cells per block is used for normalization. I use 9 orientations per block to generate the features.   

---

### Classifier
I used sklearn's SVC library as a classifier to distinguish between vehicles and non-vehicles. The best cross-validation accuracy is achieved for SVC with rbf kernel. Around 15000 samples were trained to build the classifer. Even with this modest number of samples, SVC was slow to train. The impplementation of the classifer can be found in [carClassifier.py](#link). 

---

### Car Finder
In oredr to be able to detect cars, each image is segmented into overlapping windows. Each of these windows is then fed into the classifer to predict whether the window has a vehicle or not. This sliding window technique is implemented in the CarFinder class in [carFinder.py](#link). 

---

### Road Sanity
It is important to remove regions in the image which are incorrectly predicted by the classier to contain a vehicle. Two techniques are used to reduce false positives: 1) information about the history of images seen in the video is kept in class WindowHistory implemented in [roadCache.py](#link), 2) a voting sheme is applied where pixels are assumed to be part of a vehicle only if it gets more votes (a pixel gets a vote when a window containing the pixel is predicted to have a vehicle) than a pre-defined threshold.   

---

### Main image processor
The entry point to the entire pipeline is the process_image function implemented in class CarFinder in [carFinder.py](#link). This is the function to which the VideoFileClip library passes each frame of the video. An example of a vehicle detected image from the final pipeline is shown below:
![alt text][image1]

--- 

###

### Conclusion
The project was a good learning experience where machine learning was used in conjunction with computer vision techniques. The pipeline was correctly able to take in a video file and output a video file with vehicles enclosed in bounding boxes. Although the model performed well in detecting vehicles in the video, there are a number of improvements which can still be made:
1) Only SVCs were considered as model classifiers. Other families of classifers can potentially lead to better performance. 
2) The image processing was prohibitively slow. This made iterative development a bit difficult. The pipeline can potentally be optimized to reduce latency.
3) The model does not use the fact that vehicles in smaller scales are likely to be found further away from the car. This introduces latency and reduces the model performance significantly. 
    
