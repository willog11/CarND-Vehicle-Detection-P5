**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG.png
[image2]: ./output_images/color_hog_features_LUV.png
[image3]: ./output_images/normalized_features_LUV.png
[image4]: ./output_images/first_row_of_search_areas.png
[image5]: ./output_images/second_row_of_search_areas.png
[image6]: ./output_images/third_row_of_search_areas.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The training code can be found classifier_trainer.py The logic of which is very simple:

1. Load in the provided dataset of vehicle and non-vehicle images
2. Perform feature extraction on each image to build up the characteristic dataset of each. (This is covered further below).This feature extraction contains the HOG feature extraction too.
3. Standardize the extracted features to prevent biasing
4. Assign labels and split the features into training and validation data set (80/20 (training/validation) was selected)
5. Create a linear SVM classifier and fit to the training dataset provided
6. Test the resulting classifier on the validation dataset
7. Finally store all parameters and the resulting classifier object for use later

I then explored different color spaces by training the the model on different color spaces and recording the validation result each time following the steps above.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and eventually decided upon a robust parameter set which results in clear HOG features as per below

![alt text][image1]

~~~
orient = 9  # HOG orientations
pix_per_cell = 9 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
~~~

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a final combination of LUV image space and the extraction of spatial binning of colours and a histogram of colours. Finally the HOG was extracted for each channel and added to this. The following image shows a breakdown of each step

![alt text][image2]

Finally the resulting data was normalized using sklearn StandardScaler functionality. This helped reduce biasing towards stronger features, this is very clear to see in the following image:

![alt text][image3]

As stated above the classifier was built multiple times to ensure the best colour space was used. The following table gives the breakdown of results:

| Color Space  | Score   | 
|:-------------:|:-------------:| 
| RGB      | 0.9778      | 
| HSV      | 0.9904      |
| LUV     | 0.9916     |
| HLS      |  0.9893       |
| YUV      |  0.9893       |
| YCrCb      |   0.9916      |

Finally LUV and YCrCb give the same result but in the testing stages of the car detection pipeline it was clear that LUV gave better performance.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to break the image into 3 sections which overlapped:
* First scan the areas close to the ego vehicle where the vehicles will be larger 
* Then move out one row and look for smaller objects. 
* Finally look for smaller vehicles at the horizon point.

Note for all  3 steps HOG sub sampling was used so that only one HOG extraction per each frame was required. By varying the scaling of the search area the following regions could be defined:

![alt text][image4]

![alt text][image5]

![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

ew video!
