## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image_features_plot/acar_plot8667.jpg
[image101]: ./output_images/image_features_plot/acar_plot6977.jpg
[image102]: ./output_images/image_features_plot/noncar_plot8667.jpg
[image103]: ./output_images/image_features_plot/noncar_plot6977.jpg
[image2]: ./output_images/image_features_plot/car_plot202_YCrCb.jpg
[image201]: ./output_images/image_features_plot/car_plot661_RGB.jpg
[image3]: ./output_images/search_draw_win/just_windows_full.jpg
[image301]: ./output_images/search_draw_win/just_windows.jpg
[image4]: ./output_images/search_draw_win/search_draw_win.jpg
[image401]: ./output_images/search_draw_win/search_draw_win_overlap04.jpg
[image5]: ./output_images\hog_subsample_heat/labeled_car_heat_1.5.jpg
[image501]: ./output_images\hog_subsample_heat/labeled_car_heat.jpg
[image7]: ./output_images\hog_subsample_heat/labeled_car_heat_output_bboxes.jpg
[video1]: ./output_project_video_scaled.mp4
[video101]: ./output_project_combined_video_scaled.mp4
[video102]: ./output_project_combined_video_1.5.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The functions necessary for feature extraction is two files, `data_extraction.py` and `test_feature_extract.py`, where the first file has feature extractions and the second loops through the test images to extract the features.  

I started by reading in all the `vehicle` and `non-vehicle` images available under Data-Exploration directory (test files not uploaded due to size).  There are 8792 cars and   8968 non-cars images in the dataset.

Here are couple examples of  the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image101]

Similarly, following are couple of examples of `non-vehicle` images:

![alt text][image102] ![alt text][image103]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Here is another example using the `RGB` color space and keeping rest of the parameters the same:

![alt text][image201]


#### 2. Explain how you settled on your final choice of HOG parameters.

Next I played with the number of `orientations`, trying them in the range from 6 to 9. Finally choosing `9` as it provided better accuracy.

| HOG Parameter       | Values   |
|:-------------:|:-------------:|
| orientations        | 9      |
| pixels_per_cell     | 8      |
| cells_per_block     | 2      |

The `single_img_features` function in `lesson_functions.py` file is utilized to extract features and HOG image from car and non-vehicle images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As part of feature extraction using HOG, I tried to use `RGB`, `LUV`, `Lab` and `YCrCb` for color space value. Using `YCrCb` helped the most to get higher accuracy in SVM detection.

Another important parameter `hog_channel` tried for all available options (0, 1, 2, 3 or ALL), but using `ALL` helped the most as it utilizes all image channels.

Two parameters, `spatial_size` and `hist_bins` also observed to impact on the accuracy. Finally I settled on following values for various parameters:

| Feature Parameter        | Values   |
|:-------------:|:-------------:|
| color_space      | 'YCrCb'        |
| hog_channel      | 'ALL'        |
| spatial_size     | (32, 32) |
| hist_bins        | 32 |

The `extract_features` function in `lesson_functions.py` file is utilized to pass the features to standardize the features. The `test_training_classifier.py` file contains the steps to setup the data for creation and training LinearSVC classifier.

The standardized extracted features are further scaled and transformed (from line #54 to #60 of `test_training_classifier.py` file).

As a trial for training the classifier, only a subset (1000 images) of vehicle and non-vehicle images are utilized. Once the process is established it is executed on complete dataset. The data is split using `train_test_split` module in `sklearn.model_selection` for these .png files.  

With this process and settings I observed accuracy about `98-99%`.

The trained `LinearSVC` classifier is saved to a pickle file `trained_svc_pickle.p` along with various parameters used for feature extraction and to achieve the stated accuracy.

Later on while testing with the project video, I returned to improvise the accuracy as I noticed few false positive detection, hoping to address them by tweaking these parameters. I changed the number of histogram bins to 16, also I tried playing with `C` argument to LinearSVC constructor. Setting the `C`  value to `0.01` helped me to improve the accuracy to `99.49%`. It seems to have helped to avoid few false detections in the video.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As a first attempt for searching features in an input image a sliding window is defined using `64x64` window. Computation for the location of each window corners accommodating the overlapping width equivalent of `50%` of the window size is performed. The Y start value is set to zero to generate windows throughout the image resulting into the following image display all sliding windows obtained.

![alt text][image3]

To reduce searching area on the input image, the `Y` start value is set to half of the image height. Also, the window size is changed to `96x96` and `overlapping` ratio set to `40%` pixels hoping for faster and improved searching. Following image shows reduced overlapping sliding windows.

![alt text][image301]

I used `slide_window` and `search_windows` functions from `lesson_functions.py` to implement the sliding windows functionality and searching extracted features in those windows respectively. The `test_draw_search_window.py` utilizes these functions to generate the sliding and searching windows as displayed earlier.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

The first image below shows result of vehicle feature detection for `40%` overlap for `96x96` pixel window. Though in most cases it boxed vehicle feature it missed in case of `test3.jpg`:

![alt text][image401]

So, I decided to revert back to sliding window size `96x96` pixels with `50%` overlapping which performed better in comparison as it captured all the vehicle features in all of the test images.

![alt text][image4]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The `test_video.py` file contains functions requiring to process video image frames and run through the vehicle detection functions including `find_cars_with_scale_variance` and `car_heatmap_bbox_img`.

Here's a [link to my video result](./output_project_video_scaled.mp4)

As an extension to this current project I combined the `lane detection` code with the `vehicle detection` pipeline in the following video. The `test_combined_video.py` file contains the image processing function to combine the `lane detected` image with `vehicle detection`.

Here's a [link to my video result showing vehicle and lane detection together](./output_project_combined_video_scaled.mp4 )

OR

[link to my video result showing vehicle and lane detection together](./output_project_combined_video_1.5.mp4) 



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As suggested in the project tips, I attempted to implement multi-scaled approach to address the false positive detections. I experimented that approach in `test_hog_subsample_heat.py` with hard-coded scaling values `(1.0, 1.5, 2.0)` and combined the bounding boxes identified to get the heatmap.

As a first step, I used `find_cars` function in `lesson_functions.py` file to get the HOG feature and then apply the scaling to the bottom half-of the image. It returns an image with a rectangles representing the bounding box of the detected vehicles. The `find_cars_with_scale_variance` function in `test_video.py` collects all the bounding boxes for different scale values.  These bounding box represents the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap using `add_heat` function in `lesson_functions.py` file. The heatmaps are added together for different scale values (line #58 of `car_heatmap_bbox_img` function in `test_video.py` file). Then as part of applying threshold I averaged the heatmap data with the count of positive detections (line #60 of `car_heatmap_bbox_img` function in `test_video.py` file) and thresholded that map to identify vehicle positions to drop false detection.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (line #64 of `car_heatmap_bbox_img` function in `test_video.py` file).  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image501]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried to stick with the functions learned in the lessons and mostly experimented with setting different parameters to improvise the detections. For cases where a vehicle is clearly visible when the detection pipeline identifies the feature as a vehicle, it makes sense to validate with a visual confirmation. But it becomes hard to understand why it detects something not a vehicle positively i.e. false positive detections. Though I didn't implement the heatmap approach over multiple frames to drop the false positives, I think that one piece will make the overall pipeline a bit robust.
Also when I reviewed many of the test images, I observed that most of the images were from the back side of the vehicle instead of the side of the vehicle. I suspect that may be contributing to the some of the problems in false detections.    
