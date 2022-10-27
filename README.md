# Readme
This readme is an overview of how to run the code and what packages to install. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install opencv-python, numpy and OS. 

```bash
pip install opencv-python
pip install numpy
pip install scikit-image
pip install os
```

## Usage

The file system is cascading meaning there's one initialisation script to execute the rest of the code. All functions are indipendent for each of the two algorithms and can be found in either *Image_Function_V1.py* or *Intersect.py* for the Ray-Tracing algorithm. In order to run the code only *V1_5_Opt1.py* is run. The initialisation parameters can be set accordingly. Remember to also adjust the path according to your directory and according to the data set that shall be used. 


##Detailed Functions

### V1_5.py

This is the initialisation file. All initial parameters are set here. The corner quality level is set as well as all other parameters for the feature detection. The dataset path has to be adjusted acordingly as well as in the arguments for *image_function(args)*. The threshold for the directional change algorithm is set as *threshold_tandir*. The ellipse size is set as *size_ellipse*. The algorithm is chosen by setting *algorithm* to either _tandir_ or _ray-trace_. The ego-motion-compensation is a boolean argument, as well as the *noise_factor*. 

The rest of this script is a loop to evaluate the prediction for different initialisation parameters and can be entirely left out. 

### Img_Function.py - image_function
This file contains the loading and evaluating of the images from the data set. Depending on the initialisation parameters further functions are computed. The output is the confusion matrix of the prediction when using the initialised sccene and parameters.

Since the directional change algorithm is very easily computed it is done so directly in this function using the *algorithm* initialisation parameter. For the Ray-Tracing algorithm the *Intersect.py* is used for its extensive computation.

### Intersect.py
*ellipseoid_distance* is used to test whether passed points are within the bounding box, which has an elliptical shape. The function returns a boolean value. 

*is_escape_point* checks whether passed features have a common vanishing point "behind" them. The function outputs a boolean value accordingly. 

*RayTracing_point_cloud* is the main script for the Ray-Tracing algorithm and compares passed features with one another. Depending on whether an escape point exists the intersecting point is compared to a reference index or added if no match is found within the elliptical bounding box. The function outputs a point cloud that for evaluation reasons had to be reshaped into a boolean logical vector, further increasing the computational load. The function outputs predictions for parsed features.

### Preprocessing
This file contains the Gaussian Blur applied to each image before feature extraction. the function *preprocessing* returns the blurred images. 

All other functions are used for the ego-motion-compensation. *ego_motion_compensation* uses RANSAC in order to apply the 8-Point-Algorithm and outputs an outlier boolean vector as well as the model to further calculate the rotation matrix and translation vector. *point_prediction* uses the previous function to predict the parsed features onto the next image and calculate the actual optical flow. 

The output is equivalent to the feature detection without ego-motion-compensation.

### Score
*calc_score* is the function used to compare the classification from the algorithms with the ground truth. The frame determines the size of the bounding box that is set around the measured feature. This implementation is chosen due to corner detection possibly choosing the outer pixel as viable feature. 

### Other
*Plot_V1_3.py* is used to plot the .csv files created when running the full script. *Precision_F1.txt* contains the top precision and F1-scores of each run executed. The *Gif_maker.py* was used to create the gifs found in the GIFs folder. A simple script that can be useful when saving the annotated feature predictions and wanting to plot them sequentially. 







