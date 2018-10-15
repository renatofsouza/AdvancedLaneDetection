
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.jpg "Chess board Undistorted"
[image2]: ./camera_cal/calibration1.jpg "Chess board distorted"
[image3]: ./test_images/test5.jpg "Dash cam distorted image"
[image4]: ./output_images/straight_lines1_undistorted.jpg "Dash cam undistorted image"
[image5]: ./output_images/straight_lines1_binary.jpg "Dash cam binary image"
[image6]: ./output_images/straight_lines1_persp_pts.jpg "Dash cam with perpective transform points"
[image7]: ./output_images/straight_lines1_warp.jpg "perpective transformed image (warped)"
[image8]: ./output_images/straight_lines1_warp_binary.jpg "warped binary (perpective transformed) image "
[image9]: ./output_images/histogram.jpg "Histogram showing the peaks on lane pixels "
[image10]: ./output_images/straight_lines1_sliding_window.jpg " Image containing sliding window " 
[image11]: ./output_images/straight_lines1_final_image.jpg " Final imagae " 


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the method `calibrate(images_folder, shape = (0,0), draw_images=False):`  of the file called `CameraCalibration.py`).  

The calibration process consists of taking multiple pictures of a known object and comparing coordinate points of the real object with coordinate points of the image. In this process I used a set of chessborad images taken from diffent angles and distances.   

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. Below you can see both the distorted image and the obtained undistorted output: 

![Original Image][image2]         ![Distorted Image][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The images below shows the resut of `cv2.undistort()` function to one of the dash camera pictures. The effect can be mostly seen in the botton corners of the image (car hood shape).  

![Original Image][image3]         ![Distorted Image][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The method that combines are the thresholds is `find_edges(image):` in the file  `ImageAnalyzer.py`.  Here's an example of my output for this step. 

![Binary Image][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img, src_pts, dst_pts)`, which appears in lines 112 through 117 in the file `ImageAnalyzer.py`  The `perspective_transform(img, src_pts, dst_pts)` function takes as inputs an image (`img`), as well as source (`src_pts`) and destination (`dst_pts`) points.  The source and destination points are hardcoded for now. The values where choosen by observing sample images. 

```python
# Set perspective transform source and destination points 
pt1 = [190,img.shape[0]]
pt2 = [575,462]
pt3 = [706,462]
pt4 = [1120,img.shape[0]]
src_pts = np.float32([pt1, pt2, pt3, pt4]) 

pt5 = [200,img.shape[0]]
pt6 = [200,0]
pt7 = [1000,0]
pt8 = [1000,img.shape[0]]
dst_pts = np.float32([pt5, pt6, pt7, pt8])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 200, 720      | 
| 575, 462      | 200, 0        |
| 706, 462      | 1000, 0       |
| 1120, 720     | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective Transform Points][image6]     ![Warped Image (birds eye) ][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I executed the following steps: 

(a) Executed a perpective transform (birds eye view) to the binary image. This is performed by the 
function  `perspective_transform(img, src_pts, dst_pts)` in the file `LaneTracker.py`
![Warped Image (birds eye) ][image8]

(b) Used a histogram function to plot the density of pixels in the half botton of the image.
Then I found the peaks of the left and right halves and considered them to be the lane marks. This code is performed by the function `find_lane_pixels(binary_warped)` in the file `LaneTracker.py`
![Histogram][image9]

(c) Under the same `find_lane_pixels(binary_warped)` function, I use a sliding window to detect the pixels. 
First I define my window size. 
Then I start the first window from the coordinates that matches the peaks indentified by the histogram. 
For each window, I find the x and y coordinates of each pixel. 
Then I start the next window based on the average density of the pixels in the previous windows. 
![Sliding Window][image10]

(d) Based on the x and y coordinates returned by the `find_lane_pixels(binary_warped)` function, I fit a polynomial. This is executed by the function `fit_polynomial(binary_warped)` in the file `LaneTracker.py`.

(e) Once the lane is found using the mmethod described above, the next frame I use a more efficient  search algorithm. I pass a fit and search for pixels in the new frame around this fit. The fit I use is the average of the last 5 good fits. If the algorithm gets lost, then I fall back to the previous search mechanism.  

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the curvature and the position of the vehicle is performed by the function `measure_curvature_and_center_distance(binary_warped, ploty, left_fit, right_fit)` in the file `LaneTracker.py`. 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 2 functions. The first one is `draw_lane(warped, left_fit, right_fit, ploty, src_pts, dst_pts, img_original )` in `LaneTracker.py`. This function draws the lane on the original image. The second one is ` draw_data(original_img, curv_rad, center_dist)`, also in `LaneTracker.py`. This second function draws the text that correspond to the curvature and the center lane distance. Here is an example of my result on a test image:

![Final Image Result][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The algothm worked reliabily on the video project. There were a couple of areas that the lane detection was not very smooth from one frame to the other. To solve that, I am using an average of the previous 5 fits. That helps the stability of the lane detection. 

By processing the challenge videos, there are at least a couple of situations that I could verify the current implementation will fail. 

The first one is when cracks are present in the road. I have not yet done a deep analysis, but I believe the histogram may be catching peaks where the cracks are. One possible way to improve this is to compare the x distance between the left and right histogram peaks agains the expected lane width. If the numbers are not simliar I can search for lower peaks in the histogram and compare distances again. 

Another idea is to improve the detection of the binary image to filter out cracks perhaps based on color threshold.

I can also improve the sanity check of the lane lines. For example, I can check if they are more or less parallel and discard situations where they may cross near the car. 

The second problem I detected is when there aresharp or multiple turns, one after the other. 

In order to improve this situation, I can try to fit a different function. That will help to catch changes of directions in the line.  

I can also try to play with the size of the sliding windows to be more effective in sharp turns.  
