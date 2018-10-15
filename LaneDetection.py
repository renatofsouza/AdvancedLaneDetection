import LaneTracker
import ImageAnalyzer
import CameraCalibration as camcal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio
from moviepy.editor import VideoFileClip
import cv2



lane_tracker = LaneTracker.LaneTracker()

# Calibrate camera
image_folder = 'camera_cal'
chess_board_shape = (9,6)
ret, mtx, dist, rvecs, tvecs = camcal.calibrate(image_folder, chess_board_shape)
lane_tracker.set_camera_calibration_values(ret, mtx, dist, rvecs, tvecs)

# Geenrate and save an undistorted chessbaord image using the camera calibration parameters for visualization 
# and testing purpouse.
distorted_image_path = image_folder+'/calibration1.jpg'
img_dist = plt.imread(distorted_image_path)  
img_undist = camcal.undistort_image(img_dist, mtx, dist)
file_output = 'output_images/undistorted.jpg'
plt.imsave(file_output,img_undist)

# Geenrate and save an undistorted image from dash camera using the camera calibration 
# parameters for visualization and testing purpouse.
distorted_image_path = 'test_images/straight_lines1.jpg'
img_dist = plt.imread(distorted_image_path)  
img_undist = camcal.undistort_image(img_dist, mtx, dist)
file_output = 'output_images/straight_lines1_undistorted.jpg'
plt.imsave(file_output,img_undist)

# Genarate a binary image after applying thresholds for gradient, color, etc
img_edges = ImageAnalyzer.find_edges(img_undist)
file_output = 'output_images/straight_lines1_binary.jpg'
plt.imsave(file_output,img_edges, cmap='gray')


# Read in an image
img_path = 'test_images/straight_lines1.jpg'
img = mpimg.imread(img_path)

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

lane_tracker.set_perspective_transform_points(src_pts, dst_pts)

# Transform perspective (original image)
img_warp = ImageAnalyzer.perspective_transform(img_undist,lane_tracker.src_pts, lane_tracker.dst_pts)
plt.imsave('output_images/straight_lines1_warp.jpg',img_warp)

# Transform perspective (binary image)
binary_warped = ImageAnalyzer.perspective_transform(img_edges,lane_tracker.src_pts, lane_tracker.dst_pts)
plt.imsave('output_images/straight_lines1_warp_binary.jpg',binary_warped, cmap='gray')


# Draw lines on sample images to test the choosen perspective points
lineThickness = 2
cv2.line(img_undist, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0,255,0), lineThickness)
cv2.line(img_undist, (pt2[0], pt2[1]), (pt3[0], pt3[1]), (0,255,0), lineThickness)
cv2.line(img_undist, (pt3[0], pt3[1]), (pt4[0], pt4[1]), (0,255,0), lineThickness)
cv2.line(img_undist, (pt4[0], pt4[1]), (pt1[0], pt1[1]), (0,255,0), lineThickness)
plt.imsave('output_images/straight_lines1_persp_pts.jpg',img_undist)

# Use a histogram to identify the beginning of the lanes
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)
plt.savefig('output_images/histogram.jpg')

# Find our lane pixels and fit polynomial
img_result, left_fitx, right_fitx, ploty, left_fit, right_fit = LaneTracker.fit_polynomial(binary_warped) 
plt.imsave('output_images/straight_lines1_sliding_window.jpg',img_result)

# Plot the polynomial lines onto the image
plt.gcf().clear()
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.imshow(img_result)
plt.savefig('output_images/straight_lines1_polynomial.jpg')
plt.gcf().clear()

# Final image (process the entire pipeline)
img_out= lane_tracker.process_image(img_undist)
plt.imsave('output_images/straight_lines1_final_image.jpg',img_out)

# process video
video_output1 = 'project_video_output.mp4'
#video_input1 = VideoFileClip('project_video.mp4').subclip(22,26)
video_input1 = VideoFileClip('project_video.mp4')
#video_input1 = VideoFileClip('harder_challenge_video.mp4')
processed_video = video_input1.fl_image(lane_tracker.process_image)
processed_video.write_videofile(video_output1, audio=False)

        