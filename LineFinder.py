import Calibration
'''
Steps to find lane
1. Calibrate camera - process calibration images and generate calibration matrix
2. Read image to be processed
3. Use calibration matrix to undistort the image
4. Apply techniques to process image and detect edges and find lanes.
5. Once lanes are found, apply a transform perpective to see image from birds eye view. 
6. Apply techniques to track lanes from birds eye view
7. Paint lanes on original image

'''
images_folder = "camera_cal"
chessShape = (9,6)
camcal = Calibration.CameraCalibraion(images_folder,chessShape)
camcal.calibrate(False)
camcal.test(images_folder +'/calibration19.jpg')
