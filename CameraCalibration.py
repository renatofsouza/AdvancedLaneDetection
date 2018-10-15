import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

    
def calibrate(images_folder, shape = (0,0), draw_images=False):

    # Prepare an array with all the object points. Each point has a (x,y,z) coordinate. 
    # The size of the array is the number of inside corners in the chessboard
    # Note, since the chessboard is in a plane, the z coordinate will
    # always be cosidered 0. 
    objp = np.zeros((shape[0] * shape[1],3),np.float32)
    objp[:,:2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1,2)

    # 3d points in real world space. 
    object_points = [] 
    
    # 2d points in image space
    image_points = [] 


    # Make a list of calibration images
    images = glob.glob(images_folder + '/*.jpg')
    if len(images) == 0:
        raise("Folder does not contain any calibration images. Images must be in jpg format.")

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, shape, None)

        # If found, add object points, image points
        if ret == True:
            object_points.append(objp)
            image_points.append(corners)
            img_size = (img.shape[1], img.shape[0]) # TODO: img_size is only used after the loop

            if draw_images:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, shape, corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
                
    cv2.destroyAllWindows()
    # Use the size of last image to calibrate camera given object points and image points  
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points,img_size ,None,None)
    
    return ret, mtx, dist, rvecs, tvecs

def undistort_image(img, mtx, dist):
    # Use cv2.calibrateCamera() and cv2.undistort()
    img_size = (img.shape[0],img.shape[1])
    img_undist = cv2.undistort(img,mtx,dist)

    return img_undist