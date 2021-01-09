import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "L6_CameraCalibration/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('L6_CameraCalibration/test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
  # Pass in your image into this function
  # Write code to do the following steps

  ## 1) Undistort using mtx and dist
  undist = cv2.undistort(img, mtx, dist, None, mtx)

  ## 2) Convert to grayscale
  gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

  ## 3) Find the chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)  

  ## 4) If corners found: 
  if ret == True:
    # a) draw corners
    cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    
    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    src = np.float32([corners[0], corners[nx-1], corners[nx*ny-1], corners[nx*(ny-1)]])
    # src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx])

    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    offset = 100
    dst = np.float32([[offset,offset],[gray.shape[1]-offset,offset],[gray.shape[1]-offset,gray.shape[0]-offset],[offset,gray.shape[0]-offset]])
    # width = 1100
    # height = 825
    # dst = np.float32([[100,100],[100+width,100],[100+width,100+height],[100,100+height]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
