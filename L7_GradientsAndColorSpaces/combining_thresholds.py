import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


from apply_sobel import abs_sobel_thresh
from magnitude_of_gradient import mag_thresh
from direction_of_gradient import dir_threshold

# Read in an image
image = mpimg.imread('L7_GradientsAndColorSpaces/signs_vehicles_xygrad.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

# Conbine masks
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# set font size of plot figure to 8 pt
plt.rcParams.update({'font.size': 8})

# plot figure in max window size
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# Plot the result
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image')
ax4.imshow(combined, cmap='gray')
ax4.set_title('Combined Image')
ax2.imshow(gradx, cmap='gray')
ax2.set_title('Grad X')
ax3.imshow(grady, cmap='gray')
ax3.set_title('Grad Y')
ax5.imshow(mag_binary, cmap='gray')
ax5.set_title('Grad Magnitude')
ax6.imshow(dir_binary, cmap='gray')
ax6.set_title('Grad Direction')

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

