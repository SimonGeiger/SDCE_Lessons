import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
img = mpimg.imread('L8_AdvancedComputerVision/warped-example.jpg')/255

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[np.int(img.shape[0]/2):-1 , :]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values

    # histogram = np.zeros(len(bottom_half[0]))
    # for i in range(len(bottom_half[0])):
    #   histogram[i] = np.sum(bottom_half[:,i])

    histogram = np.sum(bottom_half, axis=0)

    return histogram

# Create histogram of image binary activations
histogram = hist(img)

# Plot the result
f, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(24, 9))
f.tight_layout()

ax1.imshow(img, cmap = 'gray')
ax1.set_title('Original Image', fontsize=20)

ax2.plot(histogram)
ax2.set_title('Column Sums', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
