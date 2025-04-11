import cv2
import numpy as np
import easygui

# Open a file dialog to select an image
file_path = easygui.fileopenbox()

# Read the image
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Define the edge detection kernel (Sobel operator)
kernel1= np.array([[-1, 0, 1],
                   [-2,  0, 2],
                   [-1, 0, 1]])
kernel2=np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])

# Resize the image to 512x512
image = cv2.resize(image, (28,28))
# Apply the convolution
convolved_image = cv2.filter2D(image, -1, kernel1)
convolved_image2 = cv2.filter2D(image, -1, kernel2)
# Display the original and convolved images
cv2.imshow('Original Image', image)
cv2.imshow('Convolved Image1', convolved_image)
cv2.imshow('Convolved Image2', convolved_image2)
# Combine the convolved images
combined_image = cv2.addWeighted(convolved_image, 0.5, convolved_image2, 0.5, 0)

# Display the combined image
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()