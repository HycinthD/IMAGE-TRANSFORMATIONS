![Screenshot 2024-10-03 102251](https://github.com/user-attachments/assets/937bab42-9cf7-46a6-9bb1-ccdf54cda332)![Screenshot 2024-10-03 101936](https://github.com/user-attachments/assets/b99076d0-75ee-41ae-b422-4cf45e1d306d)# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
## Step1:
Import the necessary libraries and read the original image and save it as a image variable.

## Step2:
Use cv2.warpAffine() to perform the Translation of the image

## Step3:
Define a scaling matrix to enlarge the image by a factor and apply scaling using cv2.warpAffine().

## Step4:
Use shearing matrices for both x-axis and y-axis transformations and reflection matrices for horizontal and vertical reflections.

## Step5:
Create a rotation matrix to rotate the image by a specified angle, then apply cropping by selecting a region of the image.


## Program:
```python
Developed By:HYCINTH D
Register Number:212223240055
```
## i)Image Translation
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("rapunzel.jpg")

# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the image
plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()
```
## ii) Image Scaling
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("rapunzel.jpg")

# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the image
plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()
```
## iii)Image shearing
```
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rapunzel.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```
## iv)Image Reflection
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rapunzel.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```
## Image Rotation
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rapunzel.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)
```
vi)Image Cropping
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rapunzel.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)
```
## OUTPUT
## i)Image Translation
![Screenshot 2024-10-03 101821](https://github.com/user-attachments/assets/f1d3c172-3340-46d5-9112-e727d0d3092d)

## ii) Image Scaling
![Screenshot 2024-10-03 101936](https://github.com/user-attachments/assets/3394ad83-b00d-4a98-8347-18cd46856369)

## iii)Image shearing
![Screenshot 2024-10-03 102111](https://github.com/user-attachments/assets/64fa6a00-5662-47de-a092-68ffc77b9052)

## iv)Image Reflection
![Screenshot 2024-10-03 102222](https://github.com/user-attachments/assets/9ac5c070-5cce-47c9-bd62-de78b8da13a9)
![Screenshot 2024-10-03 102251](https://github.com/user-attachments/assets/7ffdf829-212a-4ec9-bb5c-a263bc36c916)
## v)Image Rotation
![Screenshot 2024-10-03 102411](https://github.com/user-attachments/assets/c3e066b8-bf2a-45f1-bb9d-03e969718345)

## vi)Image Cropping
![Screenshot 2024-10-03 102616](https://github.com/user-attachments/assets/13188603-5aa7-4bb1-b613-296acc8691ae)


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
