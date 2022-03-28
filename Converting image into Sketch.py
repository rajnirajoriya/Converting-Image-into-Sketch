# Importing Modules
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Loading and Plotting Original Image
img = cv2.imread("pic.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
# plt.imshow(img)
# plt.axis("off")
# plt.title("Original Image")
# plt.show()

# Converting Image to GrayScale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 8))
# plt.imshow(img_gray, cmap="gray")
# plt.axis("off")
# plt.title("GrayScale Image")
# plt.show()

# Inverting the Image
img_invert = cv2.bitwise_not(img_gray)
plt.figure(figsize=(8, 8))
# plt.imshow(img_invert, cmap="gray")
# plt.axis("off")
# plt.title("Inverted Image")
# plt.show()

img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
plt.figure(figsize=(8, 8))
# plt.imshow(img_smoothing, cmap="gray")
# plt.axis("off")
# plt.title("Smoothen Image")
# plt.show()

# Converting your image into Pencil Sketch
final = cv2.divide(img_gray, 255 - img_smoothing, scale=255)
plt.figure(figsize=(8, 8))
# plt.imshow(final, cmap="gray")
# plt.axis("off")
# plt.title("Final Sketch Image")
# plt.show()

# The code below display all the images in one frame using the subplots.
plt.figure(figsize=(20, 20))
plt.subplots(1, 5)
plt.imshow(img)
plt.axis("off")
plt.title("OriginalImage")
plt.subplots(1, 5)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplots(1, 5)
plt.imshow(img_invert, cmap="gray")
plt.axis("off")
plt.title("Inverted Image")
plt.subplots(1, 5)
plt.imshow(img_smoothing, cmap="gray")
plt.axis("off")
plt.title("Smoothen Image")
plt.subplots(1, 5)
plt.imshow(final, cmap="gray")
plt.axis("off")
plt.title("Final Sketch Image")
plt.show()
