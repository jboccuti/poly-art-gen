import os
import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from skimage.draw import polygon

filename = os.path.join(os.path.dirname(__file__), "heic1501a-sm.jpg")
# filename = os.path.join(os.path.dirname(__file__), "heic1015a.jpg")
# filename = os.path.join(os.path.dirname(__file__), "heic1310a.jpg")
# filename = os.path.join(os.path.dirname(__file__), "heic0515a-sm.jpg")

# Source Image (color)
src = cv.imread(filename)
# Source Image (grayscale)
src_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

# Automatic Canny Setup (https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
sigma = 0.33
v = np.median(src_gray)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

# CV Canny Filter
# cv_canny_img = cv.Canny(src_gray, lower, upper)
cv_canny_img = cv.Canny(src_gray, 255, 255)

# Show the output
cv.imshow("CV Canny", cv_canny_img)

# Coordinates from CV Canny
indices = np.where(cv_canny_img != [0])
coordinates = np.array([list(a) for a in zip(indices[0], indices[1])])

# Get the size of the source image
src_height, src_width = src.shape[:2]
# Divide the edges into equally spaced points
w_points = int(src_width / 100)
h_points = int(src_height / 100)
w_delta = int(src_width / w_points)
h_delta = int(src_height / h_points)
# Generate the points around the border
border_points = np.array(
    [[0, 0], [src_width, 0], [0, src_height], [src_width, src_height]]
    + [[w_delta * i, 0] for i in range(1, w_points)]
    + [[w_delta * i, src_height] for i in range(1, w_points)]
    + [[0, h_delta * i] for i in range(1, h_points)]
    + [[src_width, h_delta * i] for i in range(1, h_points)]
)

# Full input of points for the Delaunay
full_coordinates = np.concatenate([coordinates, np.flip(border_points)])
# full_coordinates = coordinates

# Create the Delaunay Triangulation
tri = Delaunay(full_coordinates)

# Create a blank image
output = np.zeros((src_height, src_width, 3), dtype=np.uint8)

# Calculate the color and draw the triangles on the image
for flipped_triangle in full_coordinates[tri.simplices]:

    # Flip the y,x to x,y so it's oriented correctly
    triangle = np.flip(flipped_triangle)

    tri_poly = src[polygon(flipped_triangle[:, 0], flipped_triangle[:, 1], src.shape)]
    tri_color = np.mean(tri_poly, axis=0)
    color_ints = tri_color.astype(int)

    output = cv.fillPoly(output, [triangle], color_ints.tolist())
    # output = cv.polylines(output, [triangle], True, (31, 32, 35))
    # cv.drawContours(output, [triangle], -1, color_ints.tolist(), thickness=cv.FILLED)


cv.imshow("Delaunay", output)
cv.imwrite(os.path.join(os.path.dirname(__file__), "output.jpg"), output)

# Exiting the window if 'q' is pressed on the keyboard.
if cv.waitKey(0) & 0xFF == ord("q"):
    cv.destroyAllWindows()
