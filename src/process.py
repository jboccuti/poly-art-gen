import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from skimage.draw import polygon
from PIL import ImageTk, Image


def process_image(pil_image, lower_thresh, upper_thresh, auto_thresh):

    cv_image = pil_to_cv(pil_image)

    # Source Image (color)
    src = cv_image
    # Source Image (grayscale)
    src_gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

    # Get the size of the source image
    src_height, src_width = src.shape[:2]

    coordinates = canny_edge_detection(
        src, src_height, src_width, src_gray, lower_thresh, upper_thresh, auto_thresh
    )
    triangulated_image = delaunay_triangulation(src, src_height, src_width, coordinates)

    return cv_to_pil(triangulated_image)


def canny_edge_detection(
    src, src_height, src_width, src_gray, lower_thresh, upper_thresh, auto_thresh
):

    # Automatic Canny Setup (https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
    sigma = 0.33
    v = np.median(src_gray)
    if auto_thresh:
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    else:
        lower = lower_thresh
        upper = upper_thresh

    # CV Canny Filter
    cv_canny_img = cv.Canny(src_gray, lower, upper)

    # Coordinates from CV Canny
    indices = np.where(cv_canny_img != [0])
    coordinates = np.array([list(a) for a in zip(indices[0], indices[1])])

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

    return full_coordinates


def delaunay_triangulation(src, src_height, src_width, coordinates):

    # Create the Delaunay Triangulation
    tri = Delaunay(coordinates)

    # Create a blank image
    output = np.zeros((src_height, src_width, 3), dtype=np.uint8)

    # Calculate the color and draw the triangles on the image
    for flipped_triangle in coordinates[tri.simplices]:

        # Flip the y,x to x,y so it's oriented correctly
        triangle = np.flip(flipped_triangle)

        tri_poly = src[
            polygon(flipped_triangle[:, 0], flipped_triangle[:, 1], src.shape)
        ]
        tri_color = np.mean(tri_poly, axis=0)
        color_ints = tri_color.astype(int)

        output = cv.fillPoly(output, [triangle], color_ints.tolist())
        # output = cv.polylines(output, [triangle], True, (31, 32, 35))
        # cv.drawContours(output, [triangle], -1, color_ints.tolist(), thickness=cv.FILLED)

    return output


def pil_to_cv(pil_image):

    cv_image = cv.cvtColor(np.asarray(pil_image), cv.COLOR_RGB2BGR)

    return cv_image


def cv_to_pil(cv_image):

    pil_image = Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))

    return pil_image
