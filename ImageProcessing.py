import numpy as np
import cv2
import scipy.spatial.distance
import math
import matplotlib.pyplot as plt


# UTILS
def resize(image, factor):
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor))


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# CONNECTED COMPONTENS MANIPULATION
def remove_small_components(components, min_size=1000):
    # Unpacking the values in the components object
    nb_components, output, stats, centroids = components
    # Removing the background (1st component) from the components computation
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # Drawing the connected components
    counter = 0
    result_image = np.zeros(output.shape)
    for i in range(nb_components):
        # Not drawing the components smaller than the selected size
        if sizes[i] >= min_size:
            counter += 1
            result_image[output == i + 1] = 255

    # print("Original components: ", nb_components, " , after filtering: ", counter)
    return result_image.astype(np.uint8)


def approximate_contours(contours):
    approximations = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        print("Found vertexes: ", approximation.shape[0])
        approximations.append(approximation)
    return approximations


def rectify_contours(contours):
    rectified_contours = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        """epsilon = 0.1 * cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, epsilon, True)
        if poly.shape[0] == 4:
            rectified_contours.append(poly)
        """
        epsilon = 0.03 * cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, epsilon, True)
        if poly.shape[0] >= 4:
            rectified_contours.append(poly)
    return rectified_contours


def compute_bounding_boxes(contours):
    boxes = []
    for contour in contours:
        boxes.append(cv2.boundingRect(contour))
    return boxes


def draw_bounding_boxes(image, boxes, colour, thickness):
    for box in boxes:
        x, y, w, h = box
        image = cv2.rectangle(image, (x, y), (x + w, y + h), colour, thickness)
    return image


def sort_corners(corners):

    sorted_corners = np.zeros(corners.shape)
    sorted_by_y = corners[corners[:, 1].argsort()]
    sorted_corners[:2] = sorted_by_y[sorted_by_y[:2, 0].argsort()]
    sorted_corners[2:] = sorted_by_y[sorted_by_y[2:, 0].argsort() + 2]

    return sorted_corners


def project_image(image, corners, title="projected"):

    corners = sort_corners(corners)

    (rows, cols, _) = image.shape

    # image center
    u0 = cols / 2.0
    v0 = rows / 2.0

    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(corners[0], corners[1])
    w2 = scipy.spatial.distance.euclidean(corners[2], corners[3])

    h1 = scipy.spatial.distance.euclidean(corners[0], corners[2])
    h2 = scipy.spatial.distance.euclidean(corners[1], corners[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((corners[0][0], corners[0][1], 1)).astype('float32')
    m2 = np.array((corners[1][0], corners[1][1], 1)).astype('float32')
    m3 = np.array((corners[2][0], corners[2][1], 1)).astype('float32')
    m4 = np.array((corners[3][0], corners[3][1], 1)).astype('float32')

    # calculate the focal distance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
            n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        if ar_real == 0:
            ar_real = 1
        W = int(w)
        H = int(W / ar_real)
    else:
        if math.isinf(ar_real) or math.isnan(ar_real):
            ar_real = 1
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(corners).astype('float32')
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

    # project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(image, M, (W, H))

    cv2.imshow(title, dst)

    cv2.waitKey(1)
    return dst


def linear_filter(image, contrast=1.5, brightness=0, threshold=50):
    imgYCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    med_lum = np.median(imgYCC[:, :, 0])

    filtered = image.copy()
    if med_lum < threshold:
        filtered = np.around(contrast*filtered + brightness)

    return filtered.astype(np.uint8)