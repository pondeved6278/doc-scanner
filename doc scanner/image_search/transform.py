from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_pts(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost  = leftMost[np.argsort(leftMost[:, 1]), :]
    (t1, b1) = leftMost

    D = dist.cdist(t1[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([t1, tr, br, b1], dtype = "float32") 

def four_point_transform(image, pts):
    rect = order_pts(pts)
    (t1, tr, br, b1) = rect

    widthA = np.sqrt(((br[0] - b1[0]) ** 2) + ((br[1] - b1[1]) ** 2))
    widthB = np.sqrt(((tr[0] - t1[0]) ** 2) + ((tr[1] - t1[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((t1[0] - b1[0]) ** 2) + ((t1[1] - b1[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth -1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype= "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped