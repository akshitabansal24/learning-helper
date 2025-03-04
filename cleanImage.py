import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
from google.cloud import storage
import bucket
import time

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformFourPoints(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0], [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def cleanImage(path):
    image = cv2.imread(path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blurred, 75, 200)

    print("STEP 1: Edge Detection Completed")

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort by area and filter out too small contours
    valid_contours = [c for c in cnts if cv2.contourArea(c) > 1000]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    screenCnt = None
    for c in valid_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Ensure it's a 4-sided shape with realistic dimensions
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2:  # Accept reasonable document-like aspect ratios
                screenCnt = approx
                break

    if screenCnt is None:
        print("Warning: No valid document contour detected. Using full image.")

        # Step 2: Adaptive Thresholding on the original image
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        T = threshold_local(gray, 11, offset=10, method="gaussian")
        cleaned = (gray > T).astype("uint8") * 255
    else:
        print("STEP 2: Found valid document contours")
        cleaned = transformFourPoints(orig, screenCnt.reshape(4, 2) * ratio)
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        T = threshold_local(cleaned, 11, offset=10, method="gaussian")
        cleaned = (cleaned > T).astype("uint8") * 255

    print("STEP 3: Image Cleaning Completed")

    cleanedFile = imutils.resize(cleaned, height=650)
    timestamp = int(time.time())
    tmp_cleanedFile = f"/tmp/cleanedFile_{timestamp}.png"
    cv2.imwrite(tmp_cleanedFile, cleanedFile)
    bucket.gcs_upload_image(tmp_cleanedFile)

    return tmp_cleanedFile
