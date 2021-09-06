import cv2
import numpy as np

# Default preset
minDisparity = 0
numDisparities = 32
blockSize = 3
P1 = 8 * 1 * blockSize * blockSize
P2 = 32 * 1 * blockSize * blockSize
disp12MaxDiff = 0
preFilterCap = 0
uniquenessRatio = 1
speckleWindowSize = 10
speckleRange = 1

class StereoSGBMMatcher:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.resize = (int(self.width / 2), int(self.height / 2))

        # Apply configuration settings
        self.sbm = cv2.StereoSGBM_create(
            minDisparity,
            numDisparities,
            blockSize,
            P1,
            P2,
            disp12MaxDiff,
            preFilterCap,
            uniquenessRatio,
            speckleWindowSize,
            speckleRange)

    def process_pair(self, rectified_pair):
        left = cv2.resize(rectified_pair[0], self.resize)
        right = cv2.resize(rectified_pair[1], self.resize)

        disparity = self.sbm.compute(left, right)

        disparity = cv2.erode(disparity, None, iterations=1)
        disparity = cv2.dilate(disparity, None, iterations=1)

        disparity = cv2.resize(disparity, (self.width, self.height))

        return disparity