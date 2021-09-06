import cv2
import numpy as np

'''
# / 1
SWS = 21
MDS = 0
NOD = 128
TTH = 10
UR = 1
SR = 15 # 14
SPWS = 1000 # 150'''

'''
# / 1.5
SWS = 21
MDS = 0
NOD = 96
TTH = 0
UR = 0
SR = 0
SPWS = 0'''

# / 2
SWS = 21
MDS = 0
NOD = 48
TTH = 0
UR = 1
SR = 2
SPWS = 2

class StereoBMMatcher:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.resize = (int(self.width / 2), int(self.height / 2))

        # Apply configuration settings
        self.sbm = cv2.StereoBM_create(numDisparities=NOD, blockSize=SWS)
        self.sbm.setPreFilterType(1)
        self.sbm.setMinDisparity(MDS)
        self.sbm.setNumDisparities(NOD)
        self.sbm.setTextureThreshold(TTH)
        self.sbm.setUniquenessRatio(UR)
        self.sbm.setSpeckleRange(SR)
        self.sbm.setSpeckleWindowSize(SPWS)

    def process_pair(self, rectified_pair):
        left = cv2.resize(rectified_pair[0], self.resize)
        right = cv2.resize(rectified_pair[1], self.resize)

        disparity = self.sbm.compute(left, right)

        disparity = cv2.dilate(disparity, None, iterations=1)

        disparity = cv2.resize(disparity, (self.width, self.height))

        return disparity