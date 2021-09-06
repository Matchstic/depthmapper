import cv2
import sys
import numpy as np
from lib.helpers import open_capture

import time

from lib.matchers.stereoBM import StereoBMMatcher
from lib.matchers.stereoSGBM import StereoSGBMMatcher
from lib.matchers.aanet import AANetMatcher

class StereoCapture:
    def __init__(self, width, height, calibrator, matcher = 'stereobm'):
        self.width = width
        self.height = height
        self.calibrator = calibrator
        self.stopped = False

        print(matcher)

        if matcher == 'stereobm':
            self.matcher = StereoBMMatcher(width, height)
        elif matcher == 'aanet':
            self.matcher = AANetMatcher(width, height, 'aanet_sceneflow.pth')
        elif matcher == 'stereosgbm':
            self.matcher = StereoSGBMMatcher(width, height)
        else:
            print("unknown matcher specified, stopping")
            sys.exit()

    def produce_depth_map(self):
        self.leftCapture = open_capture(1, self.width, self.height)    
        self.rightCapture = open_capture(0, self.width, self.height)

        cv2.namedWindow("Depth map")

        ticksize = int(1000.0 / 20.0)

        while self.stopped == False:
            start = int(time.time()*1000.0)
            end = 0

            left_grabbed, left_frame = self.leftCapture.read()
            right_grabbed, right_frame = self.rightCapture.read()

            if left_grabbed and right_grabbed:
                rectified_pair = self.rectify(left_frame, right_frame)
                disparity = self.matcher.process_pair(rectified_pair)

                end = int(time.time()*1000.0)

                disparity_normal = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                image = np.array(disparity_normal, dtype = np.uint8)
                disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
                
                # Show depth map
                # output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
                cv2.imshow("Depth map", np.hstack((disparity_color, left_frame)))
                #cv2.imshow("Depth map", disparity_color)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
            else:
                end = int(time.time()*1000.0)
            
            print('process loop in ' + str(end - start) + 'ms')

            #if (end - start < ticksize):
            #    time.sleep((ticksize - (end - start)) / 1000.0)

    def rectify(self, left_frame, right_frame):
        # Convert to greyscale
        left_grey = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Apply rectification
        return self.calibrator.rectify(left_grey, right_grey)
    
    def stop(self):
        self.leftCapture.stop()
        self.rightCapture.stop()

        self.stopped = True

