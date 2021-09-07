import cv2
import sys
import numpy as np
from lib.helpers import open_capture

import time

from lib.matchers.stereoBM import StereoBMMatcher
from lib.matchers.stereoSGBM import StereoSGBMMatcher
from lib.matchers.aanet import AANetMatcher

class StereoCapture:
    def __init__(self, config, calibrator, matcher = 'stereobm'):
        self.config = config
        self.calibrator = calibrator
        self.stopped = False

        self.width = config['general']['width']
        self.height = config['general']['height']

        self.left_camera_id = config['general']['left_camera_id']
        self.right_camera_id = config['general']['right_camera_id']

        self.show_rgb = config['general']['show_rgb_frame']

        print(matcher)

        if matcher == 'stereobm':
            self.matcher = StereoBMMatcher(config)
        elif matcher == 'aanet':
            self.matcher = AANetMatcher(config)
        elif matcher == 'stereosgbm':
            self.matcher = StereoSGBMMatcher(config)
        else:
            print("unknown matcher specified, stopping")
            sys.exit()

    def produce_depth_map(self):
        self.leftCapture = open_capture(self.left_camera_id, self.config)
        self.rightCapture = open_capture(self.right_camera_id, self.config)

        cv2.namedWindow("Depth map")

        ticks = []

        while self.stopped == False:
            start = int(time.time() * 1000.0)
            end = 0

            left_grabbed, left_frame = self.leftCapture.read()
            right_grabbed, right_frame = self.rightCapture.read()

            if left_grabbed and right_grabbed:
                rectified_pair = self.rectify(left_frame, right_frame)
                disparity = self.matcher.process_pair(rectified_pair)

                end = int(time.time() * 1000.0)

                disparity_normal = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                image = np.array(disparity_normal, dtype = np.uint8)
                disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_BONE)

                # Show depth map
                if self.show_rgb:
                    cv2.imshow("Depth map", np.hstack((disparity_color, left_frame)))
                else:
                    cv2.imshow("Depth map", disparity_color)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
            else:
                end = int(time.time() * 1000.0)

            ticks.append(end - start)

        # Log out timings
        minval = min(ticks)
        maxval = max(ticks)
        avgval = np.mean(ticks)

        print('Timings -- min: ' + str(minval) + 'ms, max: ' + str(maxval) + 'ms, mean: ' + str(avgval) + 'ms')

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

