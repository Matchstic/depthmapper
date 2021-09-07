import time
import cv2
import os
import numpy as np

from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

from lib.helpers import open_capture

# Handles creation, saving and loading of calibration state for
# stereo cameras.
class Calibration:
    def __init__(self, config, load_directory, capture_directory = '/tmp/stereo/'):
        self.width = config['general']['width']
        self.height = config['general']['height']

        self.chessboard_rows = config['calibration']['chessboard_rows']
        self.chessboard_cols = config['calibration']['chessboard_cols']
        self.chessboard_size = config['calibration']['chessboard_size']

        self.left_camera_id = config['general']['left_camera_id']
        self.right_camera_id = config['general']['right_camera_id']

        self.capture_directory = capture_directory
        self.load_directory = load_directory

        if os.path.exists(self.load_directory) == False or len(os.listdir(self.load_directory)) == 0:
            self.active_calibration = None
        else:
            self.active_calibration = self.load_calibration()

        if not os.path.exists(self.capture_directory):
            os.mkdir(self.capture_directory)
            os.mkdir(self.capture_directory + 'left')
            os.mkdir(self.capture_directory + 'right')

    def has_calibration(self):
        return self.active_calibration != None

    def capture_images(self):
        print('Calibration :: Capturing frames...')

        leftCapture = open_capture(self.left_camera_id, self.width, self.height)
        rightCapture = open_capture(self.right_camera_id, self.width, self.height)

        print('Waiting 5s...')
        time.sleep(5)

        for i in range (0, 30, 1):
            print('Waiting 1s...')
            time.sleep(1)

            _, frame1 = leftCapture.read()
            _, frame2 = rightCapture.read()

            cv2.imwrite(self.capture_directory + 'left/img' + str(i) + '.png', frame1)
            cv2.imwrite(self.capture_directory + 'right/img' + str(i) + '.png', frame2)

            print('Frame ' + str(i) + ' done')

        print('Calibration :: Captured.')

    def compute_calibration(self, show_results = False):
        print('Calibration :: Computing...')

        calibrator = StereoCalibrator( self.chessboard_rows,  self.chessboard_cols,  self.chessboard_size, (self.width, self.height))

        for i in range (0, 30, 1):
            if not os.path.exists(self.capture_directory + 'left/img' + str(i) + '.png'):
                continue
            if not os.path.exists(self.capture_directory + 'right/img' + str(i) + '.png'):
                continue

            left = cv2.imread(self.capture_directory + 'left/img' + str(i) + '.png', 1)
            right = cv2.imread(self.capture_directory + 'right/img' + str(i) + '.png', 1)

            try:
                calibrator._get_corners(left)
                calibrator._get_corners(right)
            except ChessboardNotFoundError as error:
                print(str(i) + ': ' + str(error))
            else:
                calibrator.add_corners((left, right))

        self.active_calibration = calibrator.calibrate_cameras()

        if not os.path.exists(self.load_directory):
            os.mkdir(self.load_directory)

        self.active_calibration.export(self.load_directory)

        print('Calibration :: Done')
        print('Calibration :: Exported to ' + self.load_directory)

        if show_results:
            leftCapture = open_capture(self.left_camera_id, self.width, self.height)
            rightCapture = open_capture(self.right_camera_id, self.width, self.height)

            cv2.namedWindow('Undistorted')

            print('Hit `q` or `CTRL+C` to exit preview')

            while True:
                _, left_frame = leftCapture.read()
                _, right_frame = rightCapture.read()

                left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                rectified_pair = self.rectify(left_gray_frame, right_gray_frame)

                cv2.imshow('Undistorted', np.hstack((rectified_pair[0], rectified_pair[1])))

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

    def load_calibration(self):
       return StereoCalibration(input_folder=self.load_directory)

    def rectify(self, left_frame, right_frame):
        return self.active_calibration.rectify((left_frame, right_frame))