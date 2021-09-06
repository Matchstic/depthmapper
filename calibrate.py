#!/bin/python3

import os
from pathlib import Path
import cv2

from lib.calibration import Calibration

LOAD_DIR = str(Path.home()) + '/.stereo_calibration/'
WIDTH = 640
HEIGHT = 480

def main():
    calibrator = Calibration(WIDTH, HEIGHT, LOAD_DIR)
    input('Hit any key when you have the chessboard prepared')

    # Wait for user input, then start calibration
    # Show preview whilst waiting?

    #calibrator.capture_images()
    calibrator.compute_calibration(True)

if __name__ == '__main__':
    main()