#!/bin/python3

from pathlib import Path

import argparse
import configparser

import signal

from lib.calibration import Calibration
from lib.stereo import StereoCapture

LOAD_DIR = str(Path.home()) + '/.stereo_calibration/'

parser = argparse.ArgumentParser(description='Depth mapping module')
parser.add_argument('-m', '--matcher', default='stereobm',
                    help='Matcher to use. Options: stereobm, stereosgbm, aanet')

capture = None

def signal_handler(sig, frame):
    capture.stop()

def main():
    global capture
    signal.signal(signal.SIGINT, signal_handler)

    config = configparser.ConfigParser()
    config.read('settings.conf')

    calibrator = Calibration(config, LOAD_DIR)
    if not calibrator.has_calibration():
        print('Calibration required!')
        input('run `python3 calibrate.py`')

        return

    print('Processing stereo images')

    args = parser.parse_args()

    capture = StereoCapture(config, calibrator, args.matcher)
    capture.produce_depth_map()

if __name__ == '__main__':
    main()