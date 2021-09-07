#!/bin/python3

from pathlib import Path

from lib.calibration import Calibration
import configparser

LOAD_DIR = str(Path.home()) + '/.stereo_calibration/'

def main():
    config = configparser.ConfigParser()
    config.read('settings.conf')

    calibrator = Calibration(config, LOAD_DIR)
    input('Hit any key when you have the chessboard prepared')

    calibrator.compute_calibration(True)

if __name__ == '__main__':
    main()