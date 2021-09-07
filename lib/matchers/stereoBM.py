import cv2

class StereoBMMatcher:
    def __init__(self, config):
        self.width = config['general']['width']
        self.height = config['general']['height']

        downsample = config['general']['downsample_factor']

        self.resize = (int(self.width / downsample), int(self.height / downsample))

        # Apply configuration settings
        self.sbm = cv2.StereoBM_create(
            numDisparities=config['stereobm']['num_disparities'],
            blockSize=config['stereobm']['block_size']
        )
        self.sbm.setPreFilterType(1)
        self.sbm.setMinDisparity(config['stereobm']['min_disparity'])
        self.sbm.setNumDisparities(config['stereobm']['num_disparities'])
        self.sbm.setTextureThreshold(config['stereobm']['texture_threshold'])
        self.sbm.setUniquenessRatio(config['stereobm']['uniqueness_ratio'])
        self.sbm.setSpeckleRange(config['stereobm']['speckle_range'])
        self.sbm.setSpeckleWindowSize(config['stereobm']['speckle_window'])

    def process_pair(self, rectified_pair):
        left = cv2.resize(rectified_pair[0], self.resize)
        right = cv2.resize(rectified_pair[1], self.resize)

        disparity = self.sbm.compute(left, right)

        disparity = cv2.dilate(disparity, None, iterations=1)

        disparity = cv2.resize(disparity, (self.width, self.height))

        return disparity