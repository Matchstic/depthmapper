import cv2

class StereoSGBMMatcher:
    def __init__(self, config):
        self.width = config['general']['width']
        self.height = config['general']['height']

        downsample = config['general']['downsample_factor']

        self.resize = (int(self.width / downsample), int(self.height / downsample))

        blockSize = config['stereosgbm']['block_size']

        # Apply configuration settings
        self.sbm = cv2.StereoSGBM_create(
            config['stereosgbm']['min_disparity'],
            config['stereosgbm']['num_disparities'],
            blockSize,
            config['stereosgbm']['p1_factor'] * blockSize * blockSize,
            config['stereosgbm']['p2_factor'] * blockSize * blockSize,
            config['stereosgbm']['disp_12_max_diff'],
            config['stereosgbm']['pre_filter_cap'],
            config['stereosgbm']['uniqueness_ratio'],
            config['stereosgbm']['speckle_window'],
            config['stereosgbm']['speckle_range'])

    def process_pair(self, rectified_pair):
        left = cv2.resize(rectified_pair[0], self.resize)
        right = cv2.resize(rectified_pair[1], self.resize)

        disparity = self.sbm.compute(left, right)

        disparity = cv2.erode(disparity, None, iterations=1)
        disparity = cv2.dilate(disparity, None, iterations=1)

        disparity = cv2.resize(disparity, (self.width, self.height))

        return disparity