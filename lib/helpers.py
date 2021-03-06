import cv2
from lib.camera import Camera

def gstreamer_pipeline(
    id,
    config,
    capture_width=1640,
    capture_height=1232,
    sensor_mode=3,
    flip_method=0,
):
    framerate = config['general']['framerate']
    display_width = config['general']['width']
    display_height = config['general']['height']

    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def open_capture(id, width, height):
    stream = gstreamer_pipeline(id, width, height)
    capture = Camera(stream, id)

    if not capture.isOpened():
        raise Exception('Could not open video device ' + str(id))

    return capture