import cv2
from threading import Thread

class Camera:
    def __init__(self, pipeline, id):
        self.camera = cv2.VideoCapture(pipeline)
        (self.grabbed, self.frame) = self.camera.read()

        self.frame = None
        self.stopped = False

        # Start thread for frame capture
        self.thread = Thread(target=self.thread, name='video ' + str(id), args=())
        self.thread.daemon = True
        self.thread.start()

    def isOpened(self):
        return self.camera.isOpened()

    def stop(self):
        self.stopped = True
        self.thread.join()

    def read(self):
        return (self.grabbed, self.frame)

    def thread(self):
        while self.stopped == False:
            (self.grabbed, self.frame) = self.camera.read()

        self.camera.release()