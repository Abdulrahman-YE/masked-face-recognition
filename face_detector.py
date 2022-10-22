from facenet_pytorch import MTCNN
import torch
import cv2


class FaceDetector(object):
    
    def __init__(self,resize=1, *args, **kwargs):
        self.mtcnn = MTCNN(*args, **kwargs)
        self.resize = resize

    def __call__(self, frame ):
        """تحدد الوجه في الاطار بأستخدام MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frame

            ]

        boxes, probs = self.mtcnn.detect(frame)
        faces = []
        if boxes is None:
            return faces
        for i, frame in enumerate(boxes):
            if frame is None:
                continue
            x = int(min(frame[0], frame[2])) if int(min(frame[0], frame[2])) >= 0 else 0
            w = int(max(frame[0], frame[2]))
            y = int(min(frame[1], frame[3])) if int(min(frame[1], frame[3])) >=0 else 0
            h = int(max(frame[1], frame[3]))

            print(f"(x,w,y,h):({x},{w},{y},{h})")
            faces.append((x, y, w, h))
        return faces


