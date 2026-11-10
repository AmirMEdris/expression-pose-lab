"""MediaPipe pose estimation; image and tracked video use separate task modes."""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Body connections from the MediaPipe 33-landmark topology.
CONNECTIONS = ((11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
               (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
               (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
               (27,29),(28,30),(29,31),(30,32),(27,31),(28,32))


class PoseEstimator:
    def __init__(self, model_path, video=False, max_people=4):
        self.video = video
        self.last_timestamp = -1
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=max_people)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms=None):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.video:
            if timestamp_ms is None or timestamp_ms <= self.last_timestamp:
                raise ValueError('Video timestamps must strictly increase')
            result = self.detector.detect_for_video(image, timestamp_ms)
            self.last_timestamp = timestamp_ms
        else:
            result = self.detector.detect(image)
        return [[{'x':float(p.x), 'y':float(p.y), 'z':float(p.z),
                  'visibility':float(p.visibility or 0), 'presence':float(p.presence or 0)}
                 for p in person] for person in result.pose_landmarks]

    def close(self):
        self.detector.close()


def draw_poses(frame, people, visibility=0.5):
    height, width = frame.shape[:2]
    for person in people:
        def visible(i):
            p = person[i]
            return p['visibility'] >= visibility and p['presence'] >= visibility and 0 <= p['x'] < 1 and 0 <= p['y'] < 1
        def point(i):
            return (int(person[i]['x'] * width), int(person[i]['y'] * height))
        for a,b in CONNECTIONS:
            if visible(a) and visible(b):
                cv2.line(frame, point(a), point(b), (255,180,0), 2, cv2.LINE_AA)
        for i in range(len(person)):
            if visible(i):
                cv2.circle(frame, point(i), 3, (0,210,255), -1, cv2.LINE_AA)
