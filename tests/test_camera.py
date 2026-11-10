import numpy as np
import pytest
import camera


class Capture:
    def __init__(self, opened, frame=None):
        self.opened, self.frame, self.released = opened, frame, False
    def isOpened(self):
        return self.opened
    def read(self):
        return self.frame is not None, self.frame
    def release(self):
        self.released = True


def test_falls_back_and_returns_first_real_frame(monkeypatch):
    frame = np.zeros((3,4,3),dtype=np.uint8)
    candidates = [Capture(False),Capture(True),Capture(True,frame)]
    calls=[]
    def create(index, backend):
        calls.append((index,backend))
        return candidates[len(calls)-1]
    monkeypatch.setattr(camera.cv2,'VideoCapture',create)
    cap, actual = camera.open_camera(2)
    assert len(calls)==3 and all(i==2 for i,b in calls)
    assert candidates[0].released and candidates[1].released
    assert cap is candidates[2] and not cap.released
    assert actual is frame


def test_all_failures_release_devices_and_explain_next_step(monkeypatch):
    candidates=[]
    def create(*args):
        candidate=Capture(False); candidates.append(candidate); return candidate
    monkeypatch.setattr(camera.cv2,'VideoCapture',create)
    with pytest.raises(RuntimeError,match='launch_camera.cmd'):
        camera.open_camera()
    assert len(candidates)==3 and all(c.released for c in candidates)
