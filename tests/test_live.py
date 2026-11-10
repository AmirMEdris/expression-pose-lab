from types import SimpleNamespace
import sys
import numpy as np
import pytest
import live


class Capture:
    def __init__(self): self.released=False
    def read(self): return True,np.full((48,64,3),20,dtype=np.uint8)
    def release(self): self.released=True


def test_original_defaults_and_keys():
    c=live.Controls()
    assert not c.predicting and not c.pose and c.help and c.fps
    for key,field in [('p','predicting'),('e','pose'),('i','help'),('f','fps')]:
        before=getattr(c,field)
        assert c.key(ord(key))==field and getattr(c,field)!=before
        c.key(ord(key.upper()))
        assert getattr(c,field)==before
    assert c.key(ord('a'))=='switch' and c.key(32)=='capture' and c.key(27)=='quit'


def test_screen_region_and_rgb_conversion(monkeypatch):
    calls=[]
    def grab(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(convert=lambda mode:np.array([[[255,0,0]]],dtype=np.uint8))
    monkeypatch.setitem(sys.modules,'PIL',SimpleNamespace(ImageGrab=SimpleNamespace(grab=grab)))
    assert live.grab_screen([10,20,30,40]).tolist()==[[[0,0,255]]]
    assert calls==[{'bbox':(10,20,40,60),'all_screens':False}]


def test_source_switch_releases_camera_and_reopens_it():
    cameras=[]
    def opener(index):
        cap=Capture();cameras.append(cap);return cap,cap.read()[1]
    source=live.LiveSource(camera_opener=opener,screen_grabber=lambda region:np.ones((5,7,3),dtype=np.uint8))
    source.switch('camera'); assert source.read().shape==(48,64,3)
    source.switch('screen'); assert cameras[0].released and source.read().shape==(5,7,3)
    source.switch('camera'); assert len(cameras)==2
    source.close(); assert cameras[1].released


def test_failed_switch_preserves_working_source():
    cap=Capture()
    def broken(region): raise OSError('screen unavailable')
    source=live.LiveSource(camera_opener=lambda i:(cap,cap.read()[1]),screen_grabber=broken)
    source.switch('camera')
    with pytest.raises(OSError): source.switch('screen')
    assert source.mode=='camera' and not cap.released
    source.close()


def test_live_loop_toggles_switches_saves_and_resets_pose(monkeypatch,tmp_path):
    cameras=[]; predictions=[]; poses=[]; windows=[]
    def opener(index):
        cap=Capture();cameras.append(cap);return cap,cap.read()[1]
    original_source=live.LiveSource
    monkeypatch.setattr(live,'LiveSource',lambda *args:original_source(camera_opener=opener,screen_grabber=lambda r:np.full((80,100,3),100,dtype=np.uint8)))
    class Expression:
        def __init__(self,*args): pass
        def detect(self,frame,predict=True):
            predictions.append(predict)
            return [dict(box=[1,2,10,12],**({'label':'happy'} if predict else {}))]
    class Pose:
        def __init__(self,*args,**kwargs): self.closed=False;self.calls=[];poses.append(self)
        def detect(self,frame,timestamp): self.calls.append(timestamp);return []
        def close(self): self.closed=True
    monkeypatch.setitem(sys.modules,'expression',SimpleNamespace(ExpressionEstimator=Expression))
    monkeypatch.setitem(sys.modules,'pose',SimpleNamespace(PoseEstimator=Pose,draw_poses=lambda *args:None))
    for name in ('namedWindow','resizeWindow','destroyAllWindows'):
        monkeypatch.setattr(live.cv2,name,lambda *args:None)
    monkeypatch.setattr(live.cv2,'imshow',lambda name,frame:windows.append((name,frame.copy())))
    monkeypatch.setattr(live.cv2,'getWindowProperty',lambda *args:1)
    keys=iter([ord('p'),ord('e'),ord('a'),ord('i'),ord('f'),32,ord('a'),ord('e'),27])
    monkeypatch.setattr(live.cv2,'waitKey',lambda *args:next(keys))
    args=SimpleNamespace(pose_only=False,camera=0,screen=False,screen_region=None,expression_threshold=.6,
        pose_model='lite',max_people=4,output=None,max_frames=0,landmark_threshold=.5,show_partial_poses=False,capture_dir=tmp_path)
    live.run_live(args,tmp_path)
    assert predictions==[False]+[True]*8
    assert len(cameras)==2 and all(c.released for c in cameras)
    assert len(poses)==3 and all(p.closed for p in poses)
    assert all(a<b for p in poses for a,b in zip(p.calls,p.calls[1:]))
    files=list(tmp_path.glob('face-*.png')); assert len(files)==1
    crop=live.cv2.imread(str(files[0])); assert crop.shape==(12,10,3) and (crop==100).all()
    assert any(name.startswith('Captured face') for name,frame in windows)
    assert all(frame.shape==(720,960,3) for name,frame in windows if name==live.WINDOW)
