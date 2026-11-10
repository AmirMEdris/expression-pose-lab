"""Original capstone controls over the modern inference pipeline."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import time
import cv2
import numpy as np
from camera import open_camera

WINDOW = 'Expression + Pose Lab - click here for keyboard controls'


@dataclass
class Controls:
    predicting: bool = False
    pose: bool = False
    help: bool = True
    fps: bool = True

    def key(self, code):
        if code in (27, ord('q'), ord('Q')):
            return 'quit'
        if code in (ord('a'),ord('A')):
            return 'switch'
        if code == 32:
            return 'capture'
        for letter, attribute in [('p','predicting'),('e','pose'),('i','help'),('f','fps')]:
            if code in (ord(letter),ord(letter.upper())):
                setattr(self,attribute,not getattr(self,attribute))
                return attribute
        return None


def grab_screen(region=None):
    from PIL import ImageGrab
    bbox = None if region is None else (region[0],region[1],region[0]+region[2],region[1]+region[3])
    rgb = np.asarray(ImageGrab.grab(bbox=bbox, all_screens=False).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LiveSource:
    def __init__(self, camera_index=0, region=None, camera_opener=open_camera, screen_grabber=grab_screen):
        self.index,self.region = camera_index,region
        self.camera_opener,self.screen_grabber = camera_opener,screen_grabber
        self.mode,self.capture,self.buffer = None,None,None

    def switch(self, mode):
        # Validate the replacement before releasing the working source.
        candidate = None
        if mode == 'camera':
            candidate, frame = self.camera_opener(self.index)
        elif mode == 'screen':
            frame = self.screen_grabber(self.region)
        else:
            raise ValueError('Unknown source')
        if frame is None or frame.size == 0:
            if candidate is not None:
                candidate.release()
            raise RuntimeError(f'{mode} returned an empty frame')
        if self.capture is not None:
            self.capture.release()
        self.mode,self.capture,self.buffer = mode,candidate,frame

    def read(self):
        if self.buffer is not None:
            frame,self.buffer = self.buffer,None
            return frame
        if self.mode == 'screen':
            return self.screen_grabber(self.region)
        ok,frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError('Camera stopped returning frames; press A to switch sources')
        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture,self.buffer = None,None


def save_faces(frame, faces, directory):
    directory = Path(directory)
    saved = []
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for index,face in enumerate(faces):
        x,y,w,h = face['box']
        crop = frame[max(0,y):min(frame.shape[0],y+h),max(0,x):min(frame.shape[1],x+w)]
        if crop.size == 0:
            continue
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'face-{stamp}-{index}.png'
        if not cv2.imwrite(str(path),crop):
            raise OSError(f'Could not save {path}')
        saved.append((path,crop))
    return saved


def display_frame(frame, controls, source, fps, faces, message=''):
    # Fixed canvas preserves image aspect ratio when switching screen/camera.
    canvas=np.full((720,960,3),24,dtype=np.uint8)
    scale=min(960/frame.shape[1],550/frame.shape[0])
    width,height=max(1,round(frame.shape[1]*scale)),max(1,round(frame.shape[0]*scale))
    resized=cv2.resize(frame,(width,height))
    x,y=(960-width)//2,(550-height)//2
    canvas[y:y+height,x:x+width]=resized
    def line(text,row,color=(235,235,235)):
        cv2.putText(canvas,text,(15,575+row*27),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
    status=f'Source: {source.upper()} | Predictions: {"ON" if controls.predicting else "OFF"} | Pose: {"ON" if controls.pose else "OFF"}'
    if controls.fps:
        status+=f' | FPS: {fps:.1f} | Faces: {faces}'
    line(status,0,(100,230,180))
    if controls.help:
        line('A: camera / screen    P: predictions    E: pose estimation',1)
        line('I: show / hide help    F: show / hide FPS    SPACE: save detected faces',2)
        line('ESC or Q: exit    Click this live window before pressing keys.',3)
    else:
        line('I: show controls',1)
    if message:
        line(message[:112],4,(80,200,255))
    return canvas


def run_live(args, root):
    from expression import ExpressionEstimator
    from pose import PoseEstimator,draw_poses
    from pose_quality import assess_pose
    controls=Controls()
    if args.pose_only:
        controls.pose=True
    expression=None if args.pose_only else ExpressionEstimator(root/'models/expression-baseline.h5',args.expression_threshold)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    source=LiveSource(args.camera if args.camera is not None else 0,args.screen_region)
    pose=writer=stream=None
    message=''; message_until=0; previous=0; index=0; started=time.monotonic(); fps=0.
    last_frame=None
    fps_sample=time.monotonic()
    try:
        source.switch('screen' if args.screen else 'camera')
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW,960,720)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,720))
            if not writer.isOpened():
                raise RuntimeError('Could not open live video output')
            stream=args.output.with_suffix('.jsonl').open('w',encoding='utf8')
        while not args.max_frames or index < args.max_frames:
            tick=time.monotonic()
            try:
                frame=source.read()
                if frame is None or frame.size == 0:
                    raise RuntimeError('Source returned an empty frame')
                if frame.shape[1]>1280:
                    frame=cv2.resize(frame,(1280,round(frame.shape[0]*1280/frame.shape[1])))
                last_frame=frame
            except Exception as error:
                message=str(error);message_until=tick+5
                frame=last_frame if last_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
                # Do not infer on or save stale frames after a capture failure.
                canvas=display_frame(frame,controls,source.mode,fps,0,message)
                cv2.imshow(WINDOW,canvas)
                action=controls.key(cv2.waitKey(50)&0xFF)
                if action=='quit' or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                    break
                if action=='switch':
                    try:
                        source.switch('screen' if source.mode=='camera' else 'camera')
                        if pose is not None: pose.close();pose=None
                    except Exception as switch_error:
                        message=str(switch_error);message_until=time.monotonic()+5
                continue
            if expression:
                faces=expression.detect(frame,predict=controls.predicting)
            else:
                boxes=detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(48,48))
                faces=[{'box':list(map(int,box))} for box in boxes]
            poses=[]
            if controls.pose:
                if pose is None:
                    pose=PoseEstimator(root/f'models/pose_landmarker_{args.pose_model}.task',video=True,max_people=args.max_people)
                timestamp=max(previous+1,int((tick-started)*1000));previous=timestamp
                poses=pose.detect(frame,timestamp)
            annotated=frame.copy()
            quality=[assess_pose(p,args.landmark_threshold) for p in poses]
            drawable=poses if args.show_partial_poses else [p for p,q in zip(poses,quality) if q['displayable']]
            draw_poses(annotated,drawable,args.landmark_threshold)
            for face in faces:
                x,y,w,h=face['box']
                cv2.rectangle(annotated,(x,y),(x+w,y+h),(80,230,80),2)
                if 'label' in face:
                    cv2.putText(annotated,face['label'],(x,max(y-8,16)),cv2.FONT_HERSHEY_SIMPLEX,.5,(80,230,80),2)
            notice=message if time.monotonic()<message_until else ''
            if not notice and controls.pose and len(drawable)<len(poses):
                notice='Incomplete pose withheld. Full body must be visible.'
            if not notice and source.mode=='screen':
                notice='Primary screen capture. Use --screen-region to avoid capturing this preview.'
            now=time.monotonic()
            elapsed=now-fps_sample
            fps_sample=now
            fps=.85*fps+.15/max(elapsed,.001) if fps else 1/max(elapsed,.001)
            canvas=display_frame(annotated,controls,source.mode,fps,len(faces),notice)
            cv2.imshow(WINDOW,canvas)
            if writer:
                writer.write(canvas)
                stream.write(json.dumps({'frame':index,'source':source.mode,'controls':vars(controls),'poses':poses,'pose_quality':quality,'faces':faces})+'\n')
            index+=1
            action=controls.key(cv2.waitKey(1)&0xFF)
            if action=='quit' or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                break
            try:
                if action=='switch':
                    source.switch('screen' if source.mode=='camera' else 'camera')
                    last_frame=None
                    if pose is not None: pose.close();pose=None
                elif action=='pose' and not controls.pose and pose is not None:
                    pose.close();pose=None
                elif action=='predicting' and expression is None:
                    controls.predicting=False
                    message='Predictions unavailable in --pose-only mode';message_until=time.monotonic()+4
                elif action=='capture':
                    saved=save_faces(frame,faces,args.capture_dir)
                    for path,crop in saved:
                        cv2.imshow('Captured face - '+path.name, crop)
                    message=f'Saved {len(saved)} face crop(s) to {args.capture_dir}' if saved else 'No faces detected to save'
                    message_until=time.monotonic()+5
            except Exception as error:
                message=str(error);message_until=time.monotonic()+5
        print(f'Live session ended after {index} frames',flush=True)
    finally:
        source.close()
        if pose is not None: pose.close()
        if writer is not None: writer.release()
        if stream is not None: stream.close()
        cv2.destroyAllWindows()
