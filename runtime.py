"""Bounded, independent capture/inference and visualization workers for the desktop UI."""
from pathlib import Path
from dataclasses import dataclass,replace
import queue
import threading
import time
import cv2
import numpy as np
from live import LiveSource,save_faces


def latest(q,item):
    try:q.put_nowait(item)
    except queue.Full:
        try:q.get_nowait()
        except queue.Empty:pass
        try:q.put_nowait(item)
        except queue.Full:pass


@dataclass
class Settings:
    source: str = 'Stopped'
    revision: int = 0
    camera: int = 0
    region: tuple | None = None
    image: str = ''
    model: str = ''
    predictions: bool = False
    pose: bool = False
    visual: str = 'Off'
    layer: str = ''
    strength: float = .025
    steps: int = 6
    threshold: float = .6
    colorful: bool = False
    dream_engine: str = 'Expression'
    dream_scope: str = 'Whole frame'
    dream_depth: str = 'Shapes'
    dream_resolution: int = 384
    explain_target: str = 'Top prediction'
    dream_motion: str = 'Develop'
    dream_revision: int = 0


class Engine:
    def __init__(self,root,captures):
        self.root=Path(root);self.captures=Path(captures)
        self.settings=Settings(model=str(self.root/'models/expression-baseline.h5'))
        self.lock=threading.Lock();self.stop=threading.Event();self.capture_face=threading.Event()
        self.frames=queue.Queue(1);self.visuals=queue.Queue(1);self.dream_input=queue.Queue(1);self.events=queue.Queue()
        self.generation=0
        self.worker=threading.Thread(target=self._run,daemon=True,name='capture-inference')
        self.dream_worker=threading.Thread(target=self._dream,daemon=True,name='feature-visualization')
        self.worker.start();self.dream_worker.start()

    def configure(self,**kwargs):
        with self.lock:self.settings=replace(self.settings,**kwargs)

    def switch(self,source,**kwargs):
        with self.lock:self.settings=replace(self.settings,source=source,revision=self.settings.revision+1,**kwargs)

    def snapshot(self):
        with self.lock:return replace(self.settings)

    def _run(self):
        try:self._run_impl()
        except Exception as error:self.events.put(('error','Runtime initialization failed: '+str(error)))

    def _run_impl(self):
        from backend import device_info
        info=device_info()
        self.events.put(('device',f'{info["backend"]} · {info["device"]}'))
        from expression import ExpressionEstimator
        from pose_quality import assess_pose
        source=None;pose=None;expression=None;model_path=None;revision=-1;still=None;active='Stopped';last_visual=0.;last_frame=None;fps=0.;last_tick=time.monotonic();timestamp=0
        cached_face_key=None;cached_faces=None
        try:
            while not self.stop.is_set():
                cycle_start=time.monotonic()
                s=self.snapshot()
                if s.revision!=revision:
                    revision=s.revision
                    try:
                        candidate=None;candidate_still=None
                        if s.source=='Image':
                            candidate_still=cv2.imread(s.image)
                            if candidate_still is None:raise ValueError('Cannot read this image')
                        elif s.source in ('Camera','Screen'):
                            # Switching within a live source is transactional.
                            if source is not None and source.index==s.camera:
                                previous_region=source.region;source.region=s.region
                                try:
                                    if not (source.mode=='camera' and s.source=='Camera'):source.switch(s.source.lower())
                                except Exception:source.region=previous_region;raise
                                candidate=source
                            else:
                                candidate=LiveSource(s.camera,s.region)
                                try:candidate.switch(s.source.lower())
                                except Exception:candidate.close();raise
                        if source is not None and source is not candidate:source.close()
                        source=candidate;still=candidate_still;active=s.source
                        self.capture_face.clear();last_frame=None;self.generation+=1
                        if pose is not None:pose.close();pose=None
                        self.events.put(('source',active))
                    except Exception as error:
                        self.events.put(('error',str(error)))
                        self.events.put(('source',active))
                if active=='Stopped':
                    self.stop.wait(.05);continue
                try:
                    if model_path!=s.model:
                        self.events.put(('status','Loading expression model…'))
                        replacement=ExpressionEstimator(Path(s.model),s.threshold)
                        expression=replacement;model_path=s.model;self.generation+=1
                        self.events.put(('model',model_path))
                        from visualizations import FeatureVisualizer
                        visualizer=FeatureVisualizer(expression.model)
                        self.events.put(('layers',visualizer.layers))
                        self.events.put(('status','Ready · all processing stays on this computer'))
                    frame=still.copy() if still is not None else source.read()
                    if frame is None or frame.size==0:raise RuntimeError('No frame available; choose another source or retry Camera')
                    if frame.shape[1]>960:frame=cv2.resize(frame,(960,round(frame.shape[0]*960/frame.shape[1])))
                    expression.threshold=s.threshold
                    face_key=(self.generation,s.predictions,s.threshold)
                    if still is not None and face_key==cached_face_key:faces=cached_faces
                    else:
                        faces=expression.detect(frame,predict=s.predictions)
                        cached_face_key=face_key if still is not None else None;cached_faces=faces
                    annotated=frame.copy();pose_count=0;withheld=0
                    if s.pose:
                        from pose import PoseEstimator,draw_poses
                        if pose is None:pose=PoseEstimator(self.root/'models/pose_landmarker_lite.task',video=True,max_people=4)
                        timestamp=max(timestamp+1,int(time.monotonic()*1000))
                        people=pose.detect(frame,timestamp)
                        drawable=[p for p in people if assess_pose(p,.5)['displayable']]
                        draw_poses(annotated,drawable);pose_count=len(drawable);withheld=len(people)-pose_count
                    elif pose is not None:pose.close();pose=None
                    for face in faces:
                        x,y,w,h=face['box'];cv2.rectangle(annotated,(x,y),(x+w,y+h),(180,234,94),2)
                        if 'label' in face:cv2.putText(annotated,f'{face["label"]} {face["score"]:.0%}',(x,max(18,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,234,94),1,cv2.LINE_AA)
                    now=time.monotonic();elapsed=now-last_tick;last_tick=now
                    fps=1/max(elapsed,.001) if not fps else .9*fps+.1/max(elapsed,.001)
                    largest=max(faces,key=lambda f:f['box'][2]*f['box'][3],default=None)
                    visual_frame=frame
                    if largest and (s.visual in ('Activations','GradCAM') or s.dream_scope=='Face only'):
                        x,y,w,h=largest['box'];visual_frame=frame[y:y+h,x:x+w]
                    if s.visual!='Off' and now-last_visual>.15:
                        latest(self.dream_input,(visualizer,visual_frame.copy(),s,self.generation,largest is not None));last_visual=now
                    if self.capture_face.is_set():
                        self.capture_face.clear();saved=save_faces(frame,faces,self.captures)
                        self.events.put(('status',f'Saved {len(saved)} face crop(s) in {self.captures}' if saved else 'No faces detected to save'))
                    latest(self.frames,dict(image=annotated,faces=faces,source=active,fps=fps,poses=pose_count,withheld=withheld,generation=self.generation))
                    last_frame=frame
                except Exception as error:
                    self.events.put(('error',str(error)))
                    # Pause failed source, keeping the UI available to retry.
                    if source is not None:source.close();source=None
                    if pose is not None:pose.close();pose=None
                    active='Stopped';self.events.put(('source',active));self.generation+=1
                # Bound capture/inference work to 30 FPS instead of a busy loop.
                self.stop.wait(max(.001,1/30-(time.monotonic()-cycle_start)))
        except Exception as error:self.events.put(('error',str(error)))
        finally:
            if source is not None:source.close()
            if pose is not None:pose.close()

    def _dream(self):
        rgb_dream=None;dream_model=None;previous_key=None;explainer=None;explained_model=None
        while not self.stop.is_set():
            try:visualizer,frame,s,generation,has_face=self.dream_input.get(timeout=.2)
            except queue.Empty:continue
            if self.snapshot().visual=='Off' or generation!=self.generation:continue
            try:
                key=(generation,s.dream_scope,s.dream_engine,s.dream_depth,s.dream_resolution,s.layer,s.visual,s.dream_motion,s.dream_revision)
                if key!=previous_key:
                    visualizer.previous=None
                    if rgb_dream is not None:rgb_dream.reset()
                previous_key=key
                start=time.monotonic()
                caption=None
                layer=s.layer if s.layer in visualizer.layers else visualizer.layers[0]
                if s.visual=='GradCAM':
                    if not has_face:
                        rendered=frame;caption='No face detected. Explanation needs a detected face; no heatmap is shown.'
                    else:
                        if explained_model is not visualizer.model:
                            from gradcam import GradCAM
                            explainer=GradCAM(visualizer.model);explained_model=visualizer.model
                        rendered,details=explainer.explain(frame,s.explain_target);caption=details['caption']
                elif s.visual=='DeepDream':
                    if dream_model is not visualizer.model:
                        from model_dream import ModelDream
                        rgb_dream=ModelDream(visualizer.model);dream_model=visualizer.model
                    if s.dream_motion=='Develop':
                        rendered,caption=rgb_dream.render_developed(frame,s.strength,s.steps,s.dream_resolution,preserve_color=s.colorful)
                    else:
                        rendered=rgb_dream.render(frame,layer,s.strength,s.steps,s.dream_resolution,preserve_color=s.colorful)
                        caption='Following live input · lighter effect. Use Develop this frame for a longer transformation.'
                else:rendered=visualizer.render(frame,layer,s.visual,s.strength,s.steps,colorful=s.colorful)
                if generation==self.generation:
                    latest(self.visuals,dict(image=rendered,seconds=time.monotonic()-start,mode=s.visual,layer=layer,generation=generation,engine=s.dream_engine,scope=s.dream_scope,settings=s,caption=caption))
                self.stop.wait(max(0,.15-(time.monotonic()-start)))
            except Exception as error:
                self.events.put(('error','Visualization: '+str(error)));self.configure(visual='Off')

    def close(self):
        self.stop.set()
