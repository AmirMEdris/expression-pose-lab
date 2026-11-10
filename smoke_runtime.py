"""Exercise the actual background workers without using a camera or recording it."""
from pathlib import Path
import argparse
import json
import queue
import time
from backend import select_backend
select_backend()
from runtime import Engine


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).parent;engine=Engine(root,root/'results/captures');result={}
    def wait(q,predicate=lambda packet:True):
        end=time.monotonic()+90
        while time.monotonic()<end:
            while not engine.events.empty():
                kind,value=engine.events.get_nowait()
                if kind=='error':raise RuntimeError(value)
            try:
                packet=q.get(timeout=.2)
                if predicate(packet):return packet
            except queue.Empty:pass
        raise TimeoutError('Background worker did not produce output')
    try:
        engine.configure(predictions=True,visual='Activations');engine.switch('Image',image=args.image)
        frame=wait(engine.frames);features=wait(engine.visuals)
        result.update(faces=len(frame['faces']),source=frame['source'],feature_shape=list(features['image'].shape))
        engine.configure(visual='DeepDream',colorful=False,dream_engine='Expression',dream_motion='Live')
        dream=wait(engine.visuals,lambda p:p['mode']=='DeepDream')
        import numpy as np
        assert np.array_equal(dream['image'][:,:,0],dream['image'][:,:,1])
        assert np.array_equal(dream['image'][:,:,1],dream['image'][:,:,2])
        result.update(dream_shape=list(dream['image'].shape),dream_seconds=dream['seconds'],grayscale=True)
        engine.configure(colorful=True,dream_scope='Whole frame',dream_resolution=256)
        color=wait(engine.visuals,lambda p:p['settings'].colorful)
        assert color['image'].shape[:2]==(218,256)
        result.update(color_shape=list(color['image'].shape),color_seconds=color['seconds'])
        engine.configure(visual='GradCAM',explain_target='happy')
        explained=wait(engine.visuals,lambda p:p['mode']=='GradCAM')
        assert 'happy' in explained['caption']
        result['explanation']=explained['caption']
        engine.configure(visual='DeepDream',dream_motion='Develop',dream_revision=1)
        developing=wait(engine.visuals,lambda p:p['mode']=='DeepDream' and p['settings'].dream_revision==1)
        assert 'held frame' in developing['caption']
        engine.configure(dream_revision=2)
        restarted=wait(engine.visuals,lambda p:p['mode']=='DeepDream' and p['settings'].dream_revision==2)
        assert restarted['caption'].startswith('Developing large structures')
        result['held_frame_restart']='passed'
        engine.configure(visual='GradCAM')
        import cv2
        args.output.parent.mkdir(parents=True,exist_ok=True)
        blank=args.output.with_name('blank-explanation-test.png')
        cv2.imwrite(str(blank),np.zeros((128,192,3),np.uint8))
        previous=engine.generation;engine.switch('Image',image=str(blank))
        no_face=wait(engine.visuals,lambda p:p['generation']>previous and p['mode']=='GradCAM')
        assert no_face['caption'].startswith('No face detected') and not no_face['image'].any()
        result['no_face_explanation']='passed'
        assert engine.frames.qsize()<=1 and engine.visuals.qsize()<=1
        result['status']='passed'
    finally:
        engine.close();engine.worker.join(10);engine.dream_worker.join(10)
        assert not engine.worker.is_alive() and not engine.dream_worker.is_alive()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
