"""Offline distribution smoke test: bundled models, gradients, pose, and Tk runtime."""
import json
from pathlib import Path
import time
import numpy as np
import cv2


def run(root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={};start=time.monotonic()
    try:
        import tkinter as tk
        window=tk.Tk();window.withdraw();window.update();window.destroy();result['tk']='passed'
        from expression import ExpressionEstimator
        estimator=ExpressionEstimator(root/'models/expression-baseline.h5')
        y,x=np.mgrid[:48,:48]
        frame=np.uint8((np.sin(x*.4)+np.cos(y*.3)+2)/4*255)
        from backend import predict_numpy,device_info
        result['device']=device_info()
        probabilities=predict_numpy(estimator.model,frame[None,:,:,None].astype(np.float32)/255.)
        assert probabilities.shape==(1,7) and np.isfinite(probabilities).all() and np.isclose(probabilities.sum(),1)
        result['expression']='passed';result['parameters']=estimator.model.count_params()
        from visualizations import FeatureVisualizer
        visualizer=FeatureVisualizer(estimator.model)
        for mode in ('Activations','DeepDream'):
            before=time.monotonic();image=visualizer.render(frame,mode=mode,steps=3,animate=False)
            assert image.dtype==np.uint8 and image.shape[2]==3 and image.std()>1
            cv2.imwrite(str(output/(mode.lower()+'.png')),image)
            result[mode]=dict(status='passed',seconds=time.monotonic()-before)
        from pose import PoseEstimator
        pose=PoseEstimator(root/'models/pose_landmarker_lite.task')
        try:result['blank_pose_count']=len(pose.detect(np.zeros((256,256,3),np.uint8)))
        finally:pose.close()
        result['pose']='passed';result['status']='passed'
    except Exception as error:
        import traceback
        result['status']='failed';result['error']=str(error);result['traceback']=traceback.format_exc()
        raise
    finally:
        result['seconds']=time.monotonic()-start
        (output/'selftest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
