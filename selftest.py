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
        from gradcam import GradCAM
        explainer=GradCAM(estimator.model)
        face=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        explained,details=explainer.explain(face)
        from expression import LABELS
        assert details['label']==LABELS[int(probabilities[0].argmax())]
        assert abs(details['score']-float(probabilities[0].max()))<1e-5
        assert np.isfinite(details['heatmap']).all()
        _,other=explainer.explain(face,'happy')
        assert other['label']=='happy'
        # Explanation must not alter inference weights, activations, or probabilities.
        np.testing.assert_allclose(predict_numpy(estimator.model,frame[None,:,:,None].astype(np.float32)/255.),probabilities,atol=1e-6)
        cv2.imwrite(str(output/'gradcam.png'),explained)
        result['GradCAM']=dict(status='passed',label=details['label'],score=details['score'])
        from visualizations import FeatureVisualizer
        visualizer=FeatureVisualizer(estimator.model)
        for mode in ('Activations','DeepDream'):
            before=time.monotonic();image=visualizer.render(frame,mode=mode,steps=3,animate=False)
            assert image.dtype==np.uint8 and image.shape[2]==3 and image.std()>1
            cv2.imwrite(str(output/(mode.lower()+'.png')),image)
            result[mode]=dict(status='passed',seconds=time.monotonic()-before)
        from model_dream import ModelDream
        dream=ModelDream(estimator.model)
        check=frame[None,:,:,None].astype(np.float32)/255.
        np.testing.assert_allclose(predict_numpy(dream.extractor(visualizer.layers[-1]),check),predict_numpy(visualizer.extractor(visualizer.layers[-1]),check),atol=1e-5)
        y,x=np.mgrid[:180,:256]
        rgb=np.stack([(np.sin(x*.06)+1)*110+20,(np.cos(y*.08)+1)*110+20,(np.sin((x+y)*.04)+1)*110+20],axis=-1).astype(np.uint8)
        before=time.monotonic();art=dream.render(rgb,visualizer.layers[-1],steps=3,long_edge=256,animate=False)
        assert art.shape==rgb.shape and art.dtype==np.uint8
        assert not np.array_equal(art[:,:,0],art[:,:,1])
        assert np.abs(art.astype(float)-rgb).mean()>1
        cv2.imwrite(str(output/'color-dream.png'),art)
        result['ModelDream']=dict(status='passed',seconds=time.monotonic()-before,shape=list(art.shape))
        developed,caption=dream.render_developed(rgb,steps=6,long_edge=256)
        held=dream.development['gray'].copy()
        dream.render_developed(np.zeros_like(rgb),steps=6,long_edge=256)
        np.testing.assert_array_equal(dream.development['gray'],held)
        assert dream.development['total']==12 and developed.shape==rgb.shape
        dream.reset()
        assert dream.development is None
        result['DevelopDream']=dict(status='passed',held_frame=True,reset=True)
        # Opposing channels make accidental channel mixing observable: filter 0
        # must brighten the input; filter 1 must darken it, even when negative.
        import keras
        inputs=keras.Input((48,48,1))
        conv=keras.layers.Conv2D(2,1,use_bias=False,name='opposed_filters')
        probe=keras.Model(inputs,conv(inputs))
        conv.set_weights([np.array([1.,-1.],np.float32).reshape(1,1,1,2)])
        neutral=np.full((48,48,3),128,np.uint8)
        for channel in (0,1):
            isolated=ModelDream(probe)
            bright=isolated.render(neutral,'opposed_filters',channel=channel,steps=1,animate=False,preserve_color=False)
            assert (bright.mean()>128) if channel==0 else (bright.mean()<128)
            assert isolated.measurement['after']>isolated.measurement['before']
            developed,_=isolated.render_developed(neutral,layer='opposed_filters',channel=channel,steps=1,preserve_color=False)
            assert (developed.mean()>128) if channel==0 else (developed.mean()<128)
        np.testing.assert_array_equal(conv.get_weights()[0].reshape(-1),[1.,-1.])
        # Exercise a real model target and switching, including exact displayed
        # image measurement and held-frame reset rather than residual feedback.
        target=visualizer.layers[3]
        single=dream.render(rgb,target,channel=35,steps=3,long_edge=256,animate=False)
        assert single.shape==rgb.shape and dream.measurement['channel']==35
        assert np.isfinite([dream.measurement['before'],dream.measurement['after']]).all()
        cv2.imwrite(str(output/'single-filter-dream.png'),single)
        dream.render_developed(rgb,layer=target,channel=35,steps=2,long_edge=256)
        dream.render_developed(np.zeros_like(rgb),layer=target,channel=35,steps=2,long_edge=256)
        assert dream.development['total']==4
        dream.render_developed(np.zeros_like(rgb),layer=target,channel=36,steps=2,long_edge=256)
        assert dream.development['total']==2 and not dream.development['original'].any()
        assert dream.measurement['channel']==36
        np.testing.assert_allclose(predict_numpy(estimator.model,check),probabilities,atol=1e-6)
        result['SingleFilterDream']=dict(status='passed',opposing_channel_gradients=True,target_switch_reset=True,model_unchanged=True)
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
