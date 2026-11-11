"""Offline checks that RGB visualizations use learned color gradients."""
import numpy as np


def run(root):
    import keras
    from model_dream import ModelDream
    from visualizations import FeatureVisualizer
    from filter_inspector import FilterInspector
    from gradcam import GradCAM
    from expression import ExpressionEstimator,preprocess_frame
    from backend import predict_numpy
    inputs=keras.Input((224,224,3));conv=keras.layers.Conv2D(2,1,use_bias=False,name='color_filters')
    features=conv(inputs);out=keras.layers.Dense(7,activation='softmax')(keras.layers.GlobalAveragePooling2D()(features));model=keras.Model(inputs,out)
    weights=np.zeros((1,1,3,2),np.float32);weights[0,0,0,0]=1;weights[0,0,2,1]=1;conv.set_weights([weights])
    before=[w.copy() for w in model.get_weights()];frame=np.full((224,224,3),100,np.uint8)
    dream=ModelDream(model)
    for channel,bgr_channel in ((0,2),(1,0)):
        image=dream.render(frame,'color_filters',steps=3,strength=.02,long_edge=224,channel=channel,animate=False)
        assert image[:,:,bgr_channel].mean()>image[:,:,1].mean()+5
        assert np.abs(image[:,:,2-bgr_channel].astype(int)-image[:,:,1]).max()<=1
        assert dream.measurement['after']>dream.measurement['before']
    for _ in range(7):image,_=dream.render_developed(frame,steps=24,long_edge=224,layer='color_filters',channel=1,preserve_structure=True)
    assert dream.development['total']==160 and np.abs(image.astype(float)-frame).max()<=47
    for a,b in zip(before,model.get_weights()):np.testing.assert_array_equal(a,b)
    inspection=FilterInspector(FeatureVisualizer(model)).render(frame,'color_filters',0)
    assert inspection['grid'].shape==(224,224) and '224' in inspection['reference_caption']
    _,explanation=GradCAM(model).explain(frame);assert np.isfinite(explanation['heatmap']).all()
    result=dict(status='passed',separate_rgb_gradients=True,weights_unchanged=True,held_updates=160,source_constraint=True)
    installed=root/'models/expression-rgb.keras'
    if installed.exists():
        estimator=ExpressionEstimator(installed);model=estimator.model;visualizer=FeatureVisualizer(model)
        batch=preprocess_frame(frame,model.input_shape);expected=predict_numpy(model,batch)
        assert expected.shape==(1,7) and np.isclose(expected.sum(),1)
        target=visualizer.layers[-1];dream=ModelDream(model)
        np.testing.assert_allclose(predict_numpy(dream.extractor(target),batch),predict_numpy(visualizer.extractor(target),batch),atol=1e-5)
        image,_=dream.render_developed(frame,steps=1,long_edge=224)
        assert image.shape==frame.shape
        FilterInspector(visualizer).render(frame,target,0);GradCAM(model).explain(frame)
        np.testing.assert_allclose(predict_numpy(model,batch),expected,atol=1e-6)
        result['installed_checkpoint']='passed'
    return result
