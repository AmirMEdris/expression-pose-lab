"""Activation maps and actual gradient-ascent DeepDream on expression features."""
import cv2
import numpy as np


def colorize(values):
    values=np.asarray(values,dtype=np.float32)
    lo,hi=np.percentile(values,[2,98])
    gray=np.uint8(np.clip((values-lo)/max(float(hi-lo),1e-6),0,1)*255)
    return cv2.applyColorMap(gray,cv2.COLORMAP_TURBO)


def activation_grid(features, columns=4, tile=112):
    """Show 16 stable channel IDs, independently normalized; no fake heatmaps."""
    features=np.asarray(features)
    indices=np.linspace(0,features.shape[-1]-1,min(16,features.shape[-1]),dtype=int)
    rows=(len(indices)+columns-1)//columns
    canvas=np.full((rows*(tile+24),columns*(tile+8),3),22,np.uint8)
    for cell,channel in enumerate(indices):
        x=(cell%columns)*(tile+8);y=(cell//columns)*(tile+24)
        values=features[:,:,channel]
        lo,hi=np.percentile(values,[2,98])
        gray=np.uint8(np.clip((values-lo)/max(float(hi-lo),1e-6),0,1)*255)
        canvas[y:y+tile,x:x+tile]=cv2.resize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),(tile,tile))
        cv2.putText(canvas,f'Filter {channel}',(x+4,y+tile+17),cv2.FONT_HERSHEY_SIMPLEX,.38,(205,215,225),1,cv2.LINE_AA)
    return canvas


class FeatureVisualizer:
    def __init__(self, model):
        import keras
        self.keras=keras
        self.layers=[l.name for l in model.layers if isinstance(l,keras.layers.Conv2D)]
        if not self.layers: raise ValueError('Model has no supported convolution layers')
        self.model=model;self.cache={};self.previous=None;self.previous_layer=None

    def extractor(self, layer):
        if layer not in self.layers: raise ValueError('Choose a convolution layer')
        if layer not in self.cache:
            # Reuse inference weights. The classifier itself still sees 48x48 faces.
            self.cache[layer]=self.keras.Model(self.model.inputs,self.model.get_layer(layer).output)
        return self.cache[layer]

    def render(self, frame, layer=None, mode='Activations', strength=.015, steps=3, animate=True, colorful=False):
        layer=layer or self.layers[0];extractor=self.extractor(layer)
        if mode=='Activations':
            from expression import preprocess_frame
            from backend import predict_numpy
            self.previous=None
            return activation_grid(predict_numpy(extractor,preprocess_frame(frame,self.model.input_shape))[0])
        if self.model.input_shape[-1]==3:
            from model_dream import ModelDream
            if not hasattr(self,'rgb_dream'):self.rgb_dream=ModelDream(self.model)
            return self.rgb_dream.render(frame,layer,strength,steps,preserve_color=colorful,animate=animate)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
        original=cv2.resize(gray,(48,48)).astype(np.float32)/255.
        x=self.keras.ops.convert_to_tensor(original[None,:,:,None])
        if mode=='Activations':
            self.previous=None
            from backend import predict_numpy
            return activation_grid(predict_numpy(extractor,x)[0])
        if mode!='DeepDream': raise ValueError('Unknown visualization mode')
        if animate and self.previous is not None and layer==self.previous_layer:
            x=.65*x+.35*self.previous
        if self.keras.backend.backend()=='torch':
            import torch
            for _ in range(max(1,min(int(steps),12))):
                x=x.detach().requires_grad_(True)
                activations=extractor(x,training=False)
                objective=activations.square().mean()
                gradient,=torch.autograd.grad(objective,x)
                gradient=gradient/(gradient.std(unbiased=False)+1e-8)
                x=(x+float(strength)*gradient).clamp(0,1).detach()
        else:
            import tensorflow as tf
            for _ in range(max(1,min(int(steps),12))):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    activations=extractor(x,training=False)
                    objective=tf.reduce_mean(tf.square(activations))
                gradient=tape.gradient(objective,x)
                if gradient is None: raise ValueError('Selected model does not expose input gradients')
                gradient=gradient/(tf.math.reduce_std(gradient)+1e-8)
                x=tf.clip_by_value(x+float(strength)*gradient,0,1)
        self.previous=x;self.previous_layer=layer
        dream=self.keras.ops.convert_to_numpy(x)[0,:,:,0]
        # The color palette is artistic; the underlying pixels are gradient-ascent output.
        palette=colorize(dream) if colorful else cv2.cvtColor(np.uint8(np.clip(dream,0,1)*255),cv2.COLOR_GRAY2BGR)
        return cv2.resize(palette,(480,480),interpolation=cv2.INTER_CUBIC)
