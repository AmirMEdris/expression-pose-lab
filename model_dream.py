"""Larger, multiscale Dream driven only by the loaded expression model's weights.

Color preservation restores source chroma after luminance ascent. It does not
give a grayscale classifier learned color features or alter its predictions.
"""
import cv2
import numpy as np
def preview_size(shape,long_edge):
    height,width=shape[:2]
    if min(height,width)<=0:raise ValueError('Empty image')
    scale=min(float(long_edge)/max(height,width),1.)
    # Preserve aspect ratio. Pad very narrow inputs rather than stretching them.
    return max(1,round(width*scale)),max(1,round(height*scale))


class ModelDream:
    def __init__(self,model):
        import keras
        self.keras=keras;self.model=model;self.cache={};self.previous=None;self.last_input=None;self.last_key=None;self.counter=0

    def reset(self):
        self.previous=None;self.last_input=None;self.last_key=None;self.counter=0

    def extractor(self,layer):
        if layer not in self.cache:
            source=self.keras.Model(self.model.inputs,self.model.get_layer(layer).output)
            # Clone only the convolutional feature graph, before fixed-size dense layers.
            # These are exactly the loaded classifier's weights, with variable spatial size.
            clone=self.keras.models.clone_model(source,input_tensors=[self.keras.Input((None,None,1))])
            clone.set_weights(source.get_weights());clone.trainable=False
            self.cache[layer]=clone
        return self.cache[layer]

    def render(self,frame,layer,strength=.015,steps=3,long_edge=384,preserve_color=True,animate=True):
        from backend import device_info
        edge=min(long_edge,256) if device_info()['device']=='CPU' else long_edge
        width,height=preview_size(frame.shape,edge)
        original=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
        chroma=cv2.cvtColor(original,cv2.COLOR_BGR2YCrCb)
        gray=chroma[:,:,0].astype(np.float32)/255.
        # Reflection padding supports small face crops and very thin screen regions.
        padded=cv2.copyMakeBorder(gray,0,max(0,48-height),0,max(0,48-width),cv2.BORDER_REFLECT_101)
        key=(layer,padded.shape)
        x=padded.copy()
        if animate and self.previous is not None and self.last_key==key and np.abs(padded-self.last_input).mean()<.16:
            x=.4*x+.6*self.previous
        extractor=self.extractor(layer)
        total=max(1,min(12,int(steps)))
        schedule=[(.65,max(1,total//3)),(1.,total-max(1,total//3))] if total>=3 else [(1.,total)]
        for scale,count in schedule:
            size=(max(48,round(padded.shape[1]*scale)),max(48,round(padded.shape[0]*scale)))
            x=cv2.resize(x,size,interpolation=cv2.INTER_LINEAR)
            for _ in range(count):
                dy=self.counter*7%17-8;dx=self.counter*11%17-8;self.counter+=1
                shifted=np.roll(x,(dy,dx),(0,1))
                tensor=self.keras.ops.convert_to_tensor(shifted[None,:,:,None])
                if self.keras.backend.backend()=='torch':
                    import torch
                    tensor=tensor.detach().requires_grad_(True)
                    activation=extractor(tensor,training=False)
                    gradient,=torch.autograd.grad(activation.square().mean(),tensor)
                    gradient=gradient/(gradient.std(unbiased=False)+1e-8)
                    updated=(tensor+strength*gradient).clamp(0,1).detach()
                else:
                    import tensorflow as tf
                    with tf.GradientTape() as tape:
                        tape.watch(tensor);activation=extractor(tensor,training=False);objective=tf.reduce_mean(tf.square(activation))
                    gradient=tape.gradient(objective,tensor)
                    if gradient is None:raise ValueError('Selected feature graph has no input gradient')
                    updated=tf.clip_by_value(tensor+strength*gradient/(tf.math.reduce_std(gradient)+1e-8),0,1)
                x=np.roll(self.keras.ops.convert_to_numpy(updated)[0,:,:,0],(-dy,-dx),(0,1))
        x=cv2.resize(x,(padded.shape[1],padded.shape[0]),interpolation=cv2.INTER_LINEAR)
        self.previous=x;self.last_input=padded;self.last_key=key
        luminance=np.uint8(np.clip(x[:height,:width],0,1)*255)
        if not preserve_color:return cv2.cvtColor(luminance,cv2.COLOR_GRAY2BGR)
        chroma[:,:,0]=luminance
        return cv2.cvtColor(chroma,cv2.COLOR_YCrCb2BGR)
