"""True RGB ascent through the SAME expression checkpoint, with explicit priors.

Color gradients come from learned RGB weights. Smoothing and the optional
source constraint are image priors, not additional learned art networks.
"""
import cv2
import numpy as np
from dream_selection import filter_activation


class RGBModelDream:
    total_updates=160

    def __init__(self,model):
        import keras
        self.keras=keras;self.model=model;self.cache={};self.mixed=None;self.reset()

    def reset(self):
        self.previous=None;self.last_input=None;self.last_key=None;self.counter=0;self.development=None;self.measurement=None

    def _validate_filter(self,layer,channel):
        selected=self.model.get_layer(layer)
        if not isinstance(selected,self.keras.layers.Conv2D):raise ValueError('Choose a convolution layer')
        if isinstance(channel,bool) or not isinstance(channel,int) or not 0<=channel<selected.filters:raise ValueError('Filter ID is outside the selected layer')

    def extractor(self,layer):
        if layer not in self.cache:
            source=self.keras.Model(self.model.inputs,self.model.get_layer(layer).output)
            clone=self.keras.models.clone_model(source,input_tensors=[self.keras.Input((None,None,3))])
            clone.set_weights(source.get_weights());clone.trainable=False;self.cache[layer]=clone
        return self.cache[layer]

    def _features(self,layer,channel):
        if layer is not None:
            extractor=self.extractor(layer)
            return lambda x:[filter_activation(extractor(x,training=False),channel)] if channel is not None else [extractor(x,training=False)]
        if self.mixed is None:
            names=['stage2_block2_features','stage3_block2_features','stage4_block2_features']
            if not all(name in [l.name for l in self.model.layers] for name in names):
                names=[l.name for l in self.model.layers if isinstance(l,self.keras.layers.Conv2D)][-3:]
            source=self.keras.Model(self.model.inputs,[self.model.get_layer(name).output for name in names])
            self.mixed=self.keras.models.clone_model(source,input_tensors=[self.keras.Input((None,None,3))])
            self.mixed.set_weights(source.get_weights());self.mixed.trainable=False
        def get(x):
            outputs=self.mixed(x,training=False)
            return list(outputs) if isinstance(outputs,(list,tuple)) else [outputs]
        return get

    def _prepare(self,frame,long_edge):
        from model_dream import preview_size
        from backend import device_info
        edge=min(long_edge,256) if device_info()['device']=='CPU' else long_edge
        w,h=preview_size(frame.shape,edge);original=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
        rgb=cv2.cvtColor(original,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        # Reflect very narrow images instead of distorting their aspect ratio.
        padded=cv2.copyMakeBorder(rgb,0,max(0,64-h),0,max(0,64-w),cv2.BORDER_REFLECT_101)
        return original,padded

    def _response(self,image,layer,channel):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        rgb=cv2.copyMakeBorder(rgb,0,max(0,64-rgb.shape[0]),0,max(0,64-rgb.shape[1]),cv2.BORDER_REFLECT_101)
        tensor=self.keras.ops.convert_to_tensor(rgb[None]);extractor=self.extractor(layer)
        if self.keras.backend.backend()=='torch':
            import torch
            with torch.no_grad():return float(filter_activation(extractor(tensor,training=False),channel).mean().cpu())
        return float(self.keras.ops.convert_to_numpy(self.keras.ops.mean(filter_activation(extractor(tensor,training=False),channel))))

    def _measure(self,original,image,layer,channel,before=None):
        self.measurement=dict(layer=layer,channel=channel,before=self._response(original,layer,channel) if before is None else before,after=self._response(image,layer,channel))

    def _scales(self,features,x):
        tensor=self.keras.ops.convert_to_tensor(x[None])
        if self.keras.backend.backend()=='torch':
            import torch
            with torch.no_grad():return [max(.05,float(a.abs().mean().cpu())) for a in features(tensor)]
        return [max(.05,float(np.abs(self.keras.ops.convert_to_numpy(a)).mean())) for a in features(tensor)]

    def _step(self,x,base,features,norm,strength,preserve_structure):
        # Small translations reduce dependence on one pixel alignment. Reflect
        # padding avoids wraparound seams from rolling opposite image borders.
        dy=self.counter*3%7-3;dx=self.counter*5%7-3;self.counter+=1
        padded=cv2.copyMakeBorder(x,4,4,4,4,cv2.BORDER_REFLECT_101)
        shifted=padded[4+dy:4+dy+x.shape[0],4+dx:4+dx+x.shape[1]]
        tensor=self.keras.ops.convert_to_tensor(shifted[None])
        if self.keras.backend.backend()=='torch':
            import torch
            tensor=tensor.detach().requires_grad_(True)
            acts=features(tensor);target=sum(a.mean()/v for a,v in zip(acts,norm))/len(acts)
            variation=(tensor[:,1:]-tensor[:,:-1]).square().mean()+(tensor[:,:,1:]-tensor[:,:,:-1]).square().mean()
            gradient,=torch.autograd.grad(target-.12*variation,tensor)
            gradient=gradient.detach().cpu().numpy()[0]
        else:
            import tensorflow as tf
            with tf.GradientTape() as tape:
                tape.watch(tensor);acts=features(tensor);target=sum(tf.reduce_mean(a)/v for a,v in zip(acts,norm))/len(acts)
                variation=tf.reduce_mean(tf.square(tensor[:,1:]-tensor[:,:-1]))+tf.reduce_mean(tf.square(tensor[:,:,1:]-tensor[:,:,:-1]))
                objective=target-.12*variation
            gradient=tape.gradient(objective,tensor)
            if gradient is None:raise ValueError('RGB features have no input gradient')
            gradient=gradient.numpy()[0]
        # Put the shifted gradient back; uncovered border pixels receive zero.
        unshifted=np.zeros_like(gradient)
        sy0=max(0,-dy);sy1=min(x.shape[0],x.shape[0]-dy);sx0=max(0,-dx);sx1=min(x.shape[1],x.shape[1]-dx)
        unshifted[sy0+dy:sy1+dy,sx0+dx:sx1+dx]=gradient[sy0:sy1,sx0:sx1]
        # Multi-radius smoothing suppresses fine noise without inventing colors.
        gradient=sum(cv2.GaussianBlur(unshifted,(0,0),sigma)*weight for sigma,weight in ((.6,.5),(1.4,.3),(2.8,.2)))
        scale=max(float(np.sqrt(np.mean(gradient**2))),1e-8)
        candidate=x+float(strength)*gradient/scale
        if preserve_structure:candidate=np.clip(candidate,base-.18,base+.18)
        return np.clip(candidate,0,1).astype(np.float32)

    def _display(self,x,original,preserve_color):
        h,w=original.shape[:2];pixels=np.uint8(np.clip(x[:h,:w],0,1)*255)
        image=cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
        return image if preserve_color else cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)

    def render_developed(self,frame,strength=.025,steps=6,long_edge=384,preserve_color=True,layer=None,channel=None,preserve_structure=False):
        if (layer is None)!=(channel is None):raise ValueError('Choose both a layer and a filter')
        if channel is not None:self._validate_filter(layer,channel)
        target=(layer,channel,preserve_structure,preserve_color,long_edge)
        if self.development is not None and self.development['target']!=target:self.reset()
        if self.development is None:
            original,rgb=self._prepare(frame,long_edge)
            self.development=dict(original=original,rgb=rgb,x=rgb.copy(),base=None,phase=0,done=0,total=0,norm=None,target=target,before=None)
        d=self.development;rgb=d['rgb'];features=self._features(layer,channel)
        schedule=[(112,48),(224,64),(max(rgb.shape[:2]),48)]
        for _ in range(max(1,min(24,int(steps)))):
            if d['phase']>=len(schedule):break
            edge,limit=schedule[d['phase']];scale=min(edge,max(rgb.shape[:2]))/max(rgb.shape[:2]);size=(max(64,round(rgb.shape[1]*scale)),max(64,round(rgb.shape[0]*scale)))
            if not d['done']:
                base=cv2.resize(rgb,size);x=cv2.resize(d['x'],size)
                # Carry the learned change between scales, restoring source detail.
                if d['base'] is not None:x+=base-cv2.resize(d['base'],size)
                d['x']=np.clip(x,0,1);d['base']=base;d['norm']=self._scales(features,d['x'])
            d['x']=self._step(d['x'],d['base'],features,d['norm'],strength,preserve_structure);d['done']+=1;d['total']+=1
            if d['done']>=limit:d['phase']+=1;d['done']=0
        displayed=cv2.resize(d['x'],(rgb.shape[1],rgb.shape[0]))
        if preserve_structure:displayed=np.clip(displayed,rgb-.18,rgb+.18)
        image=self._display(displayed,d['original'],preserve_color)
        if channel is not None:
            self._measure(d['original'],image,layer,channel,d['before']);d['before']=self.measurement['before']
        stage=['Building broad color patterns','Developing feature shapes','Refining detail','Dream complete'][d['phase']]
        return image,f'{stage} · {d["total"]}/{self.total_updates} updates. Learned RGB gradients from the expression model; image smoothing applied.'

    def render(self,frame,layer,strength=.015,steps=3,long_edge=384,preserve_color=True,animate=True,channel=None,preserve_structure=False):
        if channel is not None:self._validate_filter(layer,channel)
        original,rgb=self._prepare(frame,long_edge);key=(layer,channel,rgb.shape,preserve_structure,preserve_color)
        x=rgb.copy()
        if animate and self.previous is not None and self.last_key==key and np.abs(rgb-self.last_input).mean()<.12:x=.5*x+.5*self.previous
        features=self._features(layer,channel);norm=self._scales(features,x)
        for _ in range(max(1,min(12,int(steps)))):x=self._step(x,rgb,features,norm,strength,preserve_structure)
        self.previous=x;self.last_input=rgb;self.last_key=key;image=self._display(x,original,preserve_color)
        if channel is not None:self._measure(original,image,layer,channel)
        else:self.measurement=None
        return image
