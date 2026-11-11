"""Larger, multiscale Dream driven only by the loaded expression model's weights.

Color preservation restores source chroma after luminance ascent. It does not
give a grayscale classifier learned color features or alter its predictions.
"""
import cv2
import numpy as np
from dream_selection import filter_activation
def preview_size(shape,long_edge):
    height,width=shape[:2]
    if min(height,width)<=0:raise ValueError('Empty image')
    scale=min(float(long_edge)/max(height,width),1.)
    # Preserve aspect ratio. Pad very narrow inputs rather than stretching them.
    return max(1,round(width*scale)),max(1,round(height*scale))


class ModelDream:
    total_updates=362

    def __new__(cls,model):
        if cls is ModelDream and model.input_shape[-1]==3:
            from rgb_model_dream import RGBModelDream
            return RGBModelDream(model)
        return super().__new__(cls)

    def __init__(self,model):
        import keras
        self.keras=keras;self.model=model;self.cache={};self.previous=None;self.last_input=None;self.last_key=None;self.counter=0;self.development=None;self.deep_features=None;self.measurement=None

    def reset(self):
        self.previous=None;self.last_input=None;self.last_key=None;self.counter=0;self.development=None;self.measurement=None

    def _validate_filter(self,layer,channel):
        selected=self.model.get_layer(layer)
        if not isinstance(selected,self.keras.layers.Conv2D):raise ValueError('Choose a convolution layer')
        if isinstance(channel,bool) or not isinstance(channel,int) or not 0<=channel<selected.filters:
            raise ValueError('Filter ID is outside the selected layer')

    def _response(self,frame,layer,channel):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        gray=cv2.copyMakeBorder(gray,0,max(0,48-gray.shape[0]),0,max(0,48-gray.shape[1]),cv2.BORDER_REFLECT_101)
        tensor=self.keras.ops.convert_to_tensor(gray[None,:,:,None])
        def calculate():
            response=filter_activation(self.extractor(layer)(tensor,training=False),channel)
            return float(self.keras.ops.convert_to_numpy(self.keras.ops.mean(response)))
        if self.keras.backend.backend()=='torch':
            import torch
            with torch.no_grad():return calculate()
        return calculate()

    def _measure(self,original,image,layer,channel,before=None):
        before=self._response(original,layer,channel) if before is None else before
        after=self._response(image,layer,channel)
        self.measurement=dict(layer=layer,channel=channel,before=before,after=after)
        return f'Mean response {before:.3g} → {after:.3g}'

    def render_developed(self,frame,strength=.025,steps=6,long_edge=384,preserve_color=True,layer=None,channel=None,preserve_structure=False):
        """Develop a held frame progressively; each call performs a bounded chunk.

        A selected channel's spatial mean, or three late layers in mixed mode,
        guides long coarse-to-fine ascent. Classification weights stay fixed.
        """
        from backend import device_info
        target=(layer,channel,preserve_structure)
        if (layer is None)!=(channel is None):raise ValueError('Choose both a layer and a filter')
        if channel is not None:self._validate_filter(layer,channel)
        if self.development is not None and self.development['target']!=target:self.reset()
        if channel is None and self.deep_features is None:
            layers=[l for l in self.model.layers if isinstance(l,self.keras.layers.Conv2D)][-3:]
            source=self.keras.Model(self.model.inputs,[l.output for l in layers])
            self.deep_features=self.keras.models.clone_model(source,input_tensors=[self.keras.Input((None,None,1))])
            self.deep_features.set_weights(source.get_weights());self.deep_features.trainable=False
        extractor=self.extractor(layer) if channel is not None else self.deep_features
        def feature_outputs(tensor):
            acts=extractor(tensor,training=False)
            if channel is not None:return [filter_activation(acts,channel)]
            return list(acts) if isinstance(acts,(list,tuple)) else [acts]
        if self.development is None:
            edge=min(long_edge,256) if device_info()['device']=='CPU' else long_edge
            width,height=preview_size(frame.shape,edge)
            original=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
            chroma=cv2.cvtColor(original,cv2.COLOR_BGR2YCrCb)
            gray=chroma[:,:,0].astype(np.float32)/255.
            self.development=dict(chroma=chroma,original=original,gray=gray,x=gray.copy(),base=None,phase=0,done=0,norm=None,total=0,target=target,before=None)
        d=self.development;gray=d['gray']
        schedule=[(48,240),(96,80),(192,30),(max(gray.shape),12)]
        count=max(1,min(24,int(steps)))
        for _ in range(count):
            if d['phase']>=len(schedule):break
            edge,limit=schedule[d['phase']]
            edge=min(edge,max(gray.shape));scale=edge/max(gray.shape)
            size=(max(48,round(gray.shape[1]*scale)),max(48,round(gray.shape[0]*scale)))
            if d['done']==0:
                base=cv2.resize(gray,size)
                x=cv2.resize(d['x'],size)
                if d['base'] is not None:x+=.25*(base-cv2.resize(d['base'],size))
                d['x']=x;d['base']=base
                tensor=self.keras.ops.convert_to_tensor(x[None,:,:,None])
                # Fixed layer scales per octave avoid one high-amplitude layer dominating.
                activations=feature_outputs(tensor)
                d['norm']=[max(.01,float(np.abs(self.keras.ops.convert_to_numpy(a)).mean())) for a in activations]
                del activations,tensor
            i=d['done'];dy=i*3%9-4;dx=i*5%9-4
            tensor=self.keras.ops.convert_to_tensor(np.roll(d['x'],(dy,dx),(0,1))[None,:,:,None])
            if self.keras.backend.backend()=='torch':
                import torch
                tensor=tensor.detach().requires_grad_(True)
                acts=feature_outputs(tensor)
                objective=sum(a.mean()/v for a,v in zip(acts,d['norm']))/len(acts)
                gradient,=torch.autograd.grad(objective,tensor)
                updated=(tensor+strength*gradient/(gradient.std(unbiased=False)+1e-8)).clamp(0,1).detach()
            else:
                import tensorflow as tf
                with tf.GradientTape() as tape:
                    tape.watch(tensor);acts=feature_outputs(tensor)
                    objective=sum(tf.reduce_mean(a)/v for a,v in zip(acts,d['norm']))/len(acts)
                gradient=tape.gradient(objective,tensor)
                if gradient is None:raise ValueError('Feature graph has no input gradient')
                updated=tf.clip_by_value(tensor+strength*gradient/(tf.math.reduce_std(gradient)+1e-8),0,1)
            if preserve_structure:
                from dream_structure import guided_step
                updated=guided_step(self.keras.ops.convert_to_numpy(tensor)[0,:,:,0],self.keras.ops.convert_to_numpy(gradient)[0,:,:,0],np.roll(d['base'],(dy,dx),(0,1)),strength)[None,:,:,None]
            d['x']=np.roll(self.keras.ops.convert_to_numpy(updated)[0,:,:,0],(-dy,-dx),(0,1))
            d['done']+=1;d['total']+=1
            if d['done']>=limit:d['phase']+=1;d['done']=0
        displayed=cv2.resize(d['x'],(gray.shape[1],gray.shape[0]))
        if preserve_structure:displayed=np.clip(displayed,gray-.14,gray+.14)
        luminance=np.uint8(np.clip(displayed,0,1)*255)
        if preserve_color:
            chroma=d['chroma'].copy();chroma[:,:,0]=luminance
            image=cv2.cvtColor(chroma,cv2.COLOR_YCrCb2BGR)
        else:image=cv2.cvtColor(luminance,cv2.COLOR_GRAY2BGR)
        stage=['Developing large structures','Building shapes','Refining detail','Finishing','Dream complete'][d['phase']]
        if channel is not None:
            measured=self._measure(d['original'],image,layer,channel,d['before']);d['before']=self.measurement['before']
            description=f'{layer} · filter {channel} only. {measured} at the displayed resolution.'
        else:description='Your model’s three deepest convolution layers.'
        return image,f'{stage} · held frame · {d["total"]}/362 updates. {description} Choose Develop this frame to start again.'

    def extractor(self,layer):
        if layer not in self.cache:
            source=self.keras.Model(self.model.inputs,self.model.get_layer(layer).output)
            # Clone only the convolutional feature graph, before fixed-size dense layers.
            # These are exactly the loaded classifier's weights, with variable spatial size.
            clone=self.keras.models.clone_model(source,input_tensors=[self.keras.Input((None,None,1))])
            clone.set_weights(source.get_weights());clone.trainable=False
            self.cache[layer]=clone
        return self.cache[layer]

    def render(self,frame,layer,strength=.015,steps=3,long_edge=384,preserve_color=True,animate=True,channel=None,preserve_structure=False):
        from backend import device_info
        edge=min(long_edge,256) if device_info()['device']=='CPU' else long_edge
        width,height=preview_size(frame.shape,edge)
        original=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
        chroma=cv2.cvtColor(original,cv2.COLOR_BGR2YCrCb)
        gray=chroma[:,:,0].astype(np.float32)/255.
        # Reflection padding supports small face crops and very thin screen regions.
        padded=cv2.copyMakeBorder(gray,0,max(0,48-height),0,max(0,48-width),cv2.BORDER_REFLECT_101)
        if channel is not None:self._validate_filter(layer,channel)
        key=(layer,channel,padded.shape,preserve_structure)
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
                    objective=filter_activation(activation,channel).mean() if channel is not None else activation.square().mean()
                    gradient,=torch.autograd.grad(objective,tensor)
                    gradient=gradient/(gradient.std(unbiased=False)+1e-8)
                    updated=(tensor+strength*gradient).clamp(0,1).detach()
                else:
                    import tensorflow as tf
                    with tf.GradientTape() as tape:
                        tape.watch(tensor);activation=extractor(tensor,training=False)
                        objective=tf.reduce_mean(filter_activation(activation,channel)) if channel is not None else tf.reduce_mean(tf.square(activation))
                    gradient=tape.gradient(objective,tensor)
                    if gradient is None:raise ValueError('Selected feature graph has no input gradient')
                    updated=tf.clip_by_value(tensor+strength*gradient/(tf.math.reduce_std(gradient)+1e-8),0,1)
                if preserve_structure:
                    from dream_structure import guided_step
                    base=np.roll(cv2.resize(padded,size),(dy,dx),(0,1))
                    updated=guided_step(self.keras.ops.convert_to_numpy(tensor)[0,:,:,0],self.keras.ops.convert_to_numpy(gradient)[0,:,:,0],base,strength)[None,:,:,None]
                x=np.roll(self.keras.ops.convert_to_numpy(updated)[0,:,:,0],(-dy,-dx),(0,1))
        x=cv2.resize(x,(padded.shape[1],padded.shape[0]),interpolation=cv2.INTER_LINEAR)
        if preserve_structure:x=np.clip(x,padded-.14,padded+.14)
        self.previous=x;self.last_input=padded;self.last_key=key
        luminance=np.uint8(np.clip(x[:height,:width],0,1)*255)
        if not preserve_color:image=cv2.cvtColor(luminance,cv2.COLOR_GRAY2BGR)
        else:
            chroma[:,:,0]=luminance
            image=cv2.cvtColor(chroma,cv2.COLOR_YCrCb2BGR)
        if channel is not None:self._measure(original,image,layer,channel)
        else:self.measurement=None
        return image
