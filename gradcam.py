"""Class-specific Grad-CAM from the loaded expression model, without changing weights."""
import cv2
import numpy as np


def activation_heatmap(features,gradients):
    features=np.asarray(features,dtype=np.float32);gradients=np.asarray(gradients,dtype=np.float32)
    if features.shape!=gradients.shape or features.ndim!=3:
        raise ValueError('Expected matching height/width/channel tensors')
    if not np.isfinite(features).all() or not np.isfinite(gradients).all():
        raise ValueError('Nonfinite explanation values')
    cam=np.maximum((features*gradients.mean(axis=(0,1))).sum(axis=-1),0)
    peak=float(cam.max())
    return cam/peak if peak>0 else np.zeros_like(cam)


def overlay(frame,heatmap):
    heat=cv2.resize(heatmap,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_LINEAR)
    colors=cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_JET)
    alpha=.55*heat[:,:,None]
    return np.uint8(np.clip(frame*(1-alpha)+colors*alpha,0,255))


class GradCAM:
    def __init__(self,model):
        import keras
        self.keras=keras
        # Sequential models may retain old symbolic nodes from incremental builds.
        # Rebuild the actual inference graph once, with identical weights, so each
        # returned feature tensor belongs to the same graph as the class score.
        source=keras.Model(model.inputs,model.outputs[0])
        model=keras.models.clone_model(source)
        model.set_weights(source.get_weights())
        self.model=model
        conv=next((layer for layer in reversed(model.layers) if isinstance(layer,keras.layers.Conv2D)),None)
        if conv is None:raise ValueError('Explanation requires a convolution layer')
        self.layer=conv.name;self.head=model.layers[-1]
        self.kind='probability'
        if isinstance(self.head,keras.layers.Dense):self.kind='dense-logit'
        elif isinstance(self.head,keras.layers.Activation) and self.head.activation==keras.activations.softmax:self.kind='logit'
        target=self.head.input if self.kind!='probability' else model.outputs[0]
        self.joint=keras.Model(model.inputs,[conv.output,target,model.outputs[0]])

    def scores(self,hidden):
        if self.kind!='dense-logit':return hidden
        scores=self.keras.ops.matmul(hidden,self.head.kernel)
        return scores+self.head.bias if self.head.use_bias else scores

    def explain(self,frame,target='Top prediction'):
        from expression import preprocess,LABELS
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        x=self.keras.ops.convert_to_tensor(preprocess(gray))
        index=None if target=='Top prediction' else LABELS.index(target)
        if self.keras.backend.backend()=='torch':
            import torch
            import threading
            x=x.detach().requires_grad_(True)
            captured={};owner=threading.get_ident()
            def feature_hook(module,args,output):
                if threading.get_ident()==owner:captured['features']=output
            def head_hook(module,args):
                if threading.get_ident()==owner:captured['hidden']=args[0]
            handles=[self.model.get_layer(self.layer).register_forward_hook(feature_hook),self.head.register_forward_pre_hook(head_hook)]
            try:
                probabilities=self.model(x,training=False)
                features=captured['features'];hidden=captured['hidden'] if self.kind!='probability' else probabilities
                if index is None:index=int(probabilities[0].argmax().detach().cpu())
                gradient,=torch.autograd.grad(self.scores(hidden)[0,index],features)
            finally:
                for handle in handles:handle.remove()
        else:
            import tensorflow as tf
            with tf.GradientTape() as tape:
                tape.watch(x)
                features,hidden,probabilities=self.joint(x,training=False)
                if index is None:index=int(tf.argmax(probabilities[0]))
                score=self.scores(hidden)[0,index]
            gradient=tape.gradient(score,features)
            if gradient is None:raise ValueError('Model does not expose class gradients')
        convert=self.keras.ops.convert_to_numpy
        heatmap=activation_heatmap(convert(features)[0],convert(gradient)[0])
        score=float(convert(probabilities)[0,index])
        image=overlay(frame,heatmap)
        scale=480/max(image.shape[:2]);image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
        detail=f'{LABELS[index]} · model score {score:.0%}. '
        detail+=('Warmer areas indicate stronger positive support for this class.' if heatmap.max()>0 else 'No positive Grad-CAM signal for this class; showing the original face.')
        detail+=' Coarse relative map, not proof of the prediction or emotion.'
        return image,dict(label=LABELS[index],score=score,layer=self.layer,heatmap=heatmap,caption=detail)
