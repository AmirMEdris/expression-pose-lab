"""CanvasNet: a custom RGB expression CNN, trained from random initialization.

Familiar residual blocks, gradual average pooling, and a two-scale classifier.
No pretrained backbone, perceptual network, or separate art model is used.
"""
from pathlib import Path
import json
import hashlib
import torch
from torch import nn

WIDTHS=(48,96,192,320)
DROPOUT=(.05,.10,.15,.20)
IMAGE_SIZE=224
LABELS=('angry','disgust','fear','happy','sad','surprise','neutral')


class ConvNorm(nn.Sequential):
    def __init__(self,inputs,outputs,kernel=3,relu=True):
        super().__init__()
        self.add_module('conv',nn.Conv2d(inputs,outputs,kernel,padding=kernel//2,bias=False))
        self.add_module('bn',nn.BatchNorm2d(outputs,eps=1e-5,momentum=.1))
        if relu:self.add_module('relu',nn.ReLU(inplace=False))


class Residual(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.first=ConvNorm(inputs,outputs);self.second=ConvNorm(outputs,outputs,relu=False)
        self.skip=ConvNorm(inputs,outputs,1,relu=False) if inputs!=outputs else nn.Identity()
    def forward(self,x):return torch.relu(self.second(self.first(x))+self.skip(x))


class CanvasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1=ConvNorm(3,24);self.stem2=ConvNorm(24,32);self.pool=nn.AvgPool2d(2)
        self.stages=nn.ModuleList();self.dropouts=nn.ModuleList();previous=32
        for width,rate in zip(WIDTHS,DROPOUT):
            self.stages.append(nn.Sequential(Residual(previous,width),Residual(width,width)))
            self.dropouts.append(nn.Dropout2d(rate));previous=width
        self.hidden=nn.Linear(WIDTHS[-2]+WIDTHS[-1],192)
        self.head_dropout=nn.Dropout(.35);self.output=nn.Linear(192,7)
        for module in self.modules():
            if isinstance(module,nn.Conv2d):nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
    def features(self,x):
        x=self.pool(self.stem1(x*2-1));x=self.pool(self.stem2(x));outputs=[]
        for i,(stage,dropout) in enumerate(zip(self.stages,self.dropouts)):
            if i:x=self.pool(x)
            x=stage(x);outputs.append(x);x=dropout(x)
        return outputs,x
    def forward(self,x):
        features,last=self.features(x)
        # Both intermediate and deep features directly contribute to expression.
        middle=self.dropouts[-2](features[-2]).mean((2,3))
        x=torch.cat((middle,last.mean((2,3))),dim=1)
        return self.output(self.head_dropout(torch.relu(self.hidden(x))))


def export_keras(network,path,extra=None):
    """Copy all learned parameters and BN statistics into standard Keras layers.

    Checkpoint is identical for inference, activation maps, Grad-CAM, and Dream.
    Layers stay flat so the live app can select each individual convolution.
    """
    import os
    os.environ.setdefault('KERAS_BACKEND','torch')
    import keras
    from keras import layers as L
    inputs=keras.Input((IMAGE_SIZE,IMAGE_SIZE,3),name='rgb_image')
    x=L.Rescaling(2.,offset=-1.,name='rgb_normalization')(inputs);transfers=[]
    def conv(x,block,name):
        op=L.Conv2D(block.conv.out_channels,block.conv.kernel_size,padding='same',use_bias=False,name=name+'_conv')
        x=op(x);transfers.append((op,[block.conv.weight.detach().cpu().numpy().transpose(2,3,1,0)]))
        bn=L.BatchNormalization(epsilon=1e-5,momentum=.9,name=name+'_bn');x=bn(x)
        transfers.append((bn,[getattr(block.bn,k).detach().cpu().numpy() for k in ('weight','bias','running_mean','running_var')]))
        return L.ReLU(name=name+'_relu')(x) if hasattr(block,'relu') else x
    x=L.AveragePooling2D(2,name='stem_pool1')(conv(x,network.stem1,'stem1'))
    x=L.AveragePooling2D(2,name='stem_pool2')(conv(x,network.stem2,'stem2'));features=[]
    for i,(stage,rate) in enumerate(zip(network.stages,DROPOUT)):
        if i:x=L.AveragePooling2D(2,name=f'stage{i+1}_pool')(x)
        for j,block in enumerate(stage):
            name=f'stage{i+1}_block{j+1}';skip=x
            y=conv(conv(x,block.first,name+'_a'),block.second,name+'_b')
            if not isinstance(block.skip,nn.Identity):skip=conv(skip,block.skip,name+'_skip')
            x=L.ReLU(name=name+'_features')(L.Add(name=name+'_add')([y,skip]))
        features.append(x);x=L.SpatialDropout2D(rate,name=f'stage{i+1}_dropout')(x)
    middle=L.SpatialDropout2D(DROPOUT[-2],name='middle_head_dropout')(features[-2])
    x=L.Concatenate(name='two_scale_features')([L.GlobalAveragePooling2D(name='middle_summary')(middle),L.GlobalAveragePooling2D(name='deep_summary')(x)])
    hidden=L.Dense(192,activation='relu',name='expression_hidden');x=hidden(x)
    x=L.Dropout(.35,name='expression_dropout')(x)
    output=L.Dense(7,activation='softmax',name='expression_probabilities');x=output(x)
    model=keras.Model(inputs,x,name='canvasnet_rgb')
    for layer,weights in transfers:layer.set_weights(weights)
    for layer,source in ((hidden,network.hidden),(output,network.output)):
        layer.set_weights([source.weight.detach().cpu().numpy().T,source.bias.detach().cpu().numpy()])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);model.save(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    metadata=dict(name='CanvasNet RGB',architecture='custom residual RGB CNN with two-scale head',pretrained=False,
                  input_shape=[IMAGE_SIZE,IMAGE_SIZE,3],color_space='RGB',normalization='divide_by_255',labels=list(LABELS),
                  checkpoint_sha256=digest,parameters=sum(p.numel() for p in network.parameters()),
                  spatial_dropout=list(DROPOUT),head_dropout=.35,
                  dream_layers=['stage2_block2_features','stage3_block2_features','stage4_block2_features'])
    metadata.update(extra or {});path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return model


if __name__=='__main__':
    model=CanvasNet();print(model);print('Trainable parameters:',sum(p.numel() for p in model.parameters()))
