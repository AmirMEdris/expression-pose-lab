"""Structural input support of spatial units; not an attention or semantic map."""
import math


def spatial_step(parent,kernel,stride,dilation,padding,input_size):
    jump,low,high=parent
    effective=(kernel-1)*dilation+1
    before=max(0,(math.ceil(input_size/stride)-1)*stride+effective-input_size)//2 if padding=='same' else 0
    return jump*stride,low-before*jump,high+(effective-1-before)*jump


def merge_support(parents):
    if len({p[0] for p in parents})!=1:raise ValueError('Cannot merge differently spaced grids')
    return parents[0][0],min(p[1] for p in parents),max(p[2] for p in parents)


def support_box(geometry,y,x,size=48):
    gy,gx=geometry
    x0=max(0,min(size,int(x*gx[0]+gx[1])));x1=max(0,min(size,int(x*gx[0]+gx[2]+1)))
    y0=max(0,min(size,int(y*gy[0]+gy[1])));y1=max(0,min(size,int(y*gy[0]+gy[2]+1)))
    return [x0,y0,x1,y1]


def model_support(model):
    cache={}
    def visit(tensor):
        key=id(tensor)
        if key in cache:return cache[key]
        history=tensor._keras_history;op=history.operation
        name=type(op).__name__
        if name=='InputLayer':value=((1,0,0),(1,0,0))
        else:
            node=op._inbound_nodes[history.node_index]
            inputs=node.input_tensors
            parents=[visit(t) for t in inputs]
            value=tuple(merge_support([p[a] for p in parents]) for a in range(2))
            if name in ('Conv2D','MaxPooling2D','AveragePooling2D'):
                kernel=op.kernel_size if name=='Conv2D' else op.pool_size
                dilation=op.dilation_rate if name=='Conv2D' else (1,1)
                value=tuple(spatial_step(value[a],kernel[a],op.strides[a],dilation[a],op.padding,int(inputs[0].shape[a+1])) for a in range(2))
            elif name not in ('BatchNormalization','Activation','ReLU','Dropout','SpatialDropout2D','Add','Concatenate','Rescaling'):
                raise ValueError('Unsupported spatial layer: '+name)
        cache[key]=value;return value
    return {layer.name:visit(layer.output) for layer in model.layers if type(layer).__name__=='Conv2D'}
