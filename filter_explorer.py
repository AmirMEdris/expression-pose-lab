"""Audit real filter responses and verify single-unit Dreams on local FER2013 data.

Produces a self-contained local HTML report, model-bound metadata, and reproducible
numeric evidence. Dataset imagery must not be committed or published by default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def matched_controls(descriptors,scores,anchor,excluded,count=3):
    """Low-response controls within the nearest 12.5% in global mean/std space."""
    scale=np.maximum(descriptors.std(axis=0),1e-6)
    distance=np.square((descriptors-descriptors[anchor])/scale).sum(axis=1)
    distance[np.asarray(excluded,dtype=int)]=np.inf
    candidates=np.argsort(distance,kind='stable')[:min(len(scores)-len(excluded),max(count,len(scores)//8))]
    return candidates[np.argsort(scores[candidates],kind='stable')[:count]]


def scan(model,images,batch=32,progress=None):
    import torch
    import keras
    layers=[l for l in model.layers if isinstance(l,keras.layers.Conv2D)]
    net=keras.Model(model.inputs,[l.output for l in layers])
    result={l.name:dict(scores=[],positions=[],shape=None) for l in layers}
    with torch.no_grad():
        for start in range(0,len(images),batch):
            outputs=net(keras.ops.convert_to_tensor(images[start:start+batch]),training=False)
            if not isinstance(outputs,(list,tuple)):outputs=[outputs]
            for layer,a in zip(layers,outputs):
                flat=a.reshape(a.shape[0],-1,a.shape[-1]);scores,indices=flat.max(dim=1)
                row=result[layer.name];row['scores'].append(scores.cpu().numpy());row['positions'].append(indices.cpu().numpy());row['shape']=list(a.shape[1:])
            if start%(batch*16)==0:print(f'Scanned {min(start+batch,len(images))}/{len(images)} faces',flush=True)
            if progress is not None:progress(min(start+batch,len(images)),len(images))
    for row in result.values():
        row['scores']=np.concatenate(row['scores']);row['positions']=np.concatenate(row['positions'])
    return result


def targeted_dreams(net,seeds,channels,steps=24):
    """Fixed spatial target per seed, backtracking, and verification after 8-bit export."""
    import torch
    import keras
    x=keras.ops.convert_to_tensor(seeds.astype(np.float32)/255.)
    channel=torch.as_tensor(channels,device=x.device,dtype=torch.long)
    ids=torch.arange(len(seeds),device=x.device)
    with torch.no_grad():
        a=net(x,training=False);height,width=int(a.shape[1]),int(a.shape[2])
        locations=a.reshape(len(seeds),-1,a.shape[-1])[ids,:,channel].argmax(dim=1)
        yy=locations//width;xx=locations%width
        def response(value):return net(value,training=False)[ids,yy,xx,channel]
        before=response(x).detach()
    best=x.detach();score=before.clone();accepted=torch.zeros(len(seeds),dtype=torch.int32,device=x.device)
    for _ in range(steps):
        variable=best.detach().requires_grad_(True)
        values=response(variable)
        gradient,=torch.autograd.grad(values.sum(),variable)
        gradient=gradient/(gradient.flatten(1).std(dim=1,unbiased=False).reshape(-1,1,1,1)+1e-8)
        with torch.no_grad():
            next_image=best;next_score=score;improved=torch.zeros(len(seeds),dtype=torch.bool,device=x.device)
            for step in (.03,.01,.003):
                candidate=(best+step*gradient).clamp(0,1)
                value=response(candidate)
                choose=torch.isfinite(value)&(value>next_score+1e-7)
                next_image=torch.where(choose[:,None,None,None],candidate,next_image)
                next_score=torch.where(choose,value,next_score);improved|=choose
            best=next_image.detach();score=next_score.detach();accepted+=improved.int()
        if not bool(improved.any()):break
    # The measured artifact is exactly the 8-bit image displayed in the report.
    with torch.no_grad():
        exported=(best*255).round().clamp(0,255)/255.
        after=response(exported)
        keep=torch.isfinite(after)&(after>=before)
        exported=torch.where(keep[:,None,None,None],exported,x)
        after=response(exported)
    return (exported.cpu().numpy()*255).round().astype(np.uint8),dict(before=before.cpu().numpy(),after=after.cpu().numpy(),y=yy.cpu().numpy(),x=xx.cpu().numpy(),accepted=accepted.cpu().numpy())


def run(model_path,dataset,output,steps=24):
    from backend import select_backend,device_info
    select_backend()
    if device_info()['backend']!='torch':raise ValueError('Analysis uses the tested PyTorch backend. Set KERAS_BACKEND=torch.')
    import keras
    from expression import ExpressionEstimator
    from dataset import load_dataset
    from filter_geometry import model_support,support_box
    from filter_report import write_report
    started=time.monotonic();model_path=Path(model_path);output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    original=ExpressionEstimator(model_path).model
    source=keras.Model(original.inputs,original.outputs[0])
    model=keras.models.clone_model(source);model.set_weights(source.get_weights());model.trainable=False
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    data,audit=load_dataset(dataset);images,labels=data['validation']
    real=(images*255).round().astype(np.uint8)
    descriptors=np.stack([images.mean(axis=(1,2,3)),images.std(axis=(1,2,3))],axis=1)
    geometry=model_support(model)
    measurements=scan(model,images)
    layers=[];dream_images=[];evidence={};verified=0;attempts=0
    for layer_index,(name,row) in enumerate(measurements.items()):
        scores=row['scores'];pos=row['positions'];height,width,channels=row['shape']
        net=keras.Model(model.inputs,model.get_layer(name).output)
        info=dict(name=name,index=layer_index,channels=channels,shape=row['shape'],support=geometry[name],filters=[])
        for start in range(0,channels,16):
            jobs=[];seed_images=[];channel_ids=[]
            for channel in range(start,min(start+16,channels)):
                order=np.argsort(-scores[:,channel],kind='stable');top=order[:6];low=order[-3:];median=int(order[len(order)//2])
                def example(index):
                    index=int(index);y,x=divmod(int(pos[index,channel]),width)
                    return dict(image=index,score=float(scores[index,channel]),box=support_box(geometry[name],y,x),unit=[y,x])
                counts,edges=np.histogram(scores[:,channel],bins=24)
                controls=matched_controls(descriptors,scores[:,channel],int(top[0]),top)
                entry=dict(channel=channel,top=[example(i) for i in top],low=[example(i) for i in low],matched=[example(i) for i in controls],median=float(np.median(scores[:,channel])),p95=float(np.percentile(scores[:,channel],95)),minimum=float(scores[:,channel].min()),maximum=float(scores[:,channel].max()),histogram=dict(counts=counts.tolist(),edges=edges.tolist()),dreams=[])
                info['filters'].append(entry)
                for title,seed in [('Strong dataset example',real[int(top[0])]),('Typical dataset example',real[median]),('Seeded random noise',np.random.default_rng(2026+layer_index*10000+channel).integers(0,256,(48,48,1),dtype=np.uint8))]:
                    jobs.append((entry,title));seed_images.append(seed);channel_ids.append(channel)
            seeds=np.stack(seed_images)
            rendered,stats=targeted_dreams(net,seeds,channel_ids,steps)
            for j,(entry,title) in enumerate(jobs):
                index=len(dream_images);dream_images.extend([seeds[j,:,:,0],rendered[j,:,:,0]])
                before=float(stats['before'][j]);after=float(stats['after'][j]);increase=after>before+max(1e-6,abs(before)*1e-5)
                entry['dreams'].append(dict(title=title,before_image=index,after_image=index+1,before=before,after=after,
                    increased=bool(increase),accepted=int(stats['accepted'][j]),
                    unit=[int(stats['y'][j]),int(stats['x'][j])],
                    box=support_box(geometry[name],int(stats['y'][j]),int(stats['x'][j]))))
                verified+=int(increase);attempts+=1
            print(f'{name}: analyzed filters {start}–{min(start+15,channels-1)}',flush=True)
        layers.append(info)
        evidence[name+'_scores']=scores;evidence[name+'_positions']=pos
    result=dict(checkpoint_sha256=digest,model_name=model_path.name,partition='FER2013 validation / PublicTest',images=len(real),dataset=audit,layers=layers,steps=steps,seed=2026,verified_increases=verified,dream_attempts=attempts,environment=device_info(),seconds=time.monotonic()-started)
    np.savez_compressed(output.with_suffix('.npz'),real=real,dreams=np.stack(dream_images),**evidence)
    write_report(output,result,real[:,:,:,0],np.stack(dream_images))
    manifest={k:v for k,v in result.items() if k!='layers'}
    manifest['report_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    output.with_suffix('.evidence.json').write_text(json.dumps(result),encoding='utf-8')
    print(json.dumps(dict(status='complete',filters=sum(l['channels'] for l in layers),verified=verified,attempts=attempts,report=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=24)
    a=p.parse_args()
    if a.steps<1:raise ValueError('Steps must be positive')
    run(a.model,a.dataset,a.output,a.steps)
