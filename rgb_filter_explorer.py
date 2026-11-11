"""Real RGB validation examples for every filter; local lazy-loaded PNG assets."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from expression import ExpressionEstimator,LABELS
from filter_explorer import scan,matched_controls
from filter_geometry import model_support,support_box
from training_progress import Progress


def run(model_path,dataset,output,experiment=None):
    from backend import device_info
    if device_info()['backend']!='torch':raise ValueError('RGB scanning currently requires the PyTorch backend')
    model_path=Path(model_path);dataset=Path(dataset);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    model=ExpressionEstimator(model_path).model
    if model.input_shape[-1]!=3:raise ValueError('Choose an RGB checkpoint')
    progress=Progress(experiment) if experiment else None
    audit=json.loads((dataset/'prepared.json').read_text(encoding='utf-8'))
    bank=np.memmap(dataset/'faces-224.rgb',dtype=np.uint8,mode='r',shape=(audit['cached_count'],224,224,3))
    with np.load(dataset/'splits.npz') as split:ids=split['val'].copy();labels=split['labels'].copy()
    images=bank[ids].astype(np.float32)/255.
    descriptors=np.stack([images.mean((1,2,3)),images.std((1,2,3))],axis=1);geometry=model_support(model)
    def update(done,total):
        if progress:progress.update(phase='Inspecting learned RGB filters',done=done,total=total,message='Ranking real validation faces for every filter. No camera images or test images used.',progress_text=f'{done:,} / {total:,} validation faces')
    measurements=scan(model,images,batch=16,progress=update);del images
    layers=[];used=set();numeric={}
    for li,(name,row) in enumerate(measurements.items()):
        scores=row['scores'];positions=row['positions'];height,width,channels=row['shape'];filters=[]
        for channel in range(channels):
            order=np.argsort(-scores[:,channel],kind='stable');top=order[:6];controls=matched_controls(descriptors,scores[:,channel],int(top[0]),top)
            def example(i):
                i=int(i);used.add(i);y,x=divmod(int(positions[i,channel]),width)
                return dict(image=i,source_index=int(ids[i]),label=LABELS[labels[ids[i]]],score=float(scores[i,channel]),unit=[y,x],box=support_box(geometry[name],y,x,size=224))
            filters.append(dict(channel=channel,top=[example(i) for i in top],controls=[example(i) for i in controls],median=float(np.median(scores[:,channel])),maximum=float(scores[:,channel].max()),minimum=float(scores[:,channel].min())))
        layers.append(dict(name=name,index=li,channels=channels,shape=row['shape'],support=geometry[name],filters=filters))
        numeric[name+'_scores']=scores;numeric[name+'_positions']=positions
    assets=output.with_suffix('.assets');assets.mkdir(exist_ok=True);hashes={}
    for count,i in enumerate(sorted(used)):
        path=assets/f'{i}.png';cv2.imwrite(str(path),cv2.cvtColor(bank[ids[i]],cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if progress and count%100==0:progress.update(done=count,total=len(used),progress_text=f'{count:,} / {len(used):,} local evidence images',message='Saving lossless RGB examples; the report loads only the selected filter’s images.')
    with model_path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(checkpoint_sha256=digest,model_name=model_path.name,partition='ExpW custom grouped validation',images=len(ids),dataset=audit,
                layers=layers,asset_folder=assets.name,input_size=224,limitations=['Source and detected duplicate groups separated; identities may overlap.','Natural examples establish responses, not automatic semantic labels.','No Dream-seed tests in this report; use matching live-app selectors.'])
    payload=json.dumps(result,separators=(',',':')).replace('</','<\\/')
    template=Path(__file__).with_name('rgb_filter_report.html').read_text(encoding='utf-8')
    output.write_text(template.replace('__EVIDENCE_JSON__',payload),encoding='utf-8')
    numeric.update(validation_source_indices=ids)
    np.savez_compressed(output.with_suffix('.npz'),**numeric)
    output.with_suffix('.evidence.json').write_text(json.dumps(result),encoding='utf-8')
    manifest={k:v for k,v in result.items() if k!='layers'};manifest['report_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();manifest['asset_sha256']=hashes
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if progress:progress.log(f'RGB filter evidence complete: {sum(l["channels"] for l in layers):,} filters, {len(ids):,} validation images.')
    print('Report:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('dataset');p.add_argument('--output',required=True);p.add_argument('--experiment');a=p.parse_args();run(a.model,a.dataset,a.output,a.experiment)
