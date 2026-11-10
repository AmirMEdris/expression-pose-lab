"""Evaluate existing checkpoints and generate reports bound to their weight hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from backend import select_backend,device_info,predict_numpy
select_backend()


def evaluate(model_path,dataset,architecture,note=''):
    import keras
    from expression import build_model,LABELS
    from dataset import load_dataset
    from metrics import classification_metrics
    from reporting import write_report
    model_path=Path(model_path)
    model=keras.models.load_model(model_path,compile=False,safe_mode=True) if model_path.suffix=='.keras' else build_model()
    if model_path.suffix!='.keras':model.load_weights(str(model_path))
    data,audit=load_dataset(dataset)
    measurements={}
    for split in ('validation','test'):
        x,y=data[split]
        probabilities=np.concatenate([predict_numpy(model,x[i:i+128]) for i in range(0,len(y),128)])
        measurements[split]=classification_metrics(y,probabilities)
        np.savez_compressed(model_path.with_suffix('.'+split+'.npz'),labels=y,probabilities=probabilities)
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    result=dict(architecture=architecture,dataset=audit,checkpoint_sha256=digest,labels=list(LABELS),
                environment=device_info(),parameters=model.count_params(),note=note,**measurements)
    history={}
    history_path=model_path.parent/'history.csv'
    if history_path.exists():
        with history_path.open() as f:
            for row in csv.DictReader(f):
                for key,value in row.items():
                    if key!='epoch':history.setdefault(key,[]).append(float(value))
    result['epochs_completed']=len(history.get('loss',[]))
    model_path.with_suffix('.metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if model_path.suffix=='.keras':
        model_path.with_suffix('.json').write_text(json.dumps(dict(labels=list(LABELS),normalization='divide_by_255',input_shape=[48,48,1],architecture=architecture,checkpoint_sha256=digest),indent=2),encoding='utf-8')
    write_report(model_path.with_suffix('.report.html'),result,history)
    print(json.dumps({k:{m:v[m] for m in ('accuracy','macro_f1')} for k,v in measurements.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('dataset',type=Path);p.add_argument('--architecture',required=True);p.add_argument('--note',default='')
    a=p.parse_args();evaluate(a.model,a.dataset,a.architecture,a.note)
