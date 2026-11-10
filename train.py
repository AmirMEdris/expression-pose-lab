"""Train on FER2013; tune on validation, evaluate test only after checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')


def run(dataset, output, architecture='compact', epochs=20, seed=2026, batch_size=128, progress=print, cancel=None, learning_rate=None, weighting=None):
    from backend import select_backend,device_info,predict_numpy
    select_backend()
    import keras as k
    import cv2
    device=device_info()
    from architectures import create_model
    from dataset import load_dataset
    from metrics import classification_metrics
    from reporting import write_report
    from expression import LABELS
    if epochs<1 or batch_size<1: raise ValueError('Epochs and batch size must be positive')
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Choose a new empty output directory to preserve prior experiments')
    output.mkdir(parents=True,exist_ok=True)
    k.utils.set_random_seed(seed)
    progress('Runtime: '+str(device))
    progress('Reading dataset and checking exact duplicates…')
    data,audit=load_dataset(dataset,seed)
    (output/'dataset.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    progress('Dataset counts: '+str(audit['counts']))
    model=create_model(architecture)
    (output/'model-config.json').write_text(model.to_json(),encoding='utf-8')
    learning_rate=learning_rate or (.0003 if architecture=='residual' else .001)
    weighting=weighting or ('sqrt' if architecture=='residual' else 'balanced')
    optimizer=k.optimizers.AdamW(learning_rate=learning_rate,weight_decay=1e-4,clipnorm=1.) if architecture=='residual' else k.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'],jit_compile=False)
    x,y=data['train']
    class Batches(k.utils.PyDataset):
        def __init__(self,x,y,augment=False):
            super().__init__(workers=1,use_multiprocessing=False,max_queue_size=2)
            self.x=x;self.y=y;self.augment=augment;self.epoch=0;self.order=np.arange(len(y));self.on_epoch_end()
        def __len__(self):return (len(self.y)+batch_size-1)//batch_size
        def __getitem__(self,index):
            ids=self.order[index*batch_size:(index+1)*batch_size];images=self.x[ids].copy()
            if self.augment:
                rng=np.random.default_rng(seed+self.epoch*100000+index)
                for j in range(len(images)):
                    image=images[j,:,:,0]
                    if rng.random()<.5:image=np.fliplr(image)
                    dx,dy=rng.uniform(-2.88,2.88,2)
                    images[j,:,:,0]=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(48,48),borderMode=cv2.BORDER_REFLECT_101)
            return images,self.y[ids]
        def on_epoch_end(self):
            self.epoch+=1
            if self.augment:np.random.default_rng(seed+self.epoch).shuffle(self.order)
    training=Batches(x,y,True)
    validation=Batches(*data['validation'])
    class_counts=np.bincount(y,minlength=7)
    weights={i:float(len(y)/(7*max(1,n))) for i,n in enumerate(class_counts)}
    if weighting=='sqrt':weights={i:min(3.,float(np.sqrt(w))) for i,w in weights.items()}
    if weighting=='none':weights=None
    class Progress(k.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            progress(f'Epoch {epoch+1}/{epochs}: '+', '.join(f'{k}={v:.4f}' for k,v in (logs or {}).items()))
            (output/'progress.json').write_text(json.dumps(dict(epoch=epoch+1,epochs=epochs,metrics=logs)),encoding='utf-8')
        def on_train_batch_end(self,batch,logs=None):
            if cancel is not None and cancel.is_set(): self.model.stop_training=True
    callbacks=[k.callbacks.ModelCheckpoint(str(output/'best.keras'),monitor='val_loss',save_best_only=True),
               k.callbacks.ModelCheckpoint(str(output/'latest.keras')),
               k.callbacks.EarlyStopping(monitor='val_loss',patience=10 if architecture=='residual' else 6,restore_best_weights=True),
               k.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-6),
               k.callbacks.CSVLogger(str(output/'history.csv')),Progress()]
    start=time.monotonic()
    history=model.fit(training,validation_data=validation,epochs=epochs,class_weight=weights,callbacks=callbacks,verbose=0).history
    if cancel is not None and cancel.is_set():
        progress('Training cancelled. Any checkpoint is preserved; final test was not evaluated.');return None
    model=k.models.load_model(output/'best.keras',compile=False,safe_mode=True)
    progress('Selected checkpoint. Evaluating untouched test partition…')
    tx,ty=data['test']
    probabilities=np.concatenate([predict_numpy(model,tx[i:i+batch_size]) for i in range(0,len(tx),batch_size)])
    metrics=classification_metrics(ty,probabilities)
    np.savez_compressed(output/'test-predictions.npz',labels=ty,probabilities=probabilities)
    result=dict(architecture=architecture,dataset=audit,seed=seed,epochs_completed=len(history['loss']),requested_epochs=epochs,
                batch_size=batch_size,parameters=model.count_params(),training_seconds=time.monotonic()-start,
                learning_rate=learning_rate,weighting=weighting,optimizer=type(optimizer).__name__,
                checkpoint_sha256=hashlib.sha256((output/'best.keras').read_bytes()).hexdigest(),labels=list(LABELS),
                environment=dict(python=platform.python_version(),keras=k.__version__,runtime=device,numpy=np.__version__,platform=platform.platform()),
                augmentation='Horizontal flip, translation 6%; training only',class_weight=weights,test=metrics)
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'best.json').write_text(json.dumps(dict(labels=list(LABELS),input_shape=[48,48,1],normalization='divide_by_255',architecture=architecture,checkpoint_sha256=result['checkpoint_sha256']),indent=2),encoding='utf-8')
    write_report(output/'report.html',result,history)
    progress(f'Complete: accuracy {metrics["accuracy"]:.1%}; macro F1 {metrics["macro_f1"]:.1%}. Report: {output/"report.html"}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--architecture',choices=['compact','spatial','residual','baseline','original'],default='residual')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--seed',type=int,default=2026);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--learning-rate',type=float);p.add_argument('--weighting',choices=['none','sqrt','balanced'])
    a=p.parse_args();run(a.dataset,a.output,a.architecture,a.epochs,a.seed,a.batch_size,progress=lambda x:print(x,flush=True),learning_rate=a.learning_rate,weighting=a.weighting)


if __name__=='__main__':main()
