"""From-scratch GPU training for CanvasNet; local, resumable, visible progress."""
from pathlib import Path
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from canvasnet import CanvasNet,IMAGE_SIZE
from metrics import classification_metrics
from training_progress import Progress


class Faces(Dataset):
    def __init__(self,folder,split,augment=False):
        self.folder=Path(folder);self.audit=json.loads((self.folder/'prepared.json').read_text())
        with np.load(self.folder/'splits.npz') as data:self.ids=data[split].copy();self.labels=data['labels'].copy()
        self.images=None;self.augment=augment
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        if self.images is None:self.images=np.memmap(self.folder/'faces-224.rgb',dtype=np.uint8,mode='r',shape=(self.audit['cached_count'],IMAGE_SIZE,IMAGE_SIZE,3))
        i=int(self.ids[index]);image=np.array(self.images[i],copy=True)
        if self.augment:
            if random.random()<.5:image=cv2.flip(image,1)
            matrix=cv2.getRotationMatrix2D((IMAGE_SIZE/2,IMAGE_SIZE/2),random.uniform(-12,12),random.uniform(.9,1.1))
            matrix[:,2]+=np.array([random.uniform(-.06,.06),random.uniform(-.06,.06)])*IMAGE_SIZE
            image=cv2.warpAffine(image,matrix,(IMAGE_SIZE,IMAGE_SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            if random.random()<.1:
                side=random.randint(12,45);x=random.randrange(IMAGE_SIZE-side);y=random.randrange(IMAGE_SIZE-side)
                image[y:y+side,x:x+side]=image.mean((0,1)).astype(np.uint8)
        return torch.from_numpy(image.transpose(2,0,1).copy()),int(self.labels[i])


def worker_init(_):cv2.setNumThreads(1)


def color_augment(x):
    shape=(len(x),1,1,1)
    brightness=torch.empty(shape,device=x.device).uniform_(.8,1.2)
    contrast=torch.empty(shape,device=x.device).uniform_(.8,1.2)
    saturation=torch.empty(shape,device=x.device).uniform_(.75,1.25)
    gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
    x=gray+(x-gray)*saturation;mean=x.mean((1,2,3),keepdim=True)
    return ((mean+(x-mean)*contrast)*brightness).clamp_(0,1)


def atomic_save(data,path):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)


def run(args):
    if not torch.cuda.is_available():raise RuntimeError('This training run requires CUDA; no silent CPU fallback.')
    output=Path(args.output);progress=Progress(output);torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.resume and all((output/name).exists() for name in ('metrics.json','best.keras','report.html')):
        result=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
        progress.update(phase='Complete',done=1,total=1,message='This run already completed. Reusing its saved held-out evaluation.',progress_text=f'Test accuracy {result["test"]["accuracy"]:.1%} · macro F1 {result["test"]["macro_f1"]:.1%}')
        progress.log('Completed run detected; no additional training or test evaluation performed.');return
    seed=2026;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    model=CanvasNet().cuda().to(memory_format=torch.channels_last)
    data={name:Faces(args.dataset,name,name=='train') for name in ('train','val','test')}
    loaders={name:DataLoader(dataset,batch_size=args.batch_size,shuffle=name=='train',num_workers=args.workers,
               pin_memory=True,persistent_workers=args.workers>0,worker_init_fn=worker_init,drop_last=False)
             for name,dataset in data.items()}
    audit=data['train'].audit
    config=dict(architecture='CanvasNet RGB',pretrained=False,epochs=args.epochs,batch_size=args.batch_size,
        seed=seed,learning_rate=args.lr,weight_decay=.0005,label_smoothing=.05,patience=args.patience,
        checkpoint_selection='validation macro F1',precision='CUDA bfloat16 autocast; float32 master weights',
        parameters=sum(p.numel() for p in model.parameters()),input_shape=[224,224,3],
        spatial_dropout=[.05,.1,.15,.2],head_dropout=.35,split_sha256=audit['split_sha256'],
        device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),workers=args.workers)
    history=[];best=-1.;best_epoch=0;start_epoch=0
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0005,fused=True)
    counts=np.array(audit['class_counts']['train']);weights=np.sqrt(counts.sum()/(7*np.maximum(counts,1)));weights=np.minimum(weights,3);weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device='cuda'),label_smoothing=.05)
    if args.resume:
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        if saved['config']!=config:raise ValueError('Resume configuration does not match checkpoint')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);history=saved['history']
        best=saved['best'];best_epoch=saved['best_epoch'];start_epoch=saved['epoch']
    elif (output/'last.pt').exists():raise ValueError('Checkpoint exists. Use --resume or a new output folder.')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (output/'dataset.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    progress.log(f'From-scratch CanvasNet: {config["parameters"]:,} parameters on {config["device"]}; dropout active for training.')
    progress.update(phase='Training custom RGB model',epochs=args.epochs,device=config['device'],history=history)
    def evaluate(name,amp=True):
        model.eval();all_probs=[];all_y=[];loss_sum=0;seen=0
        progress.update(phase='Validating' if name=='val' else 'Final held-out evaluation',message='Dropout disabled; no weight updates. '+('Validation selects the best checkpoint.' if name=='val' else 'Measuring the selected checkpoint on the held-out test split.'))
        with torch.inference_mode():
            for step,(images,labels) in enumerate(loaders[name]):
                images=images.cuda(non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last);labels=labels.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):logits=model(images)
                logits=logits.float();loss_sum+=float(nn.functional.cross_entropy(logits,labels,reduction='sum'));seen+=len(labels)
                all_probs.append(logits.softmax(1).cpu().numpy());all_y.append(labels.cpu().numpy())
                if step%20==0:progress.update(phase='Validating' if name=='val' else 'Final held-out evaluation',done=step+1,total=len(loaders[name]),progress_text=f'{seen:,} / {len(data[name]):,} faces')
        probs=np.concatenate(all_probs);labels=np.concatenate(all_y)
        return classification_metrics(labels,probs),loss_sum/seen,labels,probs
    for epoch in range(start_epoch,args.epochs):
        torch.manual_seed(seed+epoch);model.train();running_loss=0.;correct=0;seen=0;epoch_start=time.monotonic();last=0
        progress.update(phase='Training custom RGB model',epoch=epoch+1,done=0,total=len(loaders['train']),message='Learning RGB expression features from scratch. Final test split remains untouched.')
        for step,(images,labels) in enumerate(loaders['train']):
            fraction=epoch+step/len(loaders['train']);warmup=3
            lr=args.lr*(.1+.9*fraction/warmup) if fraction<warmup else args.lr*(.05+.95*.5*(1+math.cos(math.pi*(fraction-warmup)/max(1,args.epochs-warmup))))
            for group in optimizer.param_groups:group['lr']=lr
            images=images.cuda(non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last);labels=labels.cuda(non_blocking=True);images=color_augment(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images);loss=criterion(logits,labels)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with torch.no_grad():
                running_loss+=float(nn.functional.cross_entropy(logits.float(),labels,reduction='sum'));correct+=int((logits.argmax(1)==labels).sum());seen+=len(labels)
            if time.monotonic()-last>2:
                last=time.monotonic();speed=seen/max(.1,last-epoch_start)
                progress.update(done=step+1,total=len(loaders['train']),progress_text=f'{seen:,} / {len(data["train"]):,} faces · {speed:.0f} faces/s · loss {running_loss/seen:.3f}',
                    gpu_memory=f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak',message=f'Epoch {epoch+1}; about {(len(data["train"])-seen)/max(1,speed)/60:.1f} min of training batches left. Dropout enabled.')
        metrics,val_loss,_,_=evaluate('val')
        row=dict(epoch=epoch+1,train_loss=running_loss/seen,train_accuracy=correct/seen,val_loss=val_loss,
                 val_accuracy=metrics['accuracy'],val_macro_f1=metrics['macro_f1'],learning_rate=lr,seconds=time.monotonic()-epoch_start)
        history.append(row)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];best_epoch=epoch+1
            atomic_save(dict(model=model.state_dict(),config=config,epoch=best_epoch,validation=metrics),output/'best.pt')
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,epoch=epoch+1,history=history,best=best,best_epoch=best_epoch),output/'last.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        progress.update(history=history,val_accuracy=metrics['accuracy'],best_epoch=best_epoch,best_macro_f1=best)
        progress.log(f'Epoch {epoch+1}/{args.epochs}: validation accuracy {metrics["accuracy"]:.2%}, macro F1 {metrics["macro_f1"]:.2%}; best epoch {best_epoch}; {row["seconds"]/60:.1f} min.')
        if (epoch+1)%10==0:
            preview=subprocess.run([sys.executable,str(Path(__file__).with_name('preview_canvas.py')),'--folder',str(output),'--dataset',str(args.dataset),'--epoch',str(epoch+1)])
            if preview.returncode:progress.log('Dream preview failed; training checkpoint is intact. See worker log.')
        if epoch+1>=20 and epoch+1-best_epoch>=args.patience:
            progress.log('Early stopping: validation macro F1 has not improved.');break
    best_checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(best_checkpoint['model']);model.eval()
    # Exactly one final test pass, after checkpoint selection; use float32.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    test,test_loss,labels,probabilities=evaluate('test',amp=False)
    np.savez_compressed(output/'predictions.npz',labels=labels,probabilities=probabilities,indices=data['test'].ids)
    result=dict(architecture='CanvasNet RGB · trained from scratch',dataset=audit,test=test,best_epoch=best_epoch,epochs_completed=len(history),
                validation=best_checkpoint['validation'],test_loss=test_loss,config=config,
                note='Checkpoint selected by validation macro F1. One final test pass after selection. Custom ExpW source/duplicate-group split, not identity-disjoint.')
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    from canvasnet import export_keras
    progress.update(phase='Exporting shared model',message='Copying the selected classifier weights for live inference and RGB DeepDream.',done=0,total=1)
    exported=export_keras(model,output/'best.keras',dict(dataset=audit['name'],test_accuracy=test['accuracy'],test_macro_f1=test['macro_f1'],best_epoch=best_epoch,crop_margin=.10))
    images,_=next(iter(loaders['val']));images=images[:8].cuda().float()/255
    with torch.no_grad():
        expected=model(images).softmax(1).cpu().numpy();actual=exported(images.permute(0,2,3,1),training=False).detach().cpu().numpy()
    error=float(np.max(np.abs(expected-actual)));result['export_max_probability_error']=error
    result['checkpoint_sha256']=json.loads((output/'best.json').read_text(encoding='utf-8'))['checkpoint_sha256']
    if not np.allclose(expected,actual,atol=2e-4,rtol=1e-4):raise RuntimeError(f'Keras export mismatch: {error}')
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    from reporting import write_report
    chart_history={key:[row[source] for row in history] for key,source in [('loss','train_loss'),('val_loss','val_loss'),('accuracy','train_accuracy'),('val_accuracy','val_accuracy')]}
    write_report(output/'report.html',result,chart_history)
    progress.update(phase='Complete',done=1,total=1,progress_text=f'Test accuracy {test["accuracy"]:.1%} · macro F1 {test["macro_f1"]:.1%}',message=f'Best epoch {best_epoch}. Classifier and Dream share the same exported weights. Report saved beside checkpoint.')
    progress.log(f'Complete: final test accuracy {test["accuracy"]:.2%}, macro F1 {test["macro_f1"]:.2%}; export parity error {error:.3g}.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='data/expw');parser.add_argument('--output',default='experiments/canvas-rgb-v1')
    parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=.0008);parser.add_argument('--patience',type=int,default=8);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:run(args)
    except Exception as error:Progress(args.output).update(phase='Failed',message=str(error));raise
