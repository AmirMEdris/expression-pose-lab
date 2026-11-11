"""Local, fixed-validation-input Dream comparisons during RGB training."""
from pathlib import Path
import argparse
import gc
import html
import json
import cv2
import numpy as np
import torch
from canvasnet import CanvasNet,export_keras,LABELS
from training_progress import Progress


def run(folder,dataset,epoch):
    folder=Path(folder);dataset=Path(dataset);progress=Progress(folder)
    torch.set_num_threads(4);torch.manual_seed(2026)
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    network=CanvasNet().eval();network.load_state_dict(saved['model'])
    preview_path=folder/f'preview-epoch-{epoch:02d}.keras'
    model=export_keras(network,preview_path,{'status':'Training preview','checkpoint_epoch':saved['epoch'],'crop_margin':.1})
    from model_dream import ModelDream
    audit=json.loads((dataset/'prepared.json').read_text())
    images=np.memmap(dataset/'faces-224.rgb',dtype=np.uint8,mode='r',shape=(audit['cached_count'],224,224,3))
    with np.load(dataset/'splits.npz') as data:ids=data['val'].copy();labels=data['labels'].copy()
    rng=np.random.default_rng(2026);rng.shuffle(ids)
    selected=[int(next(i for i in ids if labels[i]==label)) for label in (3,4,5,6)]
    out=folder/'dream-previews';out.mkdir(exist_ok=True);cards=[];total=4*2*160;done=0
    progress.update(phase='Checking learned RGB Dreams',done=0,total=total,message='Fixed validation examples and settings. No test images used; beauty is judged visually.')
    for number,index in enumerate(selected):
        frame=cv2.cvtColor(images[index],cv2.COLOR_RGB2BGR);name=f'epoch-{epoch:02d}-face-{number}'
        cv2.imwrite(str(out/(name+'-input.png')),frame)
        paths=[name+'-input.png']
        for guided in (True,False):
            dream=ModelDream(model)
            for _ in range(20):
                image,_=dream.render_developed(frame,strength=.012,steps=8,long_edge=224,preserve_color=True,preserve_structure=guided)
                done+=8;progress.update(done=done,total=total,progress_text=f'{done}/{total} preview updates · selected checkpoint epoch {saved["epoch"]}')
            path=name+('-guided.png' if guided else '-free.png');cv2.imwrite(str(out/path),image);paths.append(path)
            del dream;gc.collect()
        cards.append(dict(index=index,label=LABELS[labels[index]],images=paths))
    record=dict(training_epoch=epoch,checkpoint_epoch=saved['epoch'],checkpoint_file=preview_path.name,
                checkpoint_sha256=json.loads(preview_path.with_suffix('.json').read_text(encoding='utf-8'))['checkpoint_sha256'],
                selected_validation_indices=selected,strength=.012,updates=160,cards=cards)
    (out/f'epoch-{epoch:02d}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    sections=[]
    for path in sorted(out.glob('epoch-*.json'),reverse=True):
        data=json.loads(path.read_text());rows=[]
        for card in data['cards']:
            figures=''.join(f'<figure><img src="dream-previews/{html.escape(image)}"><figcaption>{caption}</figcaption></figure>' for image,caption in zip(card['images'],['Unmodified validation crop','Dream · keep structure','Dream · unrestricted']))
            rows.append(f'<h3>Dataset label: {html.escape(card["label"])}</h3><div class="row">{figures}</div>')
        sections.append(f'<section><h2>After epoch {data["training_epoch"]} · selected weights from epoch {data["checkpoint_epoch"]}</h2>'+''.join(rows)+'</section>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CanvasNet · Dream progress</title><style>body{background:#0b1120;color:#e2e8f0;font:16px/1.6 system-ui;margin:40px auto;max-width:1100px;padding:20px}h1{font-size:38px}h2{color:#5eead4}.row{display:flex;gap:18px}figure{margin:0;flex:1}img{width:100%;border-radius:12px}figcaption{color:#94a3b8;font-size:14px}section{border-top:1px solid #334155;margin-top:40px}h3{font-size:15px;font-weight:500}</style><h1>Watch the same model learn to Dream.</h1><p>Identical validation examples, RGB input, and Dream settings at each checkpoint. Both expression recognition and Dream use the displayed checkpoint’s weights. The checkpoint is selected by validation macro F1, not visual appeal.</p><p>These are real optimization outputs. Smoothing and optional source constraints affect their appearance. Patterns are not automatic eye/mouth labels. Dataset images stay local; no camera images are used. Reload this page after the next preview stage.</p>'''+''.join(sections)
    (folder/'dream-progress.html').write_text(page,encoding='utf-8');progress.log(f'Dream comparison saved after epoch {epoch}; selected checkpoint epoch {saved["epoch"]}.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',default='experiments/canvas-rgb-v1');p.add_argument('--dataset',default='data/expw');p.add_argument('--epoch',type=int,required=True);a=p.parse_args();run(a.folder,a.dataset,a.epoch)
