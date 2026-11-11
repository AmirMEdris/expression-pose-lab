"""Audit native RGB crops and build a reproducible source/duplicate-group split."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import hashlib
import json
import time
import cv2
import numpy as np
from training_progress import Progress

SIZE=224


def read_crop(item,root):
    fields=item.split();name=fields[0]
    top,left,right,bottom=map(int,fields[2:6]);label=int(fields[7])
    path=(root/'origin'/name).resolve()
    if not path.is_relative_to((root/'origin').resolve()):return None,'invalid_path'
    try:
        raw=path.read_bytes();image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None:return None,'unreadable'
        h,w=image.shape[:2];left=max(0,left);top=max(0,top);right=min(w,right+1);bottom=min(h,bottom+1)
        native=min(right-left,bottom-top)
        if native<64:return None,'native_face_below_64px'
        # Keep modest context; use the same expansion in live inference.
        margin=round(.10*max(right-left,bottom-top))
        crop=image[max(0,top-margin):min(h,bottom+margin),max(0,left-margin):min(w,right+margin)]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        if float(gray.std())<5:return None,'nearly_constant'
        crop=cv2.resize(crop,(SIZE,SIZE),interpolation=cv2.INTER_AREA if min(crop.shape[:2])>=SIZE else cv2.INTER_CUBIC)
        crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
        dct=cv2.dct(small)[:8,:8].reshape(-1);bits=dct>np.median(dct[1:]);bits[0]=False
        phash=int.from_bytes(np.packbits(bits).tobytes(),'big')
        row=dict(source=name,face_id=int(fields[1]),label=label,native_min_side=native,
                 source_sha256=hashlib.sha256(raw).hexdigest(),crop_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                 phash=phash,is_color=bool(np.abs(crop[:,:,0].astype(float)-crop[:,:,1]).mean()>1.5 or np.abs(crop[:,:,1].astype(float)-crop[:,:,2]).mean()>1.5))
        return (crop,row),None
    except (OSError,ValueError,cv2.error):return None,'read_or_decode_error'


def run(folder='data/expw',experiment='experiments/canvas-rgb-v1'):
    from sklearn.model_selection import StratifiedGroupKFold
    root=Path(folder).resolve();progress=Progress(experiment);cv2.setNumThreads(1)
    if (root/'prepared.json').exists():progress.log('Prepared ExpW cache already exists.');return
    if not (root/'extraction-complete.txt').exists():raise RuntimeError('Complete extract_expw.py first')
    lines=(root/'label.lst').read_text(encoding='utf-8-sig').splitlines();lines=[line for line in lines if line.strip()]
    rows=[];rejects=Counter();start=time.monotonic();last=0
    progress.update(phase='Preparing RGB faces',message='Checking native detail, corrupt images, and source duplicates.',done=0,total=len(lines),progress_text='Reading original photos')
    with (root/'faces-224.rgb').open('wb') as cache,ThreadPoolExecutor(max_workers=8) as pool:
        def results():
            for start in range(0,len(lines),128):
                yield from pool.map(lambda item:read_crop(item,root),lines[start:start+128])
        for i,(result,reason) in enumerate(results()):
            if result is None:rejects[reason]+=1
            else:
                crop,row=result;row['index']=len(rows);rows.append(row);cache.write(crop.tobytes())
            if time.monotonic()-last>1:
                last=time.monotonic();progress.update(done=i+1,total=len(lines),progress_text=f'{i+1:,} / {len(lines):,} labels · {len(rows):,} usable RGB crops')
    # Union original images, exact image duplicates, exact crops, and very close
    # crop pHashes. No face identity labels exist: this is NOT identity-disjoint.
    n=len(rows);parent=list(range(n));sizes=[1]*n
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a=find(a);b=find(b)
        if a==b:return
        if sizes[a]<sizes[b]:a,b=b,a
        parent[b]=a;sizes[a]+=sizes[b]
    progress.update(phase='Checking duplicate groups',message='Keeping source images and detected near-duplicates within one split.',done=0,total=n)
    sources={};image_hashes={};crop_hashes=defaultdict(list);bands=defaultdict(list);near=0
    for i,row in enumerate(rows):
        for mapping,key in ((sources,row['source']),(image_hashes,row['source_sha256'])):
            if key in mapping:union(i,mapping[key])
            else:mapping[key]=i
        crop_hashes[row['crop_sha256']].append(i);value=row['phash'];candidates=set()
        for band in range(4):candidates.update(bands[(band,(value>>(band*16))&65535)])
        for other in candidates:
            if (value^rows[other]['phash']).bit_count()<=3:union(i,other);near+=1
        for band in range(4):bands[(band,(value>>(band*16))&65535)].append(i)
        if i%1000==0:progress.update(done=i,total=n,progress_text=f'{i:,} / {n:,} crops checked')
    excluded=set();conflicts=0;duplicates=0
    for ids in crop_hashes.values():
        if len(ids)<2:continue
        for i in ids[1:]:union(ids[0],i)
        if len({rows[i]['label'] for i in ids})>1:excluded.update(ids);conflicts+=len(ids)
        else:excluded.update(ids[1:]);duplicates+=len(ids)-1
    usable=np.array([i for i in range(n) if i not in excluded]);labels=np.array([r['label'] for r in rows],dtype=np.int64)
    groups=np.array([find(i) for i in usable]);splitter=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=2026)
    folds=list(splitter.split(usable,labels[usable],groups));test=usable[folds[0][1]];val=usable[folds[1][1]]
    train=np.setdiff1d(usable,np.concatenate([test,val]));split={'train':train,'val':val,'test':test}
    group_sets={k:set(find(int(i)) for i in indices) for k,indices in split.items()}
    assert not(group_sets['train']&group_sets['val'] or group_sets['train']&group_sets['test'] or group_sets['val']&group_sets['test'])
    np.savez(root/'splits.npz',labels=labels,**split)
    with (root/'crops.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    native=np.array([rows[i]['native_min_side'] for i in usable]);with_color=sum(rows[i]['is_color'] for i in usable)
    with (root/'splits.npz').open('rb') as stream:split_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    audit=dict(name='ExpW RGB · custom grouped split',source='https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html',
        raw_label_count=len(lines),cached_count=n,usable_count=len(usable),size=SIZE,channels=3,color_space='RGB',
        min_native_face=64,crop_margin=.10,bbox_convention='right/bottom interpreted as inclusive; clamp to image',
        rejected=dict(rejects),exact_duplicate_crops_removed=duplicates,conflicting_exact_crops_removed=conflicts,
        near_duplicate_pairs=near,phash_hamming_max=3,split_seed=2026,split_sha256=split_sha,
        counts={k:len(v) for k,v in split.items()},class_counts={k:np.bincount(labels[v],minlength=7).tolist() for k,v in split.items()},
        native_side_percentiles=dict(zip(['p10','p50','p90'],np.percentile(native,[10,50,90]).tolist())),
        color_fraction=with_color/max(1,len(usable)),group_count=len(set(groups)),max_group_size=max(Counter(groups).values()),
        limitations=['Custom 80/10/10 split; not an official ExpW benchmark.','Source and detected duplicates grouped, but identities may cross splits.',
                    'pHash detects some near-duplicates, not all; labels may be noisy.','Upscaling does not recover missing original detail.',
                    'No explicit redistribution license found; dataset and trained weights stay local.'])
    (root/'prepared.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');Path(experiment,'dataset.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    progress.update(phase='Dataset ready',message=f'{len(usable):,} usable faces; source and duplicate groups do not cross splits.',done=1,total=1,progress_text=str(audit['counts']))
    progress.log(f'RGB preparation completed in {(time.monotonic()-start)/60:.1f} minutes. '+str(audit['counts']))


if __name__=='__main__':
    try:run()
    except Exception as error:Progress('experiments/canvas-rgb-v1').update(phase='Failed',message=str(error));raise
