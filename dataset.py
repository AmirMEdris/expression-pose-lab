"""FER2013 CSV or train/test image folders with reproducible, disjoint partitions."""
from pathlib import Path
import csv
import hashlib
import zipfile
import cv2
import numpy as np
from expression import LABELS


def load_dataset(path, seed=2026):
    path=Path(path)
    splits={name:[] for name in ('train','validation','test')}
    if path.suffix.lower()=='.zip':
        # Decode directly from the archive, avoiding tens of thousands of slow file opens.
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                p=Path(name)
                if p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
                if len(p.parts)!=3 or p.parts[0] not in ('train','test') or p.parts[1] not in LABELS:
                    raise ValueError('ZIP needs train/class and test/class image folders')
                if p.parts[0]=='train':target='train'
                elif p.name.startswith('PublicTest_'):target='validation'
                elif p.name.startswith('PrivateTest_'):target='test'
                else:raise ValueError('ZIP test images must retain PublicTest_ / PrivateTest_ filenames')
                image=cv2.imdecode(np.frombuffer(archive.read(name),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape!=(48,48):raise ValueError('Invalid image: '+name)
                splits[target].append((image,LABELS.index(p.parts[1])))
        protocol='Original FER2013 partitions recovered from ZIP filenames: train / PublicTest validation / PrivateTest final test'
    elif path.is_file():
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not {'emotion','pixels','Usage'}<=set(reader.fieldnames or []):
                raise ValueError('CSV needs emotion, pixels, Usage columns with original FER2013 partitions')
            mapping={'Training':'train','PublicTest':'validation','PrivateTest':'test'}
            for row in reader:
                if row['Usage'] not in mapping: raise ValueError('Unknown FER2013 Usage partition')
                values=np.asarray(row['pixels'].split(),dtype=np.int32)
                label=int(row['emotion'])
                if values.size!=2304 or values.min()<0 or values.max()>255 or not 0<=label<7:
                    raise ValueError('Malformed FER2013 image or label')
                splits[mapping[row['Usage']]].append((values.astype(np.uint8).reshape(48,48),label))
        protocol='Original CSV: Training / PublicTest validation / PrivateTest final test'
    else:
        rng=np.random.default_rng(seed)
        test_files=[p for p in (path/'test').rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')]
        original_names=bool(test_files) and all(p.name.startswith(('PublicTest_','PrivateTest_')) for p in test_files) and any(p.name.startswith('PublicTest_') for p in test_files) and any(p.name.startswith('PrivateTest_') for p in test_files)
        for partition in ('train','test'):
            for label,name in enumerate(LABELS):
                files=sorted((path/partition/name).glob('*'))
                files=[p for p in files if p.suffix.lower() in ('.jpg','.jpeg','.png')]
                if len(files)<2: raise ValueError(f'Missing images in {path/partition/name}')
                order=rng.permutation(len(files))
                nvalidation=max(1,round(.15*len(files))) if partition=='train' and not original_names else 0
                for rank,i in enumerate(order):
                    image=cv2.imread(str(files[i]),cv2.IMREAD_GRAYSCALE)
                    if image is None or image.shape!=(48,48): raise ValueError(f'Expected 48x48 grayscale image: {files[i]}')
                    target='validation' if rank<nvalidation else partition
                    if original_names and partition=='test':
                        target='validation' if files[i].name.startswith('PublicTest_') else 'test'
                    splits[target].append((image,label))
        protocol='Kaggle train/test folders: seeded stratified 15% of train for validation; provided test held out'
        if original_names: protocol='Original FER2013 partitions recovered from filenames: train / PublicTest validation / PrivateTest final test'
    # Give test first priority, then validation; exclude exact duplicates from training.
    # Hash image pixels, never labels. Report removals; do not silently claim official counts.
    seen=set();duplicates={};result={};fingerprint=hashlib.sha256()
    for name in ('test','validation','train'):
        kept=[];removed=0
        for image,label in splits[name]:
            digest=hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen: removed+=1;continue
            seen.add(digest);kept.append((image,label))
            fingerprint.update(f'{name}:{digest}:{label}\n'.encode())
        if not kept: raise ValueError('Dataset partition is empty: '+name)
        x=np.stack([p[0] for p in kept])[...,None].astype(np.float32)/255.
        y=np.asarray([p[1] for p in kept],dtype=np.int32)
        if len(np.unique(y))!=7: raise ValueError(f'{name} must contain all seven classes')
        result[name]=(x,y);duplicates[name]=removed
    audit=dict(protocol=protocol,seed=seed,fingerprint=fingerprint.hexdigest(),
               duplicate_policy='Exact decoded pixels deduplicated: test priority, then validation, then train. No near-duplicate or identity audit.',
               duplicates_removed=duplicates,counts={k:len(v[1]) for k,v in result.items()},
               class_counts={k:np.bincount(v[1],minlength=7).tolist() for k,v in result.items()})
    return result,audit
