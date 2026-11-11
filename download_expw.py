"""Download ExpW from the authors' public Drive IDs; resume interrupted parts."""
from pathlib import Path
import hashlib
import json
import time
from training_progress import Progress

FILES=[
    ('readme.txt','1sGqWNd9U_PgKYrjkiF0uCEk3yb_szMeJ',575),
    ('label.lst','1jTPlwK7J4E5VD_mPPYEv_UTfJBVECS-W',4641240),
    ('origin.7z.001','1GJenevMXHC1UKsrc63YkmXMMYN-fWHXA',1048576000),
    ('origin.7z.002','1RR_EQ1a9X-UUlKqmYYTXNynFtfjYU4Re',1048576000),
    ('origin.7z.003','19lC9EQsDP1UWUgeN8hMthy3Q9hc9xMjN',1048576000),
    ('origin.7z.004','1gcMYhAty1EGzG4QXrMKJw39tuya4u7gS',1048576000),
    ('origin.7z.005','1cZbHg-L8AP8vhwB-K5Th8Jj0s4d8wvJ7',1048576000),
    ('origin.7z.006','1xEtQoe5zPBkGt9YDjqhyayzecHhsQH5-',1048576000),
    ('origin.7z.007','1rJbpGr32IltJ-DiJtSYaSTHPmzh4jW-j',1048576000),
    ('origin.7z.008','1B7ghcYnI3aJjiFSvzx3yVH7hIvK8WSkf',773544419),
]


def run(folder,experiment):
    import gdown
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);progress=Progress(experiment)
    total=sum(size for _,_,size in FILES);finished=0;manifest=[]
    try:
        for name,file_id,size in FILES:
            path=folder/name;last=0
            def update(current,expected):
                nonlocal last
                if time.monotonic()-last<.5 and current<size:return
                last=time.monotonic();done=finished+min(current,size)
                progress.update(phase='Downloading ExpW',message=f'Official archive: {name}',done=done,total=total,progress_text=f'{done/1e9:.2f} / {total/1e9:.2f} GB')
            progress.log('Downloading '+name)
            if not path.exists() or path.stat().st_size!=size:
                gdown.download(id=file_id,output=str(path),quiet=True,use_cookies=False,resume=True,progress=update,timeout=60,retries=3)
            if path.stat().st_size!=size:raise ValueError('Unexpected byte count for '+name)
            with path.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            manifest.append(dict(name=name,drive_id=file_id,bytes=size,sha256=digest));update(size,size);finished+=size
        (folder/'download-manifest.json').write_text(json.dumps(dict(source='https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html',files=manifest),indent=2),encoding='utf-8')
        progress.update(phase='Download complete',message='Archives verified by byte count and recorded SHA-256. Preparing image extraction next.',done=total,total=total,progress_text=f'{total/1e9:.2f} / {total/1e9:.2f} GB')
        progress.log('All official ExpW parts downloaded.')
    except Exception as error:
        progress.update(phase='Failed',message=str(error));progress.log('Download error: '+str(error));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folder',default='data/expw');p.add_argument('--experiment',default='experiments/canvas-rgb-v1')
    a=p.parse_args();run(a.folder,a.experiment)
