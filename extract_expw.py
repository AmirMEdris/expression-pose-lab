"""Extract the official split archive after validating every entry path."""
from pathlib import Path
import time
import subprocess
import os
import multivolumefile
import py7zr
from py7zr.callbacks import ExtractCallback
from training_progress import Progress


def extract(folder='data/expw', experiment='experiments/canvas-rgb-v1'):
    root=Path(folder).resolve(); progress=Progress(experiment)
    if (root/'extraction-complete.txt').exists():
        progress.log('ExpW extraction already complete.'); return
    progress.update(phase='Extracting ExpW',message='Checking archive paths before extracting original images.',done=0,total=1,progress_text='Reading archive index')
    class Callback(ExtractCallback):
        def __init__(self,total): self.done=0; self.total=total; self.last=0
        def report_start_preparation(self): pass
        def report_start(self,*args): pass
        def report_update(self,*args): pass
        def report_end(self,path,size):
            self.done+=1
            if time.monotonic()-self.last>1:
                self.last=time.monotonic(); progress.update(done=self.done,total=self.total,progress_text=f'{self.done:,} / {self.total:,} images',message='Extracting the authors’ original images to local storage.')
        def report_warning(self,message): progress.log(message)
        def report_postprocess(self): pass
    try:
        with multivolumefile.open(root/'origin.7z',mode='rb') as parts:
            with py7zr.SevenZipFile(parts,'r') as archive:
                entries=archive.list()
                for entry in entries:
                    target=(root/entry.filename).resolve()
                    if not target.is_relative_to(root) or entry.is_symlink:
                        raise ValueError('Unsafe archive entry: '+entry.filename)
                progress.log(f'Validated {len(entries):,} archive entries.')
                native=Path('data/tools/7zr.exe').resolve()
                if not native.exists():
                    archive.extractall(path=root,callback=Callback(len(entries)))
                else:
                    # This folder contains only this task's ExpW extraction.
                    # Every archive entry was checked above before allowing writes.
                    with (Path(experiment)/'7zip.log').open('w',encoding='utf-8') as log:
                        worker=subprocess.Popen([str(native),'x',str(root/'origin.7z.001'),'-o'+str(root),'-y','-aoa'],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        while worker.poll() is None:
                            count=sum(1 for _ in os.scandir(root/'origin')) if (root/'origin').exists() else 0
                            progress.update(done=count,total=len(entries),progress_text=f'{count:,} / {len(entries):,} images',message='Extracting original photos with the native 7-Zip engine.')
                            time.sleep(2)
                        if worker.returncode:raise RuntimeError('7-Zip extraction failed; see 7zip.log')
        (root/'extraction-complete.txt').write_text('Archive extraction and CRC checks completed.\n',encoding='utf-8')
        progress.update(phase='Extraction complete',message='Original RGB images ready for crop and duplicate audit.',done=1,total=1,progress_text='Complete')
        progress.log('ExpW extraction complete.')
    except Exception as error:
        progress.update(phase='Failed',message=str(error)); raise


if __name__=='__main__': extract()
