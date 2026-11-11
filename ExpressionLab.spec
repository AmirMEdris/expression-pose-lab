# Build on Windows: python -m PyInstaller --noconfirm ExpressionLab.spec
import os
from pathlib import Path
os.environ['KERAS_BACKEND']='torch'
from PyInstaller.utils.hooks import collect_data_files,copy_metadata,collect_submodules
datas=[('models/expression-baseline.h5','models'),('models/pose_landmarker_lite.task','models')]
datas+=[('docs/evaluation/expression-baseline.metrics.json','models'),('docs/evaluation/expression-baseline.report.html','models')]
for path in Path('models').glob('expression-rgb.*'):
    if path.suffix in ('.keras','.json','.html','.npz'):datas.append((str(path),'models'))
    elif path.is_dir() and path.name.endswith('.filters.assets'):datas.append((str(path),'models/'+path.name))
datas+=collect_data_files('mediapipe')+collect_data_files('cv2')
for package in ('torch','keras','mediapipe','numpy','pillow','h5py'):
    datas+=copy_metadata(package)
a=Analysis(['desktop.py'],pathex=[],binaries=[],datas=datas,
    hiddenimports=['PIL.ImageGrab','PIL.ImageTk','torch','keras','selftest']+collect_submodules('keras.src.backend.torch',filter=lambda name: not name.endswith('_test')),
    hookspath=[],hooksconfig={'matplotlib':{'backends':['Agg']}},runtime_hooks=[],excludes=['IPython','pytest','pandas','sklearn','tensorflow','jax','jupyter'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='Expression Lab',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='Expression Lab')
