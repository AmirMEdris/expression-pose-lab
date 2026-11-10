"""Local, offline filter explorer. All displayed inputs are lossless 8-bit images."""
import base64
import io
import json
from pathlib import Path
import numpy as np


def atlas(images,columns=64):
    from PIL import Image
    images=np.asarray(images,dtype=np.uint8)
    canvas=np.zeros((((len(images)+columns-1)//columns)*48,columns*48),np.uint8)
    for i,image in enumerate(images):canvas[(i//columns)*48:(i//columns+1)*48,(i%columns)*48:(i%columns+1)*48]=image
    buffer=io.BytesIO();Image.fromarray(canvas).save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def write_report(path,result,real,dreams):
    template=Path(__file__).with_name('filter_report.html').read_text(encoding='utf-8')
    payload=dict(result,real_atlas=atlas(real),dream_atlas=atlas(dreams))
    encoded=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
    Path(path).write_text(template.replace('__EVIDENCE_JSON__',encoded),encoding='utf-8')
