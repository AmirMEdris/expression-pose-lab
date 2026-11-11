"""Publishable demonstration: only procedural pixels, never dataset/camera images."""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def procedural_input(size=512):
    rng=np.random.default_rng(2041);y,x=np.mgrid[:size,:size].astype(np.float32)/size
    clouds=cv2.resize(rng.random((12,12),dtype=np.float32),(size,size),interpolation=cv2.INTER_CUBIC)
    clouds=cv2.GaussianBlur(clouds,(0,0),8)
    ribbons=.5+.5*np.sin(2*np.pi*(2.1*x+1.3*y+.22*np.sin(5*y)))
    glow=np.exp(-((x-.62)**2+(y-.4)**2)*5)
    rgb=np.stack([.18+.55*ribbons+.15*glow,.2+.5*clouds+.17*glow,.35+.4*(1-ribbons)+.12*glow],axis=-1)
    return cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2BGR)


def generate(model_path,output):
    from expression import ExpressionEstimator
    from model_dream import ModelDream
    from visualizations import FeatureVisualizer
    model_path=Path(model_path);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    model=ExpressionEstimator(model_path).model
    if model.input_shape[-1]!=3:raise ValueError('Gallery requires an RGB expression model')
    source=procedural_input();dream=ModelDream(model)
    for _ in range(20):art,_=dream.render_developed(source,strength=.012,steps=8,long_edge=512,preserve_color=True,preserve_structure=False)
    visualizer=FeatureVisualizer(model);maps=visualizer.render(source,layer=visualizer.layers[len(visualizer.layers)//2],mode='Activations')
    canvas=Image.new('RGB',(1600,710),'#0b1120');draw=ImageDraw.Draw(canvas)
    def font(size,bold=False):
        for path in ([Path('C:/Windows/Fonts/seguisb.ttf')] if bold else [Path('C:/Windows/Fonts/segoeui.ttf')])+[Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
            if path.exists():return ImageFont.truetype(str(path),size)
        return ImageFont.load_default(size=size)
    draw.text((34,24),'CANVASNET  /  ONE EXPRESSION MODEL',font=font(17,True),fill='#5eead4')
    draw.text((34,56),'Learned color. Visible patterns.',font=font(36,True),fill='#e2e8f0')
    panels=[(source,'PROCEDURAL INPUT','Seeded color fields. No photographs.'),(maps,'ACTUAL FEATURE RESPONSES','Sixteen channels; each tile has its own scale.'),(art,'RGB DEEPDREAM','Same expression weights. 160 pixel updates.')]
    for index,(pixels,title,subtitle) in enumerate(panels):
        left=34+index*522;image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB));image.thumbnail((488,488),Image.Resampling.LANCZOS)
        draw.rounded_rectangle((left,124,left+488,612),radius=12,fill='#111c30')
        canvas.paste(image,(left+(488-image.width)//2,124+(488-image.height)//2))
        draw.text((left,628),title,font=font(16,True),fill='#5eead4');draw.text((left,654),subtitle,font=font(15),fill='#a7b6c8')
    draw.text((34,686),'Generated patterns are not automatic labels for eyes, mouths, or emotion.',font=font(12),fill='#94a3b8')
    canvas.save(output)
    manifest=dict(checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),input='procedural color fields; seed 2041; no dataset or camera images',strength=.012,updates=160,keep_structure=False,rgb=True,feature_layer=visualizer.layers[len(visualizer.layers)//2])
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Synthetic-only gallery saved:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',default='docs/neural-gallery.png');a=p.parse_args();generate(a.model,a.output)
