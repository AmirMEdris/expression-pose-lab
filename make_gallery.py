"""Reproducible documentation art from a synthetic stimulus and actual model outputs."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from backend import select_backend
select_backend()
from expression import ExpressionEstimator
from visualizations import FeatureVisualizer
from model_dream import ModelDream


def main():
    root=Path(__file__).parent;out=root/'docs';out.mkdir(exist_ok=True)
    model=ExpressionEstimator(root/'models/expression-baseline.h5').model
    visualizer=FeatureVisualizer(model)
    y,x=np.mgrid[-1:1:320j,-1:1:384j]
    frame=np.uint8(np.clip(np.stack([.5+.25*np.sin(9*x+5*y*y),.5+.25*np.cos(10*y-3*x*x),.5+.25*np.sin(5*x+6*y)],axis=-1),0,1)*255)
    maps=visualizer.render(frame,visualizer.layers[2],'Activations')
    frames=[]
    dream=ModelDream(model)
    for _ in range(62):
        artwork,_=dream.render_developed(frame,.025,6)
        frames.append(artwork)
    def font(size):
        try:return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)
        except OSError:return ImageFont.load_default(size=size)
    canvas=Image.new('RGB',(1440,790),'#0b1120');draw=ImageDraw.Draw(canvas)
    draw.text((48,30),'EXPRESSION + POSE LAB',font=font(19),fill='#5eead4')
    draw.text((48,68),'A window into the network.',font=font(46),fill='#e2e8f0')
    draw.text((48,136),'Your expression model. Larger dreams. Source colors preserved.',font=font(20),fill='#94a3b8')
    tiles=[('01 / INPUT','Synthetic RGB pattern · no face data',frame),('02 / EXPRESSION FEATURES','16 grayscale filter responses',maps),('03 / YOUR MODEL DREAM','Your weights · source colors preserved',frames[-1])]
    for i,(title,caption,array) in enumerate(tiles):
        left=48+i*456
        draw.rounded_rectangle((left,200,left+432,732),radius=16,fill='#111c30',outline='#293548')
        draw.text((left+18,218),title,font=font(17),fill='#5eead4')
        image=Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB));image.thumbnail((396,410),Image.Resampling.LANCZOS)
        canvas.paste(image,(left+(432-image.width)//2,265+(410-image.height)//2))
        draw.text((left+18,692),caption,font=font(15),fill='#94a3b8')
    draw.text((48,753),'Artistic feature amplification is not an explanation of emotional state.',font=font(16),fill='#94a3b8')
    canvas.save(out/'neural-gallery.png')
    images=[Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((320,320)) for f in frames]
    images[0].save(out/'deepdream.gif',save_all=True,append_images=images[1:],duration=160,loop=0)


if __name__=='__main__':main()
