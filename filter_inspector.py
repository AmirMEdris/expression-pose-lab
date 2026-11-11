"""Read a selected channel on the classifier's actual input; no pixel ascent."""
import cv2
import numpy as np
from expression import preprocess
from filter_geometry import model_support,support_box
from dream_selection import filter_activation


def response_summary(values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected a finite spatial response map')
    low=float(values.min());high=float(values.max())
    varied=high-low>max(1e-7,max(abs(low),abs(high))*1e-6)
    peak=tuple(int(v) for v in np.unravel_index(values.argmax(),values.shape)) if varied else None
    normalized=(values-low)/(high-low) if varied else np.zeros_like(values)
    return dict(low=low,high=high,peak=peak,normalized=normalized)


def overlay_response(gray,normalized,geometry):
    """Locate grid values at their receptive-field centers, then interpolate.

    This is a coarse response display, not pixel attribution. Exact support
    of a selected unit is shown separately as a box on the original input.
    """
    yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]].astype(np.float32)
    gy,gx=geometry
    map_x=(xx-(gx[1]+gx[2])/2)/gx[0]
    map_y=(yy-(gy[1]+gy[2])/2)/gy[0]
    heat=cv2.remap(normalized,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    colors=cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_INFERNO)
    original=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(original,.55,colors,.45,0)


class FilterInspector:
    def __init__(self,visualizer):
        self.visualizer=visualizer
        self.geometry=model_support(visualizer.model)

    def render(self,frame,layer,channel):
        from backend import predict_numpy
        values=preprocess(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        activation=predict_numpy(self.visualizer.extractor(layer),values)
        selected=filter_activation(activation,channel)[0,:,:,0]
        summary=response_summary(selected)
        gray=np.rint(values[0,:,:,0]*255).astype(np.uint8)
        original=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        reference=cv2.resize(original,(480,480),interpolation=cv2.INTER_NEAREST)
        box=None
        if summary['peak'] is None:
            image=original
            explanation='No response anywhere on this face.' if summary['high']==0 else 'Uniform response: no distinct strongest region.'
        else:
            box=support_box(self.geometry[layer],*summary['peak'])
            x0,y0,x1,y1=box
            cv2.rectangle(reference,(x0*10,y0*10),(x1*10-1,y1*10-1),(65,220,255),3)
            image=overlay_response(gray,summary['normalized'],self.geometry[layer])
            explanation='Yellow = higher response; purple = lower. Colors are relative, not confidence.'
        number=self.visualizer.layers.index(layer)+1
        caption=f'Layer {number} / Filter {channel} on the unmodified input. {explanation} Range {summary["low"]:.3g} to {summary["high"]:.3g}.'
        reference_caption='Exact 48×48 input, enlarged. '+('Box = the region available to a strongest unit; not an eye/mouth outline.' if box else 'No distinct response region to box.')
        return dict(image=cv2.resize(image,(480,480),interpolation=cv2.INTER_NEAREST),reference=reference,caption=caption,reference_caption=reference_caption,box=box,peak=summary['peak'],low=summary['low'],high=summary['high'],grid=selected)
