import cv2
import numpy as np
from prepare_expw import read_crop


def test_expw_crop_uses_documented_bbox_order_and_rgb(tmp_path):
    folder=tmp_path/'origin';folder.mkdir()
    image=np.zeros((120,160,3),np.uint8);image[:60,:,2]=220;image[60:,:,1]=160
    cv2.imwrite(str(folder/'face.png'),image)
    result,error=read_crop('face.png 0 10 20 119 109 22.9 3',tmp_path)
    assert error is None
    rgb,row=result
    assert rgb.shape==(224,224,3) and row['native_min_side']==100
    assert row['label']==3 and row['is_color']
    assert rgb[10,:,0].mean()>200 and rgb[10,:,2].max()==0


def test_expw_rejects_upscaled_tiny_and_constant_inputs(tmp_path):
    (tmp_path/'origin').mkdir();cv2.imwrite(str(tmp_path/'origin/flat.png'),np.full((120,120,3),128,np.uint8))
    assert read_crop('flat.png 0 0 0 30 30 5 6',tmp_path)[1]=='native_face_below_64px'
    assert read_crop('flat.png 0 0 0 119 119 5 6',tmp_path)[1]=='nearly_constant'


def test_expw_label_cannot_escape_image_directory(tmp_path):
    assert read_crop('../outside.jpg 0 0 0 100 100 5 0',tmp_path)[1]=='invalid_path'
