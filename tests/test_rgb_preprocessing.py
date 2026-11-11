import cv2
import numpy as np
from expression import preprocess_frame,preprocess,face_crop
from filter_inspector import overlay_response


def test_rgb_channels_are_not_accidentally_bgr_or_gray():
    image=np.zeros((224,224,3),np.uint8);image[:]=[25,100,200]
    actual=preprocess_frame(image,(None,224,224,3))
    assert actual.shape==(1,224,224,3)
    np.testing.assert_allclose(actual[0,0,0],np.array([200,100,25])/255.,atol=1e-7)


def test_legacy_preprocessing_stays_identical():
    frame=np.random.default_rng(7).integers(0,256,(81,63,3),dtype=np.uint8)
    np.testing.assert_array_equal(preprocess_frame(frame,(48,48,1)),preprocess(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)))


def test_crop_margin_clamps_at_frame_boundary():
    frame=np.zeros((80,100,3),np.uint8)
    assert face_crop(frame,(0,2,60,60),.1).shape==(68,66,3)
    assert face_crop(frame,(10,10,30,30),0).shape==(30,30,3)


def test_rgb_filter_overlay_preserves_source_order():
    image=np.zeros((5,5,3),np.uint8);image[:]=[10,80,200]
    before=image.copy();grid=np.array([[0,1],[0,0]],np.float32)
    result=overlay_response(image,grid,((2,0,2),(2,0,2)))
    color=cv2.applyColorMap(np.array([[255]],np.uint8),cv2.COLORMAP_INFERNO)
    expected=cv2.addWeighted(image[1:2,3:4],.55,color,.45,0)
    np.testing.assert_array_equal(result[1,3],expected[0,0]);np.testing.assert_array_equal(image,before)
