import numpy as np
import pytest
import cv2
from filter_inspector import response_summary,overlay_response


@pytest.mark.parametrize('value',[0.,2.,-2.])
def test_uniform_response_does_not_invent_a_highlight(value):
    result=response_summary(np.full((12,12),value,np.float32))
    assert result['peak'] is None
    assert not result['normalized'].any()
    assert result['low']==result['high']==value


def test_peak_and_relative_map_preserve_signed_response_order():
    result=response_summary(np.array([[-8,-4],[-6,-10]],np.float32))
    assert result['peak']==(0,1)
    assert result['normalized'][0,1]==1 and result['normalized'][1,1]==0
    assert result['low']==-10 and result['high']==-4


@pytest.mark.parametrize('values',[np.zeros((0,2)),np.array([[np.nan]]),np.zeros((2,2,2))])
def test_invalid_map_rejected(values):
    with pytest.raises(ValueError):response_summary(values)


def test_response_overlay_uses_grid_centers_and_keeps_original_unchanged():
    gray=np.full((5,5),128,np.uint8);original=gray.copy()
    grid=np.array([[0,1],[0,0]],np.float32)
    # Grid centers at y=1,3 and x=1,3: peak must project to input (1,3).
    rendered=overlay_response(gray,grid,((2,0,2),(2,0,2)))
    hot=cv2.applyColorMap(np.array([[255]],np.uint8),cv2.COLORMAP_INFERNO)
    expected=cv2.addWeighted(np.full((1,1,3),128,np.uint8),.55,hot,.45,0)
    np.testing.assert_array_equal(rendered[1,3],expected[0,0])
    np.testing.assert_array_equal(gray,original)
