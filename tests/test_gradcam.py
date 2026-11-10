import numpy as np
import pytest
from gradcam import activation_heatmap,overlay


def test_class_gradients_select_different_regions():
    features=np.array([[[1,0],[0,1]],[[0,0],[0,0]]],dtype=np.float32)
    first=np.broadcast_to([1.,0.],features.shape);second=np.broadcast_to([0.,1.],features.shape)
    np.testing.assert_array_equal(activation_heatmap(features,first),[[1,0],[0,0]])
    np.testing.assert_array_equal(activation_heatmap(features,second),[[0,1],[0,0]])


def test_negative_or_zero_support_does_not_fabricate_heat():
    values=np.ones((2,2,3))
    heat=activation_heatmap(values,-values)
    np.testing.assert_array_equal(heat,np.zeros((2,2)))
    frame=np.full((5,8,3),130,np.uint8)
    np.testing.assert_array_equal(overlay(frame,heat),frame)


def test_invalid_gradients_rejected():
    with pytest.raises(ValueError):activation_heatmap(np.ones((2,2,3)),np.ones((2,3,3)))
    with pytest.raises(ValueError):activation_heatmap(np.ones((2,2,3)),np.full((2,2,3),np.nan))
