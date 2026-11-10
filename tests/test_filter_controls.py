import numpy as np
from filter_explorer import matched_controls


def test_controls_exclude_top_images_and_extremely_different_brightness():
    descriptors=np.array([[.8,.2]]*8+[[.01,.01]]*24)
    scores=np.arange(32,dtype=float);scores[8:]=-100
    selected=matched_controls(descriptors,scores,0,[0,1],count=3)
    assert selected.tolist()==[2,3,4]
