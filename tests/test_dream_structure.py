import numpy as np
from dream_structure import guided_step


def test_large_repeated_updates_cannot_destroy_source_structure():
    reference=np.linspace(0,1,48*48,dtype=np.float32).reshape(48,48)
    x=reference.copy();gradient=np.ones_like(x)
    for _ in range(400):x=guided_step(x,gradient,reference,.06)
    assert x.min()>=0 and x.max()<=1
    assert np.max(np.abs(x-reference))<=.140001
    assert (x-reference).mean()>0


def test_zero_response_does_not_invent_patterns():
    reference=np.full((48,48),.5,np.float32)
    np.testing.assert_array_equal(guided_step(reference,np.zeros_like(reference),reference,.025),reference)


def test_fine_alternating_noise_is_reduced_before_ascent():
    y,x=np.mgrid[:48,:48]
    gradient=(np.where((x+y)%2,1.,-1.)+.5).astype(np.float32)
    image=guided_step(np.full((48,48),.5,np.float32),gradient,np.full((48,48),.5,np.float32),.002)
    assert np.max(np.abs(np.diff(image[5:-5,5:-5],axis=1)))<.01
