import numpy as np
import pytest
from dream_selection import resolve_filter,filter_activation


CATALOG=[dict(name=f'conv_{i}',channels=n) for i,n in enumerate([64,64,128,128,256,256])]


def test_explorer_numbering_maps_to_exact_channel():
    assert resolve_filter(CATALOG,4,35)==('conv_3',35)
    assert resolve_filter(CATALOG,1,0)==('conv_0',0)
    assert resolve_filter(CATALOG,6,255)==('conv_5',255)


@pytest.mark.parametrize('layer,channel',[(0,0),(7,0),(True,0),(4,-1),(4,128),(4,35.0),(4,True)])
def test_invalid_selection_never_falls_back_to_mixed(layer,channel):
    with pytest.raises(ValueError):resolve_filter(CATALOG,layer,channel)


def test_selected_objective_excludes_other_channels():
    activation=np.zeros((1,3,4,128),np.float32)
    activation[...,35]=-2
    activation[...,34]=1000
    activation[...,36]=5000
    selected=filter_activation(activation,35)
    assert selected.shape==(1,3,4,1)
    assert selected.mean()==-2  # Signed mean: squaring would reward negative responses.
    activation[...,34]=1e8
    activation[...,36]=-1e8
    assert filter_activation(activation,35).mean()==-2


@pytest.mark.parametrize('channel',[-1,128,True,35.0])
def test_invalid_channel_rejected(channel):
    with pytest.raises(ValueError):filter_activation(np.zeros((1,2,2,128)),channel)
