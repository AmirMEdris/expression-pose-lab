import pytest
from model_dream import preview_size


@pytest.mark.parametrize('shape,edge,expected',[
    ((1080,1920,3),384,(384,216)),
    ((1920,1080,3),384,(216,384)),
    ((30,1200,3),256,(256,6)),
    ((48,48,3),384,(48,48)),
])
def test_preview_preserves_aspect_without_upscaling(shape,edge,expected):
    assert preview_size(shape,edge)==expected


def test_empty_image_rejected():
    with pytest.raises(ValueError):preview_size((0,100,3),384)
