import numpy as np
import pytest
from expression import preprocess, describe


def test_preprocessing_preserves_training_scale():
    batch = preprocess(np.full((80,60),255,dtype=np.uint8))
    assert batch.shape == (1,48,48,1)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch, 1)
    np.testing.assert_array_equal(preprocess(np.zeros((10,10),dtype=np.uint8)), 0)


def test_invalid_crop_fails_clearly():
    with pytest.raises(ValueError):
        preprocess(np.zeros((0,0),dtype=np.uint8))


def test_uncertain_output_keeps_probabilities():
    result = describe([1/7]*7, .6)
    assert result['label'] == 'uncertain'
    assert len(result['probabilities']) == 7


def test_label_order_matches_original_training():
    assert describe([0,0,0,1,0,0,0], .6)['label'] == 'happy'


@pytest.mark.parametrize('values', [[0]*7, [float('nan')]*7, [1,0], [-1,2,0,0,0,0,0]])
def test_invalid_model_outputs_rejected(values):
    with pytest.raises(ValueError):
        describe(values, .6)
