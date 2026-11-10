import csv
import json
import numpy as np
import pytest
from dataset import load_dataset
from metrics import classification_metrics
from reporting import write_report
from visualizations import activation_grid


def test_metrics_orientation_missing_class_and_calibration():
    y=np.array([0,0,1,1]);p=np.array([[.9,.1],[.8,.2],[.6,.4],[.3,.7]])
    m=classification_metrics(y,p,classes=2)
    assert m['confusion_matrix']==[[2,0],[1,1]]
    assert m['accuracy']==.75 and m['recall']==[1,.5]
    assert m['precision'][0]==pytest.approx(2/3)
    assert m['macro_f1']==pytest.approx((.8+2/3)/2)
    assert sum(b['count'] for b in m['calibration'])==4


def test_metrics_reject_invalid_probability_vectors():
    with pytest.raises(ValueError):classification_metrics([0],[[.1,.1]],2)
    with pytest.raises(ValueError):classification_metrics([],np.empty((0,7)))


def test_dataset_preserves_partitions_and_removes_cross_split_duplicates(tmp_path):
    path=tmp_path/'fer.csv'
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['emotion','pixels','Usage'])
        for partition,offset in [('Training',0),('PublicTest',20),('PrivateTest',40)]:
            for label in range(7):writer.writerow([label,' '.join([str(offset+label)]*2304),partition])
        writer.writerow([0,' '.join(['40']*2304),'Training'])
    data,audit=load_dataset(path)
    assert audit['counts']=={'test':7,'validation':7,'train':7}
    assert audit['duplicates_removed']['train']==1
    assert data['train'][0].shape==(7,48,48,1)
    assert data['test'][0][0,0,0,0]==pytest.approx(40/255)
    assert load_dataset(path)[1]['fingerprint']==audit['fingerprint']


def test_dataset_requires_explicit_csv_partitions(tmp_path):
    p=tmp_path/'bad.csv';p.write_text('emotion,pixels\n0,1')
    with pytest.raises(ValueError,match='Usage'):load_dataset(p)


def test_report_has_true_prediction_labels_and_escaped_metadata(tmp_path):
    m=classification_metrics(np.arange(7),np.eye(7))
    destination=tmp_path/'report.html'
    write_report(destination,dict(test=m,architecture='<script>',dataset={'source':'<evil>'}),{'loss':[2,1],'val_loss':[2,1.2]})
    text=destination.read_text(encoding='utf-8')
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert 'Rows: true class' in text and '100.0%' in text


def test_constant_feature_maps_render_without_nan():
    rendered=activation_grid(np.ones((6,6,16)))
    assert rendered.shape==(544,480,3) and rendered.dtype==np.uint8
