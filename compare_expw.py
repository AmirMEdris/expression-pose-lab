"""Compare a prior grayscale checkpoint on the selected RGB run's held-out split.

The RGB probabilities are reused from its one final evaluation. Prior FER2013
exposure may overlap ExpW images, so this is a practical comparison, not a clean
architecture ablation or proof of generalization beyond these datasets.
"""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
from expression import ExpressionEstimator,preprocess
from backend import predict_numpy
from metrics import classification_metrics


def run(rgb_run,gray_checkpoint,dataset):
    root=Path(rgb_run);dataset=Path(dataset);gray_checkpoint=Path(gray_checkpoint)
    audit=json.loads((dataset/'prepared.json').read_text(encoding='utf-8'))
    with np.load(root/'predictions.npz') as data:ids=data['indices'];labels=data['labels'];rgb_probabilities=data['probabilities']
    bank=np.memmap(dataset/'faces-224.rgb',dtype=np.uint8,mode='r',shape=(audit['cached_count'],224,224,3))
    estimator=ExpressionEstimator(gray_checkpoint)
    if estimator.model.input_shape[-1]!=1:raise ValueError('Comparison checkpoint must be grayscale')
    probabilities=[]
    for start in range(0,len(ids),128):
        batch=np.concatenate([preprocess(cv2.cvtColor(bank[i],cv2.COLOR_RGB2GRAY)) for i in ids[start:start+128]])
        probabilities.append(predict_numpy(estimator.model,batch))
    probabilities=np.concatenate(probabilities);gray=classification_metrics(labels,probabilities);rgb=classification_metrics(labels,rgb_probabilities)
    rng=np.random.default_rng(2026);correct_rgb=rgb_probabilities.argmax(1)==labels;correct_gray=probabilities.argmax(1)==labels
    # Paired image bootstrap describes this sample only, not population shift.
    deltas=[]
    for _ in range(2000):
        sample=rng.integers(0,len(labels),len(labels));deltas.append(float((correct_rgb[sample].astype(float)-correct_gray[sample]).mean()))
    result=dict(dataset=audit['name'],test_count=len(labels),rgb=rgb,grayscale=gray,
        grayscale_checkpoint=str(gray_checkpoint),grayscale_checkpoint_sha256=hashlib.sha256(gray_checkpoint.read_bytes()).hexdigest(),
        accuracy_difference=rgb['accuracy']-gray['accuracy'],paired_image_bootstrap_95_percent_interval=np.percentile(deltas,[2.5,97.5]).tolist(),
        caveats=['Different training datasets, architectures, and input resolutions: not an architecture ablation.',
                 'FER2013/ExpW cross-dataset image or identity overlap was not audited; prior grayscale exposure may include similar examples.',
                 'Bootstrap treats images as independent; related identities can make its interval too narrow.',
                 'Neither dataset score measures live-camera accuracy.'])
    np.savez_compressed(root/'grayscale-comparison-predictions.npz',indices=ids,labels=labels,probabilities=probabilities)
    (root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['test_count','accuracy_difference','paired_image_bootstrap_95_percent_interval']}),flush=True)
    print('RGB accuracy / macro F1',rgb['accuracy'],rgb['macro_f1'],'Gray',gray['accuracy'],gray['macro_f1'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rgb-run',default='experiments/canvas-rgb-v1');p.add_argument('--gray-checkpoint',default='experiments/residual-gpu-v1/best.keras');p.add_argument('--dataset',default='data/expw');a=p.parse_args();run(a.rgb_run,a.gray_checkpoint,a.dataset)
