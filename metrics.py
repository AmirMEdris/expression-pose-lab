"""Dependency-light classification analytics with rows=true, columns=predicted."""
import numpy as np


def classification_metrics(labels, probabilities, classes=7):
    labels=np.asarray(labels,dtype=int);probabilities=np.asarray(probabilities,dtype=float)
    if labels.ndim!=1 or len(labels)==0 or probabilities.shape!=(len(labels),classes):
        raise ValueError('Expected nonempty labels and N x classes probabilities')
    if (labels<0).any() or (labels>=classes).any() or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1,atol=1e-4):
        raise ValueError('Invalid labels or probabilities')
    predicted=probabilities.argmax(1)
    cm=np.zeros((classes,classes),dtype=int)
    np.add.at(cm,(labels,predicted),1)
    tp=np.diag(cm).astype(float)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(classes),where=cm.sum(1)>0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(classes),where=cm.sum(0)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(classes),where=precision+recall>0)
    confidence=probabilities.max(1);correct=predicted==labels
    bins=[];ece=0.
    for i in range(10):
        selected=(confidence>=i/10)&((confidence<(i+1)/10) if i<9 else (confidence<=1))
        if selected.any():
            accuracy=float(correct[selected].mean());conf=float(confidence[selected].mean())
            ece+=float(selected.mean())*abs(accuracy-conf)
            bins.append(dict(lower=i/10,count=int(selected.sum()),accuracy=accuracy,confidence=conf))
    return dict(accuracy=float(correct.mean()),macro_f1=float(f1.mean()),confusion_matrix=cm.tolist(),
                precision=precision.tolist(),recall=recall.tolist(),f1=f1.tolist(),support=cm.sum(1).tolist(),
                nll=float(-np.log(np.clip(probabilities[np.arange(len(labels)),labels],1e-8,1)).mean()),ece=ece,calibration=bins)
