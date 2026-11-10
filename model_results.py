"""Resolve results belonging to the active weights, never an unrelated experiment."""
import hashlib
import json
from pathlib import Path


def find_report(model_path):
    model_path=Path(model_path)
    candidates=[(model_path.with_suffix('.metrics.json'),model_path.with_suffix('.report.html')),
                (model_path.parent/'metrics.json',model_path.parent/'report.html')]
    digest=None
    for metrics,report in candidates:
        if not metrics.is_file() or not report.is_file():continue
        try:
            metadata=json.loads(metrics.read_text(encoding='utf-8'))
            if digest is None:digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
            if metadata.get('checkpoint_sha256')==digest:return report,metadata
        except (OSError,ValueError):continue
    return None,None


def find_filter_report(model_path):
    model_path=Path(model_path)
    report=model_path.with_suffix('.filters.html');manifest=model_path.with_suffix('.filters.json')
    if not report.is_file() or not manifest.is_file():return None,None
    try:
        metadata=json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('checkpoint_sha256')!=hashlib.sha256(model_path.read_bytes()).hexdigest():return None,None
        if metadata.get('report_sha256')!=hashlib.sha256(report.read_bytes()).hexdigest():return None,None
        return report,metadata
    except (OSError,ValueError):return None,None
