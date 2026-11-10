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
