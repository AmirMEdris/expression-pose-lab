import hashlib
import json
from model_results import find_report


def test_report_must_match_active_checkpoint(tmp_path):
    weights=tmp_path/'best.keras';weights.write_bytes(b'model-one')
    report=tmp_path/'best.report.html';report.write_text('<h1>Results</h1>')
    metrics=tmp_path/'best.metrics.json';metrics.write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}))
    assert find_report(weights)[0]==report
    weights.write_bytes(b'different-model')
    assert find_report(weights)==(None,None)


def test_no_report_never_selects_unrelated_html(tmp_path):
    weights=tmp_path/'best.keras';weights.write_bytes(b'model')
    (tmp_path/'report.html').write_text('unrelated report')
    assert find_report(weights)==(None,None)


def test_training_directory_report_is_supported(tmp_path):
    weights=tmp_path/'best.keras';weights.write_bytes(b'model')
    (tmp_path/'report.html').write_text('report')
    (tmp_path/'metrics.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(b'model').hexdigest()}))
    assert find_report(weights)[0]==tmp_path/'report.html'
