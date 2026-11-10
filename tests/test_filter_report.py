import hashlib
import json
from model_results import find_filter_report


def test_filter_report_requires_matching_weights_and_html(tmp_path):
    model=tmp_path/'test.keras';model.write_bytes(b'weights')
    report=model.with_suffix('.filters.html');report.write_text('local report')
    manifest=model.with_suffix('.filters.json')
    metadata=dict(checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(metadata))
    assert find_filter_report(model)[0]==report
    report.write_text('different report')
    assert find_filter_report(model)==(None,None)
    report.write_text('local report');model.write_bytes(b'other weights')
    assert find_filter_report(model)==(None,None)
