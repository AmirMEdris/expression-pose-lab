"""Standalone offline HTML experiment reports; no external scripts or telemetry."""
import html
import json
from pathlib import Path
from expression import LABELS


def write_report(destination, result, history):
    escape=html.escape
    metrics=result['test'];cm=metrics['confusion_matrix']
    rows=''
    for i,label in enumerate(LABELS):
        total=max(1,sum(cm[i]));cells=''
        for value in cm[i]:
            alpha=.08+.8*value/total
            cells+=f'<td style="background:rgba(94,234,212,{alpha:.3f})">{value}<small>{value/total:.0%}</small></td>'
        rows+=f'<tr><th>{label}</th>{cells}</tr>'
    perclass=''.join(f'<tr><th>{label}</th><td>{metrics["precision"][i]:.1%}</td><td>{metrics["recall"][i]:.1%}</td><td>{metrics["f1"][i]:.1%}</td><td>{metrics["support"][i]}</td></tr>' for i,label in enumerate(LABELS))
    def chart(keys,title):
        series=[history.get(k,[]) for k in keys];count=max((len(s) for s in series),default=0)
        if not count:return ''
        maximum=max(1.,max((max(s) for s in series if s),default=1))
        paths=''
        for key,values,color in zip(keys,series,('#5eead4','#a78bfa')):
            points=' '.join(f'{45+i/max(1,count-1)*680:.1f},{230-v/maximum*190:.1f}' for i,v in enumerate(values))
            paths+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        return f'<h3>{title}</h3><p class="muted">Mint: training · Violet: validation · Epoch 1–{count}</p><svg role="img" aria-label="{title}" viewBox="0 0 760 260"><path d="M45 25V230H735" stroke="#64748b" fill="none"/><text x="5" y="40">{maximum:.2f}</text><text x="20" y="230">0</text>{paths}</svg>'
    model_name=escape(result['architecture']);audit=escape(json.dumps(result['dataset'],indent=2))
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expression Lab · Model results</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1120;color:#e2e8f0;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:60px 28px}}.eyebrow{{color:#5eead4;letter-spacing:.18em;font-size:12px}}h1{{font-size:46px;line-height:1.1;margin:18px 0}}h2{{margin-top:42px}}.muted,small{{color:#94a3b8}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:30px 0}}.card{{border:1px solid #293548;background:#111c30;border-radius:16px;padding:24px}}.number{{font-size:34px;font-weight:700;color:#5eead4}}table{{width:100%;border-collapse:separate;border-spacing:4px}}td,th{{padding:12px;text-align:center;border-radius:5px}}small{{display:block}}td small{{color:#e2e8f0}}th{{font-size:12px}}.scroll{{overflow:auto}}pre{{overflow:auto;padding:22px;background:#111c30;border-radius:12px;font-size:13px}}svg{{width:100%;background:#111c30;border-radius:12px}}svg text{{fill:#94a3b8;font-size:13px}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}h1{{font-size:32px}}}}
</style><main><div class="eyebrow">EXPRESSION + POSE LAB / MODEL RESULTS</div><h1>How this model performs.</h1><p class="muted">{model_name} · FER2013 evaluation</p><p>{escape(result.get('note','Checkpoint selected using validation loss. Final test held out from training.'))}</p>
<div class="cards"><div class="card">Test accuracy<div class="number">{metrics['accuracy']:.1%}</div></div><div class="card">Macro F1<div class="number">{metrics['macro_f1']:.1%}</div></div><div class="card">Test examples<div class="number">{sum(metrics['support']):,}</div></div></div>
<p>These results describe facial-expression labels in this dataset. They do not establish a person's emotional state or guarantee performance on live camera footage.</p>
{chart(('loss','val_loss'),'Learning curves · cross-entropy')}{chart(('accuracy','val_accuracy'),'Learning curves · accuracy')}
<h2>Where predictions succeed—and fail</h2><p class="muted">Rows: true class. Columns: predicted class. Percentages normalized within each true class.</p><div class="scroll"><table><tr><th>True ↓ / Predicted →</th>{''.join('<th>'+x+'</th>' for x in LABELS)}</tr>{rows}</table></div>
<h2>Every class counts</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>{perclass}</table>
<h2>Confidence quality</h2><p>Expected calibration error: {metrics['ece']:.3f} (10 bins). Negative log-likelihood: {metrics['nll']:.3f}. Raw probabilities are not calibrated; the app's uncertainty threshold is a display rule.</p>
<h2>Reproducibility</h2><pre>{audit}</pre><p class="muted">Machine-readable metrics and per-example predictions are saved alongside the checkpoint. Dataset images are not embedded in this report.</p></main></html>'''
    Path(destination).write_text(content,encoding='utf-8')
