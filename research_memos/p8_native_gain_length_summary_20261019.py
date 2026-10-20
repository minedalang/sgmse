"""Lead's descriptive arithmetic on immutable JSON; no audio/model imports or inference."""
from pathlib import Path
import hashlib
import json
import math
import statistics

SOURCE = Path('/data/coding/sgmse/experiments/p8_voicebank_native_content_pilot_20261019/attempt02/metrics.json')
EXPECTED = 'e0ed9b10c646cc19d28cc8c6c6cd0225fd87c1b19b77bac108b1bb19a17066e0'
DEST = Path(__file__).with_suffix('.json')

def main():
    payload = SOURCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED
    m = json.loads(payload)
    assert m['status'] == 'completed'
    items = {item['key']: item for item in m['items']}
    rows = []
    for case in m['summary']['per_case']:
        name = case['basename']
        front, off = (items[name + '__' + arm] for arm in ['frontend', 'off'])
        fw, ow = front['score']['word'], off['score']['word']
        assert front['input']['rms'] > 0 and off['input']['rms'] > 0
        delta = ow['N_hyp'] - fw['N_hyp']
        assert delta == fw['D'] - ow['D'] + ow['I'] - fw['I']
        rows.append(dict(
            basename=name, noise=case['noise'], snr_db=case['snr_db'],
            rms_gain_db=20 * math.log10(off['input']['rms'] / front['input']['rms']),
            peak_gain_db=20 * math.log10(off['input']['peak'] / front['input']['peak']),
            frontend_hypothesis_words=fw['N_hyp'], off_hypothesis_words=ow['N_hyp'],
            hypothesis_word_delta=delta, word_edit_error_delta=ow['edit_distance'] - fw['edit_distance']))
    assert len(rows) == 16
    strata = {}
    for snr in [-18, 0]:
        group = [r for r in rows if r['snr_db'] == snr]
        gains = [r['rms_gain_db'] for r in group]
        strata[str(snr)] = dict(cases=len(group), median_rms_gain_db=statistics.median(gains),
            min_rms_gain_db=min(gains), max_rms_gain_db=max(gains),
            longer_ASR_hypotheses=sum(r['hypothesis_word_delta'] > 0 for r in group),
            total_hypothesis_word_delta=sum(r['hypothesis_word_delta'] for r in group),
            total_word_edit_error_delta=sum(r['word_edit_error_delta'] for r in group))
    result = dict(source=str(SOURCE), source_sha256=EXPECTED,
        scope='Post-hoc descriptive Lead analysis of all 16 existing records; not a new model experiment',
        formula='20*log10(actual_ASR_PCM_RMS_OFF/actual_ASR_PCM_RMS_frontend); no flooring/window/selection',
        word_count_identity='delta N_hyp = D_frontend-D_OFF + I_OFF-I_frontend',
        rows=rows, strata=strata, model_calls=0, audio_decodes=0, network_calls=0,
        limitations=['No amplitude intervention/redecoding; association is not causation',
            'ASR hypothesis length is not a count of actual spoken words',
            'Four repeated utterances/two speakers; no independent-sample inference'])
    with DEST.open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    print(json.dumps(strata))

if __name__ == '__main__':
    main()
