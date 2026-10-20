#!/usr/bin/env python3
"""Fixed16 fresh-speaker retrieval; reuse bounded HTTP range transport unchanged."""
import argparse,importlib.util,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OLD=ROOT/'experiments/p8_q_vcv_clean_calibration_20261018'
SPEAKERS=['f18','m5','f20','m15']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=json.loads((HERE/'config.yaml').read_text());assert sha(OLD/'fetch_subset.py')==cfg['old_fetch_sha256'];assert sha(OLD/'retrieval/selected_manifest_frozen.json')==cfg['old_manifest_sha256']
    previous=json.loads((OLD/'retrieval/selected_manifest_frozen.json').read_text());assert len(previous['files'])==28
    excluded=sorted({r['speaker'] for r in previous['files']});assert set(excluded)=={'f6','m2','f11','m4','m8','m9','m10','f22','f23'} and not set(SPEAKERS)&set(excluded)
    spec=importlib.util.spec_from_file_location('original_fetch',OLD/'fetch_subset.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    def select(inventory,manual):
        indexed={}
        for r in inventory:
            if r.get('nominal'):
                name=r['basename'].lower();assert name not in indexed;indexed[name]=r
        names={sp:[f'{sp}a{c}a1.wav' for c in 'kgtd'] for sp in SPEAKERS}
        assert all(n in indexed for ns in names.values() for n in ns),'Missing fixed fresh sample; no replacement'
        rows=[dict(indexed[n],role='primary',manual_boundaries=None) for sp in SPEAKERS for n in names[sp]]
        pairs=[{'pair_id':sp+'_'+contrast.replace('/',''),'speaker':sp,'contrast':contrast,'members':names[sp][offset:offset+2],'context':'aCa','stress':'1'} for sp in SPEAKERS for contrast,offset in [('k/g',0),('t/d',2)]]
        return {'speakers':SPEAKERS,'pairs':pairs,'files':rows,'manual_targets_in_archive':[],'identity_basis':'Corpus filename nominal labels independent of models','manual_boundary_caveat':'No manual boundaries used; nominal labels do not establish perfect identity','frozen_before_model_results':True,'excluded_previous28_speakers':excluded,'old_manifest_sha256':cfg['old_manifest_sha256'],'old_fetch_module_sha256':cfg['old_fetch_sha256'],'wrapper_sha256':sha(__file__)}
    mod.select=select;mod.DERIVED_CAP=16_000_000
    # Availability check is local only; actual fetch rechecks the server central directory.
    select(json.loads((OLD/'retrieval/central_directory_inventory.json').read_text()),{})
    audit=ROOT/'experiments/p8_q_external_corpus_audit_20261018';parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',choices=['https://zenodo.org/api/records/3538223/files/train.zip/content','https://zenodo.org/records/3538223/files/train.zip?download=1']);args=parser.parse_args()
    args.output=HERE/'retrieval';args.metadata=audit/'record.json';args.manual_mlf=audit/'handsegm.91.mlf.txt';mod.run(args)
if __name__=='__main__':main()
