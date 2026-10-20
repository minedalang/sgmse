"""Summarize downloaded official metadata and MLF without model or network use."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT / 'record.json').read_text())
fetch = json.loads((ROOT / 'fetch_metrics.json').read_text())
metadata = record['metadata']
entries = []
current = None
for line in (ROOT / 'handsegm.91.mlf.txt').read_text().splitlines():
    line = line.strip()
    if line.startswith('"'):
        current = {'filename': line.strip('"'), 'segments': []}
        entries.append(current)
    elif current is not None and re.fullmatch(r'\d+\s+\d+\s+\S+', line):
        start, end, label = line.split()
        current['segments'].append({'start': int(start), 'end': int(end), 'label': label})
vowels = {'aa1', 'aa2', 'iy1', 'iy2', 'uw1', 'uw2'}
for entry in entries:
    consonants = [s['label'] for s in entry['segments'] if s['label'] not in vowels | {'sil'}]
    if len(consonants) != 1:
        raise ValueError(entry)
    entry['consonant_label'] = consonants[0]
    stem = Path(entry['filename']).stem
    match = re.fullmatch(r'([fm])(\d+)([aiu])(.+)([aiu])([12])', stem)
    if match:
        entry['filename_fields'] = dict(zip(('gender', 'speaker_id', 'first_vowel', 'consonant', 'second_vowel', 'stress'), match.groups()))
counts = dict(sorted(Counter(e['consonant_label'] for e in entries).items()))
target = {c: [e['filename'] for e in entries if e['consonant_label'] == c] for c in ('k', 'g', 'd')}
readme = (ROOT / 'readme_extracted.txt').read_text()
result = {
    'experiment_id': ROOT.name, 'status': 'completed_metadata_only',
    'official_record': 'https://zenodo.org/records/3538223',
    'title': metadata['title'], 'doi': metadata['doi'],
    'fetch': fetch,
    'license': {'metadata_access_right': metadata.get('access_right'),
                'metadata_license': metadata.get('license'),
                'explicit_license_identifier': True,
                'full_legal_license_text_in_record_or_readme': False,
                'readme_license_mentions': [line for line in readme.splitlines() if re.search(r'licen[cs]|copyright|rights reserved|creative commons', line, re.I)],
                'assessment': 'CC BY 4.0 is explicitly designated by metadata. Full legal terms are not reproduced in fetched record JSON or README. License legal code was outside the three authorized fetch URLs and was not fetched; detailed reuse obligations are not audited here.'},
    'downloadable_files': [{'name': f['key'], 'size_bytes': f['size'], 'checksum': f['checksum']} for f in record['files']],
    'train_zip_size_bytes': next(f['size'] for f in record['files'] if f['key'] == 'train.zip'),
    'corpus_documentation': {
        'talkers': {'female': 12, 'male': 12, 'native_language': 'English', 'age_range': [18,49]},
        'consonants': 24, 'target_consonants_listed': ['k','g','d'],
        'vowels_ipa': ['iː','uː','æ'], 'vowel_contexts': 9,
        'stress_patterns': {'1': 'front', '2': 'end'},
        'nominal_full_factorial_tokens': 10368,
        'train': {'female_talkers': 8, 'male_talkers': 8, 'usable_clean_tokens': 6664},
        'test': {'held_out_talkers': 8, 'sets': 7, 'tokens_per_set': 384, 'instances_per_consonant_per_set':16},
        'development': 'dev.zip and dev_binaural.zip are listed; specific development speaker allocation/count is not specified in the fetched README.',
        'sample_rate_after_postprocessing_hz': 25000,
        'filename_pattern': '(f|m)<speaker_id><vowel><consonant><vowel><stress>.wav',
        'filename_example_documentation_inconsistencies': ['README states speaker IDs 1–24 but uses m25adu2.wav as example.', 'README describes the final vowel of m25adu2.wav as /iː/ despite u in filename; no silent correction assumed.'],
        'pronunciation_screening': 'README reports pilot listening tests to identify unusable tokens due to poor pronunciations or other problems; no per-file independently corrected identity table inspected.'},
    'manual_segmentation': {
        'metadata_description': '91 hand-segmented VCVs in HTK format; at least three items per consonant with identical first/second vowels, plus 19 randomly selected VCVs.',
        'parsed_entry_count': len(entries), 'unique_filenames': len({e['filename'] for e in entries}),
        'consonant_counts': counts, 'target_consonant_filenames': target,
        'filename_parse_count': sum('filename_fields' in e for e in entries),
        'filename_consonant_vs_segment_disagreements': [e['filename'] for e in entries if e.get('filename_fields',{}).get('consonant') != e['consonant_label']],
        'same_vowel_context_count': sum(e['filename_fields']['first_vowel']==e['filename_fields']['second_vowel'] for e in entries),
        'entries': entries,
        'interpretation': 'Hand segmentation describes boundaries. Neither filename consonant labels nor matching MLF segment labels establish independently human-corrected phoneme identity.'},
    'pdf_extraction': {'library': 'existing pypdf from /tmp/codex_pypdf; no installation', 'output': 'readme_extracted.txt'},
    'audio_archives_downloaded': False, 'training_or_inference': False, 'source_data_modified': False,
    'raw_artifact_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('record.json','readme.pdf','handsegm.91.mlf.txt','readme_extracted.txt')},
}
if (ROOT/'metrics.json').exists():
    raise FileExistsError('metrics.json')
(ROOT/'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:result[k] for k in ('train_zip_size_bytes','license')},indent=2))
print('manual entries:',len(entries),'counts:',counts,'targets:',target)
