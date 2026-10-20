# Official resource metadata audit

Completed 2026-10-18. Research Lead review and the bounded next step are recorded below.

## Research Lead decision after review

Accept this resource as a candidate for an independent-label diagnostic
positive control. The nominal consonant labels originate from the corpus
recording/annotation protocol, independently of our wav2vec2 predictions;
they do not recreate DNS transcripts. They are sufficient to plan a clean
recognizer calibration pilot, with pronunciation ambiguity explicitly
flagged. They are not a guarantee of perfect human-corrected phonetic truth.

No complete same-speaker k/g pair has yet been verified from the 91 manual
entries alone. The next step is to inspect archive membership and freeze a
small same-speaker, same-vowel, same-stress subset before enhancement. First
check the clean recognition/confusion pattern using corpus labels. If the
recognizer cannot distinguish the clean targets, stop using that recognizer
as a recovery judge for this material. A model failure on noisy alone must
not be interpreted as human or information-theoretic non-recoverability.

A later minimal-pair pilot must use identical noise waveform/offset across
the two recorded identities and separate speech-changed/noise-fixed from
speech-fixed/noise-changed contrasts. Coarticulation, duration and lexical
context differ from DNS sentences and remain competing explanations. A
positive VCV result validates the controlled diagnostic, not recoverability
of the previously inspected extreme-SNR DNS examples. No new enhancement
method or training is authorized by the metadata result alone.

The audited resource is [The Consonant Challenge Corpus](https://zenodo.org/records/3538223), DOI 10.5281/zenodo.3538223. Only the three approved official endpoints were fetched: record JSON, readme.pdf, and handsegm.91.mlf.txt. All returned HTTP 200 at their requested URLs. Their bodies total 103,288 bytes (8,852 + 86,400 + 8,036), below the shared 2,000,000 byte cap. No request failed, no proxy fallback was used, and no audio archive was fetched. Final URLs, timings, status codes, and SHA-256 hashes are in fetch_metrics.json and metrics.json. There was no training, inference, registration, email, or modification to existing data.

## License and rights evidence

The official metadata explicitly gives access_right=open and license.id=cc-by-4.0. Thus a specific license designation is present. Full legal license terms are absent from the fetched JSON and README; the README contains no license/copyright statement. The license legal code was outside the three authorized URLs and was not retrieved. Detailed legal obligations were not assessed. This distinguishes an explicit license identifier from having inspected its full legal text.

## Documented corpus structure

The README specifies 12 female and 12 male native English talkers aged 18–49, producing 24 consonants including /k g d/. Three vowels /iː uː æ/ form nine ordered vowel contexts; two stress patterns yield a nominal 10,368 VCVs. Pilot listening tests screened unusable pronunciations or other problems. After screening, 6,664 clean training tokens come from eight female and eight male talkers; the remaining eight talkers supply independent tests. Seven test sets each have 384 tokens, 16 per consonant, across clean and six masker conditions. The README does not give a specific development speaker allocation/count. Postprocessed audio is documented as 25 kHz.

Filename fields indicate speaker gender, speaker ID, VCV, and final stress digit (1 front, 2 end). The README example is internally inconsistent: it specifies IDs 1–24 but uses speaker 25; m25adu2.wav is described with final /iː/ despite its filename u. These inconsistencies were recorded without silently resolving them.

The manual MLF parses into exactly 91 unique files with one consonant segment each, spanning all 24 consonant labels. All 91 filenames parse and their consonant field agrees with the segment label. Complete filename/segment lists and per-consonant counts are persisted in metrics.json.

| Label | Manual MLF count | Filenames |
|---|---:|---|
| k | 3 | f23iki1.lab; m9aka1.lab; m10uku2.lab |
| g | 3 | m8igi1.lab; m2ugu1.lab; f11aga1.lab |
| d | 4 | m4udu1.lab; m4udu2.lab; f22idi1.lab; m8ada1.lab |

Counts for all consonant labels: b=8, ch=6, d=4, dh=3, dj=3, f=5, g=3, h=3, k=3, l=3, m=3, n=5, ng=6, p=3, r=3, s=3, sh=6, t=3, th=3, v=3, w=3, y=3, z=3, zh=3. The metadata describes at least three identical-vowel-context items per consonant plus 19 randomly selected VCVs. Hand segmentation supplies boundary annotations; matching filename and MLF labels do not establish independently human-corrected phoneme identity.

## Downloadable file sizes from official metadata

| File | Bytes |
|---|---:|
| train.zip | 169,755,879 |
| dev_binaural.zip | 104,063,498 |
| segmentation_testsets.zip | 38,727 |
| segmentation_training.mlf.txt | 500,224 |
| readme.pdf | 86,400 |
| handsegm.91.mlf.txt | 8,036 |
| asr.zip | 1,080,519 |
| dev.zip | 69,148,137 |
| test_binaural.zip | 201,826,628 |
| test.zip | 142,547,415 |

train.zip was not fetched. Metadata MD5 values are preserved in metrics.json. PDF text was extracted using the already available pypdf at /tmp/codex_pypdf; no software was installed. Raw official documents and extracted text remain in this experiment directory. fetch.py records the bounded retrieval procedure; audit.py reproduces the metadata/MLF analysis offline and refuses to overwrite metrics.json.
