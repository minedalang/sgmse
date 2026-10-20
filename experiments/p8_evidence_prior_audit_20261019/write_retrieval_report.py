from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent
D=json.loads((H/'retrieval_metrics.json').read_text());C=D['galdse_commit'];BASE=f'https://github.com/wangchengzhong/GALDSE/blob/{C}/'
for r in D['records']:
 r.setdefault('bytes',0);r.setdefault('sha256',None);r.setdefault('final_url',None);r.setdefault('http_status',None)
# Preserve the interrupted fallback which did not return to request logger.
D['records'].append({'name':'galdse/code/galdse/backbones/ncsnpp_utils/layers.py','url':f'https://raw.githubusercontent.com/wangchengzhong/GALDSE/{C}/galdse/backbones/ncsnpp_utils/layers.py','retrieved_utc':None,'proxy':True,'status':'interrupted_not_saved','bytes':0,'sha256':None,'final_url':None,'http_status':None,'error':'Intentional KeyboardInterrupt stopped broad utilities; no utility payload saved; exact request start unavailable, after direct timeout at 2026-10-19T00:59:48.090533+00:00 and before targeted retrieval at 2026-10-19T01:01:14.860900+00:00.'})
tree=json.loads((H/'galdse/tree.json').read_text());actual={str(p.relative_to(H/'galdse/code')) for p in (H/'galdse/code').rglob('*.py')};(H/'galdse/selected_code_paths.json').write_text(json.dumps([r for r in tree['tree'] if r['path'] in actual],indent=2)+'\n')
assert all(hashlib.sha256((H/r['name']).read_bytes()).hexdigest()==r['sha256'] for r in D['records'] if r['status']=='saved')
assert sum(r['bytes'] for r in D['records'] if r['status']=='saved')==D['saved_payload_bytes']
D['all_saved_payload_hashes_reverified']=True
(H/'retrieval_metrics.json').write_text(json.dumps(D,indent=2)+'\n')
text=f'''# Primary-source retrieval and code audit

Retrieved the four specified papers and official GALDSE sources. Network retrieval took {D['wall_seconds']:.2f} seconds, saving {D['saved_payload_bytes']:,} payload bytes; essential code totals {D['essential_code_bytes']:,} bytes. No training, inference, models, datasets, clones, installs, cleanup, commits or pushes. The root Research Lead owns scientific comparison and next decisions; this report records source boundaries only.

GALDSE master resolved through [GitHub commit API](https://api.github.com/repos/wangchengzhong/GALDSE/commits/master) to `{C}`, committer date 2025-03-11T09:09:09Z. README, tree and code use this immutable commit. GALD-SE abs identifies latest v5, retrieved as exact [2409.15101v5 HTML](https://arxiv.org/html/2409.15101v5). Other papers are exact v1. All request UTC timestamps, HTTP status, bytes, SHA256, final URL, transport failures and proxy use are in retrieval_metrics.json; saved HTML also has plain text extracts.

|Primary source|Frozen page|Retrieval scope/boundary|
|---|---|---|
|ArtiFree|[2509.19495v1](https://arxiv.org/html/2509.19495v1)|Embedding variability and semantic consistency across multiple diffusion samples; multi-run ensemble selection is material to its mechanism. Do not present it as a single-sample controller.|
|PASE|[2511.13300v1](https://arxiv.org/html/2511.13300v1)|Section on DRD specifies frozen clean-input teacher and trainable noisy-input WavLM student, followed by dual-stream vocoder. This is representation denoising/distillation and synthesis, not merely a frozen SSL loss applied to an existing OUVE sampler.|
|Grounded Decoding SE|[2609.04245v1](https://arxiv.org/html/2609.04245v1)|Evidence-conditioned autoregressive discrete speech-token generation; adaptive grounding penalizes Hamming distance in factorized FSQ space, with local LLM refinement. It is not continuous OUVE diffusion.|
|GALD-SE paper|[2409.15101v5](https://arxiv.org/html/2409.15101v5)|Section II-B mask-derived anisotropic noise and II-C conditioning differ from how the current repository routes guidance. Paper and current implementation must be separately identified.|
|GALDSE current official implementation|[pinned README]({BASE}README.md#L15)|README Update describes changed channels, attention-to-conformer bottleneck, and removed explicit guidance inputs. This rationale and claimed limited degradation are author statements, not independently reproduced here.|

Pinned GALDSE code provides a more precise interpretation of “removed explicit guidance”:

|Evidence|Pinned source lines|Observed code behavior|
|---|---|---|
|Frontend and anisotropic training noise|[model.py 111–124]({BASE}galdse/model.py#L111)|Frontend obtains psm; its detached square determines mask; random training noise is multiplied by that mask. A third psm channel is concatenated with the two-channel current state before the backbone call.|
|Forward wrapper|[model.py 165–167]({BASE}galdse/model.py#L165)|The wrapper forwards input, timestep and eta schedule to the backbone; it does not independently strip guidance.|
|Guidance-dependent deterministic state transformation|[TinyNCSNpp.py 267–288]({BASE}galdse/backbones/TinyNCSNpp.py#L267)|Backbone computes low-energy state magnitude, uses the final input channel and eta in an enhancement ratio, multiplies the first two state channels, then slices to those two channels. Thus a separate guidance channel is removed before the learned trunk, while guidance still changes the trunk's effective input.|
|Sampling-time guidance path|[stochastic_process.py 192]({BASE}galdse/stochastic_process.py#L192)|Sampler concatenates current state and the complement recovered from mask before calling the model, feeding the same guidance-dependent pre-warp.|
|Anisotropic reverse noise|[stochastic_process.py 246–252]({BASE}galdse/stochastic_process.py#L246)|Noise is multiplied by mask before the posterior standard deviation is applied.|
|Anisotropic initialization and drift|[stochastic_process.py 278–283]({BASE}galdse/stochastic_process.py#L278), [115–131]({BASE}galdse/stochastic_process.py#L115), [311–320]({BASE}galdse/stochastic_process.py#L311)|Initialization uses masked noise around noisy y; forward diffusion interpolates toward noisy y and adds the supplied masked noise. Mask guidance is retained in stochastic geometry.|
|Changed bottleneck|[TinyNCSNpp.py 42–53]({BASE}galdse/backbones/TinyNCSNpp.py#L42), [361–375]({BASE}galdse/backbones/TinyNCSNpp.py#L361)|Default two input channels and enabled time/frequency conformer; six conformer blocks replace the alternate attention bottleneck path.|

Consequently the pinned implementation supports a narrow statement: the learned backbone trunk consumes transformed two-channel state without a separate appended guidance feature channel. It does not support saying guidance is absent, because a mask-derived signal still controls the input pre-warp and anisotropic noise. This is static source inspection only; no checkpoint configuration, runtime execution or claimed quality comparison was verified.

The first targeted selector accidentally included backbone utility dependencies. Its iteration was interrupted after conformer.py and ncsnpp.py; no utility payload was saved. Only the essential model.py, stochastic_process.py and TinyNCSNpp.py were then added. selected_code_paths.json now lists the five actually saved code files, with Git blob IDs and tree sizes. The original candidates remain documented in tree_selection_candidates.json and retrieval script history is explained in metrics. Direct requests for conformer/model timed out once each, then succeeded through the authorized localhost7890 proxy. A utility direct request timed out and its proxy fallback was intentionally interrupted; it was not retried. No successful payload was fetched twice. These bounded transport failures are preserved, not hidden as research failures.

Read-only local evidence hashes cover the completed five-case ablation report/metrics, fixed-time CTC v2 report, and original user listening observation; see local_source_locks.json. All four remain byte-identical. Git status outside this new audit directory is unchanged. Existing dirty files were not edited; all writes are confined to this distinct experiment directory. The root RESEARCH_PLAN was read, never edited. Source-side claims, controlled VCV proxies, and the existing natural 5243 human observation remain distinct evidence levels. No superiority, novelty or research-direction decision is made here.
'''
(H/'retrieval_report.md').write_text(text)
size=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
D['final_artifact_file_bytes']=size;D['saved_payload_cap_passed']=D['saved_payload_bytes']<=5000000;D['essential_code_cap_passed']=D['essential_code_bytes']<=500000;D['wall_cap_passed']=D['wall_seconds']<=300
(H/'retrieval_metrics.json').write_text(json.dumps(D,indent=2)+'\n')
print(json.dumps({'saved_payload':D['saved_payload_bytes'],'artifact_bytes':size,'all_payload_hashes_valid':True,'local_sources_unchanged':D['local_sources_unchanged'],'outside_git_status_unchanged':D['git_status_outside_audit_unchanged']},indent=2))
