# Natural5243 local direct-condition necessity — selection prepared only

No model imports/initialization,checkpoint loads,frontend/backend/recognizer calls,training,downloads or audio variant outputs. Execution is not authorized. metrics.json and experiment_report.md remain reserved for a future run. Current artifacts freeze selection and CPU in-memory eligibility only; they are not an inference result or a ready execution driver.

The scientific rule was fixed before inspecting intervention outputs:200ms target19.35–19.55s,internal20ms raised-cosine fades,known unit-gain clean PCM16 removed only from direct noisy condition,original full-float frontend anchor retained. Candidate starts run every320samples(20ms) from18.75s,with200ms windows fully inside18.75–20.75s. LEFT must end≤19.15s;RIGHT must start≥19.75s. Eligibility requires squared L2 clean-removal energy Econtrol≥Etarget so alpha≤1. Choose the nearest eligible interval on each side,then earliest start on ties. All91candidate windows including positional exclusions are frozen;4LEFT and10RIGHT candidates pass both rules.

|Arm|Seconds|Sample interval|Unscaled weighted clean energy|Alpha|Ideal removed energy|
|---|---|---|---:|---:|---:|
|target|[19.35, 19.55]|[309600,312800)|0.144544300035225|1|0.144544300035225|
|left|[18.81, 19.01]|[300960,304160)|0.191479793383499|0.86883840586509|0.144544300035225|
|right|[19.83, 20.03]|[317280,320480)|0.154687342296019|0.966658549260326|0.144544300035225|

Energy denotes sum((w×clean)²),the squared sample-space L2 norm,not time-normalized physical energy. Both controls were selected solely by clean energy and time,not by model outputs/CTC preferences. Alpha is an analytic energy-matching attenuation,not a fitted clean/noise mixture gain. The taper is float64 ones with first320samples0.5−0.5cos(linspace(0,π,320)),final320 reversed;endpoints0,plateau1,exact symmetry. Targetalpha1;controls never amplify clean removal.

Each dry condition exists only in memory:copy original float32 noisy,assign float32(noisy_float64−alpha×w×clean_PCM16) within its selected window. Nativefloat anchor remains untouched. All three are finite,bitwise identical to original outside the window,and retain exactly the original global peak0.431304931640625. Physical noise is the same paired source;it is not resynthesized or gain-fit. Subtracting the remaining clean contribution leaves the same inferred noise residual as the original mixture,up to explicit edit-roundoff bounds:

|Arm|Actual float32 edit-energy relative error|Maximum inferred-noise residual change|Allowed float32 roundoff bound|
|---|---:|---:|---:|
|target|1.6923878e-09|1.8170137e-09|5.1415554e-08|
|left|5.4474448e-10|1.8563401e-09|5.1415554e-08|
|right|3.3441793e-09|1.8353509e-09|5.1415554e-08|

All residuals also pass the completed CPU audit's conservative3PCM16-LSB mixture bound9.1552734375e−5 plus edit roundoff. Actual residual maxima are approximately3.052e−5. Sourceclean/noisy/noise hashes,full grid,taper hash,condition hashes,energy errors and all dry guards are retained. No full conditions or audio variants were written.

Source verification covers42rehashes before/after,including clean/noisy/noise,core/sampler code,original source clips,successful native replay metadata and recovered float frontend/backend,14RNG arrays,and original fixed-time CTC config/script/metrics/probability metadata. Native baseline frontend and backend array hashes match successful attempt02. The large checkpoint lock is explicitly inherited from that successful runtime's actual pre/post rehash,not loaded or reread here. Exact seed369124093 and original reference peak are frozen. No classifier score is used for control selection.

Independent CPU validation reconstructs all91window energies and eligibility,nearest-side choices,all3in-memory condition hashes,outside-window identity and original peak. It rehashes42sources,confirms frozen selection/config bytes unchanged since first write,and verifies future-call caps plus absent reserved execution artifacts. See validate_preparation.py and preparation_validation.json.

Preparation encountered a NumPyfloat32 JSON scalar formatting defect after selection/config were written. The failure log/record is preserved. Only scalar serialization was corrected;missing preparation metadata was regenerated with exact existing selection/config byte checks. The terminal summary encountered the same scalar-formatting issue after valid metadata had been written;its print serialization was corrected without another scientific selection run. Neither event involved a model call or change to windows/alpha/config. preparation_metrics.json records this provenance;the frozen hashes are unchanged.

Planned future budget remains exactly4OFF N30 calls(native exact gate,target,left,right),0frontend,and4short calls to the existing recognizer,240seconds/30MB. This preparation does not execute or authorize them. A later reviewed driver must first exactly reproduce saved native raw/PCM baseline using frozen anchor,reference,RNG and first-state tracing;any mismatch stops before edits. No Router,extra variants,seed search or training is included.

The target200ms interval brackets an existing recognizer20ms proxy near19.45–19.47s;it is not manually isolated/d/. Energy-matched controls can alter different phonetic content and neither equate local intelligibility nor isolate a single consonant. Clean-aware local condition removal is OOD,while anchor still carries speech. This design can compare positional necessity within one selected event,not establish independent noisy recoverability,human PER,hallucination rate,Router benefit or population improvement. Root Research Lead owns final interpretation/execution decision. Existing dirty files/data/core/checkpoint/listening server remain preserved;root plan not edited.
