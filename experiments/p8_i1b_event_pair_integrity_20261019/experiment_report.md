# I1b technical operation integrity: completed

All24eventslots/96operations retained including unavailable/unsafe. Automatic spectral candidates are not phoneme truth; constant Hann spectral leakage is not evidence of a true event. Shared X-S residual may contain speech. Positioncontrol can contain speech, twofamilies/sham do not prove naturalness or remove detectable edit artifacts. No waveform output/classifier/model/training; all scientific/noise/split gates remainfalse. Main and independent exact rules, full candidates/rankings/float&PCM checks in metrics.

## Worker execution and validation

Unique main session51030 exit0,1.063286751s, PID43026; unique completed-only validator session92912 exit0,3.474548471s, PID43036. PIDs from actual markers/runtime, not independently ps-observed. Exact commands:

`/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_i1b_event_pair_integrity_20261019/run_integrity.py --execute-after-root-review`

`/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_i1b_event_pair_integrity_20261019/validate_integrity.py --validate-after-completed-run`

No synthetic rerun/retry/rule alteration. Eachhard120s and entire8MiB cap retained. Main24originalPCM decodes, validatoradditional24,48total; each1,384,834samplevalues/2,769,668decodedpayloadB. Extra wholefileSHA accounting: preflight24hash/2,770,724B plus mainpre/post48hash/5,541,448B plus validatorpre/post48hash/5,541,448B =120hash reads/**13,853,620B**. MainoriginalfullWAVidentity read2,770,724B and validatoradditional2,770,724B =**5,541,448B separately read into RAM**, then headers/PCM decoded from memory and original/sham identities hashed/copied in RAM. Do not call this zeroaudio-byte processing. No newWAV/NPY/PCMpayload files or playback.

Validator independently rebuilds allframes, allcandidate scores/rankings/controlpositions, masks, floatarithmetic, exactpayload/mask/sourcehashes and all96operations without importing business functions. Sharedruntime.py is IO/lifecycle/guards/writes, both use sameNumPyFFT/cos backend; this is NOT independent FFTalgorithm verification. IndependentPCM decoder struct.iter_unpack versus mainnp.frombuffer; sourceoriginalint16roundtrip separately proved. Technical arithmetic checks don't validate naturalness/semantics.

## Outcomes and all12 coordinates

All24target slotsavailable; all96operationssafe, unsafe0/unavailable0. These counts are contingent observations, not a rule permitting future unsafe omission. No clipping/renormalization/replacement or threshold change. Eachtarget/control80ms=1280samples; samplecoords below startinclusive/endexclusive; time seconds=start/16000. Candidate selection uses Cleanoracle analysis only, no noisy/ASR/output selection. Names transient/sustained are automatic spectralstructure labels, **not phonemes or trueweakconsonants**.

| ID | event | target[start,end) | control[start,end) | candidate score | untapered targetenergy | controlenergy | absenergy mismatch | sharedcontrol dependence |
|---|---|---|---|---:|---:|---:|---:|---|
| p233_027 | transient | [45920,47200) | [20160,21440) | 0.876030715567 | 0.399894261733 | 0.403993552551 | 0.00409929081798 | [] |
| p233_027 | sustained | [13760,15040) | [34720,36000) | 0.852909158727 | 0.512516705319 | 0.509238471277 | 0.00327823404223 | [] |
| p233_028 | transient | [17600,18880) | [22560,23840) | 0.715453077812 | 0.824194732122 | 0.825954241678 | 0.001759509556 | [] |
| p233_028 | sustained | [20640,21920) | [43360,44640) | 0.783202566014 | 0.152911853045 | 0.158557780087 | 0.00564592704177 | [] |
| p236_036 | transient | [13920,15200) | [37600,38880) | 0.787612623244 | 0.885705045424 | 0.942976830527 | 0.0572717851028 | [] |
| p236_036 | sustained | [42240,43520) | [20800,22080) | 0.842579044893 | 0.354736139067 | 0.339504120871 | 0.0152320181951 | [] |
| p236_040 | transient | [8800,10080) | [49600,50880) | 0.643187914955 | 0.297882293351 | 0.291528921574 | 0.00635337177664 | [] |
| p236_040 | sustained | [44000,45280) | [13920,15200) | 0.559381891504 | 0.0326266949996 | 0.0326178595424 | 8.83545726538e-06 | [] |
| p239_026 | transient | [27200,28480) | [30560,31840) | 0.616693514329 | 0.445252612233 | 0.427883993834 | 0.0173686183989 | [] |
| p239_026 | sustained | [44640,45920) | [29600,30880) | 0.819303248786 | 0.175243357196 | 0.191040369682 | 0.0157970124856 | [] |
| p239_028 | transient | [24960,26240) | [42080,43360) | 0.881982364762 | 1.34005779494 | 1.31723419763 | 0.0228235973045 | [] |
| p239_028 | sustained | [47360,48640) | [39040,40320) | 0.88373675817 | 1.52133701462 | 1.51695863809 | 0.00437837652862 | [] |
| p243_026 | transient | [50400,51680) | [52480,53760) | 0.841569270781 | 0.0466756802052 | 0.0444665960968 | 0.00220908410847 | [] |
| p243_026 | sustained | [11360,12640) | [59840,61120) | 0.976693159658 | 0.639151738957 | 0.636436665431 | 0.00271507352591 | [] |
| p243_027 | transient | [46560,47840) | [53920,55200) | 0.760484931818 | 0.492312167771 | 0.510643390939 | 0.0183312231675 | ['sustained'] |
| p243_027 | sustained | [15520,16800) | [53920,55200) | 0.930351849091 | 0.492043801583 | 0.510643390939 | 0.0185995893553 | ['transient'] |
| p244_028 | transient | [16000,17280) | [32960,34240) | 0.85230981701 | 1.60406087711 | 1.62174251862 | 0.0176816415042 | [] |
| p244_028 | sustained | [18560,19840) | [33600,34880) | 0.978924952556 | 2.89883112255 | 2.89948609285 | 0.000654970295727 | [] |
| p244_034 | transient | [45280,46560) | [35040,36320) | 0.742434410123 | 0.197705131024 | 0.227185881697 | 0.0294807506725 | [] |
| p244_034 | sustained | [11840,13120) | [35360,36640) | 0.760981942344 | 1.9842056809 | 2.05810080376 | 0.0738951228559 | [] |
| p250_038 | transient | [51840,53120) | [44480,45760) | 0.665255544494 | 0.137093812227 | 0.136593126692 | 0.000500685535371 | [] |
| p250_038 | sustained | [15040,16320) | [38560,39840) | 0.892243280034 | 0.418979187496 | 0.420028199442 | 0.00104901194572 | [] |
| p250_050 | transient | [21920,23200) | [50880,52160) | 0.590532990975 | 0.447837056592 | 0.454561201856 | 0.00672414526343 | [] |
| p250_050 | sustained | [40800,42080) | [43680,44960) | 0.258781719553 | 0.491666070186 | 0.466507165693 | 0.0251589044929 | [] |

## All96 operations: actual tapered energies and numerical error

Positioncontrol only minimizes untapered fullwindowCleanenergy difference; it may contain speech and is not a negative truth label. Samepositions are used for bothfamilies. Tapereddeltaenergy mismatches below need not vanish. p243_027 transient/sustained both usecontrol53920, an explicit within-sentence dependency, not independentexamples.

| ID | event | family | site | status | deltaenergy | control−target deltaenergy | floatsharedresidual error | applieddelta error | cleanPCM qerror | noisyPCM qerror | quantizedarms+rawN reconerror | quantizedarms+quantizedN reconerror |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p233_027 | transient | cosine | target | safe | 0.202571582898 | 0 | 6.93889390391e-18 | 4.14707330976e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | transient | cosine | position_control | safe | 0.12157471511 | -0.0809968677878 | 1.38777878078e-17 | 6.50521303491e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | transient | linear | target | safe | 0.201530051202 | 0 | 1.38777878078e-17 | 5.20417042793e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | transient | linear | position_control | safe | 0.114646641855 | -0.0868834093466 | 1.38777878078e-17 | 4.33680868994e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | sustained | cosine | target | safe | 0.274223341064 | 0 | 6.93889390391e-18 | 6.23416249179e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | sustained | cosine | position_control | safe | 0.275449492993 | 0.00122615192923 | 3.46944695195e-18 | 5.70053173502e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | sustained | linear | target | safe | 0.27289571034 | 0 | 6.93889390391e-18 | 6.31547765473e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_027 | sustained | linear | position_control | safe | 0.274590454742 | 0.00169474440254 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | transient | cosine | target | safe | 0.305220501938 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | transient | cosine | position_control | safe | 0.392423818533 | 0.0872033165944 | 1.38777878078e-17 | 6.07153216592e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | transient | linear | target | safe | 0.288543778488 | 0 | 1.38777878078e-17 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | transient | linear | position_control | safe | 0.387141562907 | 0.0985977844189 | 3.46944695195e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | sustained | cosine | target | safe | 0.0635513232501 | 0 | 1.38777878078e-17 | 6.50521303491e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | sustained | cosine | position_control | safe | 0.0689937961476 | 0.00544247289742 | 2.77555756156e-17 | 1.21430643318e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | sustained | linear | target | safe | 0.063072596419 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p233_028 | sustained | linear | position_control | safe | 0.0686252254664 | 0.00555262904737 | 1.38777878078e-17 | 1.12757025938e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | transient | cosine | target | safe | 0.475188700018 | 0 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | transient | cosine | position_control | safe | 0.346931251595 | -0.128257448422 | 2.77555756156e-17 | 1.25767452008e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | transient | linear | target | safe | 0.47320443621 | 0 | 1.38777878078e-17 | 1.35525271561e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | transient | linear | position_control | safe | 0.341168670578 | -0.132035765633 | 1.38777878078e-17 | 1.21430643318e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | sustained | cosine | target | safe | 0.18455278855 | 0 | 1.38777878078e-17 | 1.31188462871e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | sustained | cosine | position_control | safe | 0.108454570017 | -0.0760982185331 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | sustained | linear | target | safe | 0.182498277298 | 0 | 1.38777878078e-17 | 1.34441069388e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_036 | sustained | linear | position_control | safe | 0.0982959052386 | -0.0842023720591 | 2.77555756156e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | transient | cosine | target | safe | 0.0909459291137 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | transient | cosine | position_control | safe | 0.125838033664 | 0.0348921045501 | 1.38777878078e-17 | 1.12757025938e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | transient | linear | target | safe | 0.0907603621182 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | transient | linear | position_control | safe | 0.125419710731 | 0.0346593486133 | 1.38777878078e-17 | 6.83047368666e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | sustained | cosine | target | safe | 0.0150321137185 | 0 | 1.38777878078e-17 | 6.83047368666e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | sustained | cosine | position_control | safe | 0.0158277378256 | 0.000795624107124 | 1.38777878078e-17 | 6.50521303491e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | sustained | linear | target | safe | 0.0147593169138 | 0 | 1.38777878078e-17 | 6.80336863235e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p236_040 | sustained | linear | position_control | safe | 0.0155420890086 | 0.000782772094791 | 6.93889390391e-18 | 6.80760379708e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | transient | cosine | target | safe | 0.0926414468928 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | transient | cosine | position_control | safe | 0.107659597553 | 0.0150181506604 | 3.46944695195e-18 | 1.04083408559e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | transient | linear | target | safe | 0.0971533490502 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | transient | linear | position_control | safe | 0.101788055157 | 0.00463470610647 | 2.77555756156e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | sustained | cosine | target | safe | 0.0829871525653 | 0 | 1.38777878078e-17 | 6.42050974019e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | sustained | cosine | position_control | safe | 0.0907906279445 | 0.00780347537924 | 6.93889390391e-18 | 3.45081222711e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | sustained | linear | target | safe | 0.0806750828702 | 0 | 1.38777878078e-17 | 5.69206140555e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_026 | sustained | linear | position_control | safe | 0.0890494780158 | 0.00837439514558 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | transient | cosine | target | safe | 0.661555617916 | 0 | 1.38777878078e-17 | 1.37693675906e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | transient | cosine | position_control | safe | 0.534211731602 | -0.127343886315 | 1.38777878078e-17 | 2.51534904017e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | transient | linear | target | safe | 0.657860578715 | 0 | 1.38777878078e-17 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | transient | linear | position_control | safe | 0.50357769753 | -0.154282881185 | 2.77555756156e-17 | 2.40963932835e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | sustained | cosine | target | safe | 0.805998040204 | 0 | 1.38777878078e-17 | 1.18042511529e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | sustained | cosine | position_control | safe | 0.833025670323 | 0.0270276301189 | 2.77555756156e-17 | 5.42101086243e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | sustained | linear | target | safe | 0.802927941041 | 0 | 2.77555756156e-17 | 1.36067372647e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p239_028 | sustained | linear | position_control | safe | 0.829628036098 | 0.0267000950567 | 6.93889390391e-18 | 5.55111512313e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | transient | cosine | target | safe | 0.016121418163 | 0 | 3.46944695195e-18 | 3.03576608296e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | transient | cosine | position_control | safe | 0.0174485657898 | 0.00132714762683 | 3.46944695195e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | transient | linear | target | safe | 0.0161395153459 | 0 | 6.93889390391e-18 | 3.25260651746e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | transient | linear | position_control | safe | 0.0168243053343 | 0.000684789988367 | 1.73472347598e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | sustained | cosine | target | safe | 0.315638033879 | 0 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | sustained | cosine | position_control | safe | 0.324325680485 | 0.00868764660608 | 6.93889390391e-18 | 3.36102673471e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | sustained | linear | target | safe | 0.311418105325 | 0 | 3.46944695195e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_026 | sustained | linear | position_control | safe | 0.317362427903 | 0.00594432257799 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | transient | cosine | target | safe | 0.145394300504 | 0 | 6.93889390391e-18 | 6.07153216592e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | transient | cosine | position_control | safe | 0.224251178856 | 0.0788568783522 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | transient | linear | target | safe | 0.1427625547 | 0 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | transient | linear | position_control | safe | 0.225407890218 | 0.0826453355183 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | sustained | cosine | target | safe | 0.223050014048 | 0 | 1.38777878078e-17 | 6.16639985601e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | sustained | cosine | position_control | safe | 0.224251178856 | 0.00120116480835 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | sustained | linear | target | safe | 0.219788372397 | 0 | 6.93889390391e-18 | 5.23127548224e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p243_027 | sustained | linear | position_control | safe | 0.225407890218 | 0.00561951782098 | 6.93889390391e-18 | 6.93889390391e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | transient | cosine | target | safe | 0.679558344218 | 0 | 2.77555756156e-17 | 1.07607065619e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | transient | cosine | position_control | safe | 0.75130510589 | 0.071746761672 | 2.77555756156e-17 | 1.21430643318e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | transient | linear | target | safe | 0.673207776237 | 0 | 2.77555756156e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | transient | linear | position_control | safe | 0.744163793371 | 0.0709560171336 | 2.77555756156e-17 | 1.04083408559e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | sustained | cosine | target | safe | 1.31329036184 | 0 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | sustained | cosine | position_control | safe | 1.38516187031 | 0.0718715084658 | 2.77555756156e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | sustained | linear | target | safe | 1.27186599731 | 0 | 1.38777878078e-17 | 1.04083408559e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_028 | sustained | linear | position_control | safe | 1.34312213049 | 0.0712561331737 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | transient | cosine | target | safe | 0.101023530968 | 0 | 6.93889390391e-18 | 2.83586630741e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | transient | cosine | position_control | safe | 0.113851432558 | 0.01282790159 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | transient | linear | target | safe | 0.100413745039 | 0 | 3.46944695195e-18 | 2.60208521397e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | transient | linear | position_control | safe | 0.113480898581 | 0.0130671535423 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | sustained | cosine | target | safe | 0.430768101792 | 0 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | sustained | cosine | position_control | safe | 0.787348056144 | 0.356579954351 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | sustained | linear | target | safe | 0.433170698074 | 0 | 2.77555756156e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p244_034 | sustained | linear | position_control | safe | 0.768216238163 | 0.335045540088 | 6.93889390391e-18 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | transient | cosine | target | safe | 0.0701668241575 | 0 | 3.46944695195e-18 | 1.70761842166e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | transient | cosine | position_control | safe | 0.0702710947349 | 0.000104270577382 | 1.73472347598e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | transient | linear | target | safe | 0.0697448331939 | 0 | 3.46944695195e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | transient | linear | position_control | safe | 0.0699773856213 | 0.000232552427384 | 1.73472347598e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | sustained | cosine | target | safe | 0.221588525344 | 0 | 1.73472347598e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | sustained | cosine | position_control | safe | 0.149048052505 | -0.0725404728393 | 3.46944695195e-18 | 6.07153216592e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | sustained | linear | target | safe | 0.220117031311 | 0 | 3.46944695195e-18 | 1.73472347598e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_038 | sustained | linear | position_control | safe | 0.146297534444 | -0.0738194968665 | 6.93889390391e-18 | 3.46944695195e-18 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | transient | cosine | target | safe | 0.240477510348 | 0 | 1.38777878078e-17 | 1.29562159612e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | transient | cosine | position_control | safe | 0.173285924497 | -0.0671915858514 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | transient | linear | target | safe | 0.238867772016 | 0 | 1.38777878078e-17 | 1.30646361785e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | transient | linear | position_control | safe | 0.174036722012 | -0.0648310500049 | 1.38777878078e-17 | 1.21430643318e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | sustained | cosine | target | safe | 0.252349435375 | 0 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | sustained | cosine | position_control | safe | 0.057947073825 | -0.19440236155 | 2.77555756156e-17 | 1.29020058526e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | sustained | linear | target | safe | 0.248723730576 | 0 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |
| p250_050 | sustained | linear | position_control | safe | 0.056865492386 | -0.19185823819 | 1.38777878078e-17 | 1.38777878078e-17 | 1.52587890625e-05 | 1.52587890625e-05 | 3.0517578125e-05 | 3.0517578125e-05 |

## Identity/sham, float and quantization are distinct checks

All24originalfullWAV hashes match zerooperationsham fullWAV bytecopies exactly in RAM; no H1reencoding. Originalint16 samplesroundtrip exactly independently of fullWAVcontainer bytes. For all96 maskoutsideclean/noisy exactequality retained. Floatsharedresidual reconmax **2.77555756156e-17**, applieddeltamax **5.55111512313e-17**. SafeeditedPCM perarm ties-even error max **1.52587890625e-05** (≤halfLSB bound). Quantizedarms+rawsharedN reconmax **3.0517578125e-05**; independentlyquantizedN reconmax **3.0517578125e-05**. These are separately reported and never passed off as exactlyidentical/noisepure. RawN hasunknownpurity, may retain editedevent leakage; S0+N=X0 algebra doesn't establish naturallyabsentevent or truehypothesislabels.

## Preservation and interpretation boundaries

All11preparation/10source recordlocks and574oldI0/I1a/VB32/residual/cachefiles unchanged; originalmetricsSHA `24c0ee1297e261dbecc4cfa9b2f3441292e45e71709cb2fddb0b60462362464c` protected by uniquevalidator. Configfalsepermissionfields are historical snapshots, no gatepromotion. Allscientific/noise/split/traininggate values remainfalse; sourceidentityunknown. No crossutterance noise reuse, model/ASR/classifier/AUC/dprime, training or newaudiooutputs. Networkforbiddenattempts=[] in bothphases; bytes0 is refusalpolicy accounting not packetcapture. Only newdirwritten, no push. SourceWAVs not reread for reportfinalization.

Directoryexternal gitobservedchanges versuspreflight:
```json
{
  "added": [],
  "removed": []
}
```

Root may concurrently update/data/research or addreviewJSON; disclosedobservations are timestamped states, not unconditional entirefilesystem immutability. Bridge39452/listening16022 remainrunning. Main/validation do not prove Noisy contains discriminative information, eventidentity, naturalediting, leakageabsence or generalization. Position/family/sham controls are integrityconditions, not an experiment already passing shortcuttests. Allcandidate/frame/selectionrank/source/mask/payload hashes retained in originalmetrics; no array payload stored. Worker stops after technicalaudit, noI2/probe/classifier or furtherwaveform action.

## Lead-requested posthoc descriptive energy-mismatch check

Across **48 valid target/control × taper-family comparisons** (24targets×2families; dimensionless ratio, not48independent events), max |E_delta_control−E_delta_target|/E_delta_target = **0.8277770635004515 (82.77770635%)**, p244_034/sustained/cosine. This posthoc descriptor is not a preregistered success endpoint and not a technical failure; it shows closest **untapered** window-energy selection did not actually balance tapered edit energy. Thus controls cannot be called exactly energy-matched or used to upgrade a content-identifiability claim. All coordinates/rules/parameters and source files retained; no selection/mask/control rerun.

Reporting-only finalization initially had a Python comprehension SyntaxError before any report/provenance writes; corrected only the new non-frozen report script. Neither scientific main nor independent validator was rerun, no frozen/runtime/source file changed.
