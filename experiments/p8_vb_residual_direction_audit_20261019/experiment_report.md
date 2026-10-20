# Fixed residual direction versus step length: existing full PCM audit

## Scope and execution

Research Lead-approved new CPU-only scientific diagnosis, not reopening generation, ASR, quality measurement, selector or training. All32 prescribed existing F/OFF pairs and8 scaled-clean PCM controls,72 unique WAVs. Full16k mono PCM16, original lengths and amplitudes, no centering/resampling/lag/frequency/crop/gainfit. Original noisy/noise/raw.npy not decoded; raw and all other old artifacts only hashed for preservation. No new waveform written, including no oracle blend.

Actual main command `python3 -B /data/coding/sgmse/experiments/p8_vb_residual_direction_audit_20261019/run_audit.py`, unique session95349 exit0, elapsed1.984868038s; exclusive STARTED UTC2026-10-19T07:16:36.325637+00:00 records actual PID21031 (not claimed independently observed). Unique independent validation `python3 -B /data/coding/sgmse/experiments/p8_vb_residual_direction_audit_20261019/validate_audit.py`, session76201 exit0, elapsed11.701257314s. Main decodes72, validator additional72, total **144 real PCM decodes**. Cached8 clean reused across conditions. No external array/audio/model libraries; standard library wave/struct/math.fsum. Main60s/2MiB cap, validator separate60s. Counters models/ASR/newquality/network/training/audiooutputs/resampling/retries all0.

Current AGENTS/lowercase research entry/frozen analysis/config read; full attachment4551B SHAa235e23451a88273189fb9d649904940a868725db05f3dd38983e49290b73a6b. Self readonly SQLite provider codex_tools_compat; actual bridge39452/listening16022 alive, no model process. Current dirty tree retained; no push.

## Core quantities

S scaled-clean PCM, F frontend PCM, B OFF PCM; e=F−S,d=B−F, P=mean(S²), E0=mean(e²), A=mean(e*d), D=mean(d²), E1=mean((B−S)²). Full-line quadratic E(alpha)=E0+2alphaA+alpha²D. Clean-aware alpha=clip(−A/D,0,1), no output synthesis. Common alpha minimizes equal-case P-normalized MSE. This oracle is not deployable and not selected by ASR or quality.

All32 optimal alpha values are **strictly interior**, range0.0181861678–0.7615874866. Thus every saved full direction has a negative initial MSE slope yet its MSE-minimum on this segment lies before full OFF. This describes PCM pointwise squared error only; it is not actual phone restoration, a tested control algorithm, or user-perceived benefit.

All-case equalcase normalizedMSE: F **0.263909585992**, OFF **0.317828917582**, common-alpha oracle **0.256215524323** at alpha **0.261108468195**, mean per-case oracle **0.253907507159**. Common-alpha reduction relative F is 2.9154157624412873%; this is an in-sample clean-aware MSE bound, not an evaluated blend or new quality/content endpoint. No threshold-based pass or upgrade to training.

## Gain versus non-gain

Projection uses beta=<d,F>/<F,F>, d_gain=betaF, d_rest=d−betaF, **not fitting any gain to clean**. Gain/rest contributions are each 2mean(e*dpart)+mean(dpart²); cross orthogonality and sum identity independently checked. Non-gain energy fraction arithmetic case mean **0.983988491194** (not pooled energy ratio). Equalcase normalized MSE contribution: gain **−0.007727041569**, rest **+0.061646373160**, sum **+0.053919331590**. Most modification energy is not a global amplitude scale; this alone cannot identify residual noise, phase/time effects, detail or phone correctness.

**Do not equate MSE decomposition with SI-SDR explanation.** Pure nonzero global scaling leaves SI-SDR invariant yet changes amplitude-sensitive MSE. Original saved OFF−front SI-SDR mean−1.326094626dB is merely referenced, never recomputed or optimized over alpha. No STOI/PESQ/ASR calls; no invented human listening.

## All21 strata: equalcase clean-power normalized MSE

| Stratum | n | common alpha | F | OFF | common oracle | mean case oracle | alpha zero/interior/one | mean rest-energy fraction | gain contribution/P | rest contribution/P | saved SI-SDR OFF−F dB |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| all | 32 | 0.2611084682 | 0.263909586 | 0.3178289176 | 0.2562155243 | 0.2539075072 | 0/32/0 | 0.9839884912 | -0.007727041569 | 0.06164637316 | -1.326094626 |
| noise_N1 | 16 | 0.2542453317 | 0.3857226151 | 0.4597444057 | 0.3759876637 | 0.3720668557 | 0/16/0 | 0.9737432132 | -0.01471956216 | 0.08874135274 | -1.67971917 |
| noise_N2 | 16 | 0.2748704061 | 0.1420965569 | 0.1759134295 | 0.136422067 | 0.1357481586 | 0/16/0 | 0.9942337692 | -0.0007345209797 | 0.03455139358 | -0.9724700822 |
| snr_-18 | 16 | 0.2602739007 | 0.4788472456 | 0.5761426595 | 0.465100234 | 0.4606671394 | 0/16/0 | 0.9700909865 | -0.0154414452 | 0.1127368591 | -1.761424467 |
| snr_0 | 16 | 0.268544339 | 0.0489719264 | 0.05951517564 | 0.04732941393 | 0.04714787492 | 0/16/0 | 0.9978859959 | -1.263794128e-05 | 0.01055588719 | -0.8907647851 |
| N1_-18 | 8 | 0.252048811 | 0.714472921 | 0.8487230744 | 0.6972745266 | 0.6896086655 | 0/8/0 | 0.9501633181 | -0.02937301078 | 0.1636231641 | -2.314975589 |
| N1_0 | 8 | 0.2737525516 | 0.05697230919 | 0.07076573705 | 0.05468789488 | 0.05452504589 | 0/8/0 | 0.9973231084 | -6.611353686e-05 | 0.01385954139 | -1.044462752 |
| N2_-18 | 8 | 0.2767505705 | 0.2432215701 | 0.3035622447 | 0.2328709384 | 0.2317256133 | 0/8/0 | 0.9900186549 | -0.001509879614 | 0.06185055417 | -1.207873346 |
| N2_0 | 8 | 0.2580085561 | 0.0409715436 | 0.04826461424 | 0.03996843344 | 0.03977070396 | 0/8/0 | 0.9984488834 | 4.083765429e-05 | 0.007252232985 | -0.7370668186 |
| speaker_p227 | 8 | 0.2116554969 | 0.2690099159 | 0.342478983 | 0.2633027308 | 0.2627822361 | 0/8/0 | 0.9889928711 | -0.003583873952 | 0.07705294103 | -1.375907712 |
| speaker_p228 | 8 | 0.3798805532 | 0.2776071727 | 0.3027987448 | 0.2624748333 | 0.2574077891 | 0/8/0 | 0.9742687095 | -0.01555583817 | 0.04074741024 | -1.080469174 |
| speaker_p230 | 8 | 0.260458972 | 0.2817308312 | 0.3424192069 | 0.2731372486 | 0.2721642314 | 0/8/0 | 0.9772663284 | -0.007955436971 | 0.06864381269 | -1.414703649 |
| speaker_p231 | 8 | 0.1954492787 | 0.2272904241 | 0.2836187356 | 0.2237577364 | 0.2232757721 | 0/8/0 | 0.9954260557 | -0.003813017186 | 0.06014132867 | -1.43329797 |
| utterance_p227_030 | 4 | 0.2125641534 | 0.2944984508 | 0.3699783867 | 0.288565911 | 0.2885250748 | 0/4/0 | 0.982515821 | -0.007390218866 | 0.08287015485 | -1.512113401 |
| utterance_p227_031 | 4 | 0.2106894402 | 0.2435213811 | 0.3149795793 | 0.238039327 | 0.2370393973 | 0/4/0 | 0.9954699212 | 0.0002224709625 | 0.07123572721 | -1.239702023 |
| utterance_p228_026 | 4 | 0.5326112213 | 0.3060460748 | 0.2996669538 | 0.2783010958 | 0.2728228176 | 0/4/0 | 0.9592487308 | -0.02491229654 | 0.01853317561 | -0.7439976741 |
| utterance_p228_032 | 4 | 0.2464054784 | 0.2491682707 | 0.3059305358 | 0.2423732533 | 0.2419927605 | 0/4/0 | 0.9892886882 | -0.006199379794 | 0.06296164487 | -1.416940674 |
| utterance_p230_027 | 4 | 0.2907792134 | 0.2548682016 | 0.3073850421 | 0.2442563669 | 0.2426278631 | 0/4/0 | 0.9834336967 | -0.0005517888681 | 0.05306862932 | -1.034318333 |
| utterance_p230_029 | 4 | 0.2306939609 | 0.3085934608 | 0.3774533718 | 0.301789484 | 0.3017005997 | 0/4/0 | 0.9710989602 | -0.01535908507 | 0.08421899607 | -1.795088965 |
| utterance_p231_029 | 4 | 0.2358292274 | 0.2049562703 | 0.2446674389 | 0.2007761078 | 0.2001009855 | 0/4/0 | 0.9982117861 | -0.001265066563 | 0.0409762352 | -1.033646345 |
| utterance_p231_032 | 4 | 0.1678061622 | 0.249624578 | 0.3225700323 | 0.2465329126 | 0.2464505588 | 0/4/0 | 0.9926403253 | -0.006360967809 | 0.07930642214 | -1.832949595 |

## Full32 per-case audit

| Case | alpha* | E0/P | E1/P | Eoracle/P | A/P | D/P | beta existing-delta/F | rest-energy fraction | gain contribution/P | rest contribution/P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p227_030_N1_snr_m18.wav | 0.20413888256 | 0.73365255165 | 0.92857165826 | 0.7199251611 | -0.067245349708 | 0.32940980603 | 0.40482605762 | 0.9311502235 | -0.029121049302 | 0.22404015592 |
| p227_030_N1_snr_p0.wav | 0.27175147806 | 0.062570775812 | 0.080263634918 | 0.059708550096 | -0.010532512045 | 0.038757883196 | 0.0031355726375 | 0.99977205175 | -0.0001129419869 | 0.017805801093 |
| p227_030_N2_snr_m18.wav | 0.21723540519 | 0.31259338633 | 0.38986144316 | 0.30614566615 | -0.029680797991 | 0.13662965282 | 0.0071008290082 | 0.99976991122 | -0.00042249734712 | 0.077690554183 |
| p227_030_N2_snr_p0.wav | 0.20487503526 | 0.069177089219 | 0.081216810606 | 0.068320922039 | -0.0041789727039 | 0.020397666795 | -0.0037570799788 | 0.99937109766 | 9.5613170746e-05 | 0.011944108217 |
| p227_031_N1_snr_m18.wav | 0.16002211923 | 0.65096719918 | 0.87719766562 | 0.64244737822 | -0.053241520582 | 0.3327135076 | 0.031845441813 | 0.99892364403 | 0.0004885223651 | 0.22574194407 |
| p227_031_N1_snr_p0.wav | 0.3807746852 | 0.056310878775 | 0.065120086731 | 0.05095445999 | -0.014067160956 | 0.036943529869 | -0.021279760885 | 0.98886690071 | 0.0011647387565 | 0.0076444692 |
| p227_031_N2_snr_m18.wav | 0.32985995705 | 0.21502189131 | 0.25269775527 | 0.20297469791 | -0.036522145677 | 0.11072015531 | 0.008131611241 | 0.999598918 | -0.00087758973352 | 0.038553453693 |
| p227_031_N2_snr_p0.wav | 0.018186167792 | 0.051785555111 | 0.064902809456 | 0.051781053001 | -0.00024755680778 | 0.01361236796 | -0.0089143619933 | 0.99449022222 | 0.00011421246203 | 0.013003041883 |
| p228_026_N1_snr_m18.wav | 0.76158748663 | 0.88060775248 | 0.7750066452 | 0.76353357626 | -0.15372387057 | 0.20184663387 | 1.2676435068 | 0.84840326809 | -0.096608798576 | -0.0089923086997 |
| p228_026_N1_snr_p0.wav | 0.33077486575 | 0.068803730666 | 0.076447814665 | 0.066332600676 | -0.0074707308398 | 0.022585545679 | 0.0073760119222 | 0.99787430275 | -0.00031156295588 | 0.0079556469548 |
| p228_026_N2_snr_m18.wav | 0.28426807917 | 0.22458175787 | 0.29178967317 | 0.21199446873 | -0.044279643257 | 0.15576720181 | 0.045903016704 | 0.99073794378 | -0.0027214695625 | 0.069929384862 |
| p228_026_N2_snr_p0.wav | 0.26265117641 | 0.05019105802 | 0.055423682281 | 0.049430624641 | -0.0028952216762 | 0.011023067613 | 0.00049285748162 | 0.99997940865 | -7.3550765117e-06 | 0.005239979337 |
| p228_032_N1_snr_m18.wav | 0.27802067086 | 0.75391073321 | 0.88066393273 | 0.73184234707 | -0.07937678184 | 0.2855067632 | 0.28674963496 | 0.96544408814 | -0.026293926251 | 0.15304712578 |
| p228_032_N1_snr_p0.wav | 0.1588868761 | 0.049801659817 | 0.063618344722 | 0.049290388437 | -0.0032178326614 | 0.020252350228 | 0.0091630015465 | 0.99634463778 | -0.00055344423158 | 0.014370129137 |
| p228_032_N2_snr_m18.wav | 0.17154663059 | 0.16625592136 | 0.25004363969 | 0.16250238151 | -0.021880580476 | 0.12754887928 | 0.025149312136 | 0.99557227597 | 0.0020525929219 | 0.081735125402 |
| p228_032_N2_snr_p0.wav | 0.40624458092 | 0.026704768387 | 0.029396225954 | 0.024335925138 | -0.0058310765504 | 0.014353610668 | -0.0017411049976 | 0.99979375082 | -2.7416145394e-06 | 0.0026941991815 |
| p230_027_N1_snr_m18.wav | 0.21305945122 | 0.69579530708 | 0.87054317607 | 0.68197264865 | -0.064877001919 | 0.30450187283 | 0.18383132374 | 0.96895832216 | 0.0049479522381 | 0.16979991675 |
| p230_027_N1_snr_p0.wav | 0.23980140958 | 0.051429059148 | 0.069665341525 | 0.049413920945 | -0.0084033626249 | 0.035043007627 | 0.006637496388 | 0.99885436471 | -0.00020754854239 | 0.01844383092 |
| p230_027_N2_snr_m18.wav | 0.46485769874 | 0.2481897317 | 0.25873688659 | 0.21576210338 | -0.069758182788 | 0.15006352046 | 0.0904749579 | 0.96623972755 | -0.0069583741782 | 0.017505529066 |
| p230_027_N2_snr_p0.wav | 0.23676274015 | 0.024058708672 | 0.030594764204 | 0.02336277942 | -0.0029393529214 | 0.012414761375 | -0.0020136185128 | 0.99968237226 | 1.0815010199e-05 | 0.0065252405216 |
| p230_029_N1_snr_m18.wav | 0.21382646914 | 0.82651653069 | 1.0129249018 | 0.81162535858 | -0.069641388042 | 0.32569114723 | 0.74281155542 | 0.91233144927 | -0.061873738974 | 0.24828211012 |
| p230_029_N1_snr_p0.wav | 0.30737690089 | 0.058274771674 | 0.070433786199 | 0.055292807161 | -0.0097013292276 | 0.03156167298 | 0.0044618652793 | 0.99943651192 | -0.00019814905296 | 0.012357163578 |
| p230_029_N2_snr_m18.wav | 0.25398994832 | 0.31452242326 | 0.38293549326 | 0.30555248836 | -0.03531610191 | 0.13904527382 | 0.071982221315 | 0.97586429277 | 0.00063576987261 | 0.067777300124 |
| p230_029_N2_snr_p0.wav | 0.21970328135 | 0.035060117407 | 0.043519305717 | 0.034331744522 | -0.0033152571998 | 0.015089702709 | -0.0070887613594 | 0.99676358681 | -2.2214046163e-07 | 0.00845941045 |
| p231_029_N1_snr_m18.wav | 0.11697911643 | 0.51385647616 | 0.59903488567 | 0.5123348998 | -0.013007247791 | 0.11119290509 | 0.028046772178 | 0.99754062013 | -0.003610950058 | 0.088789359568 |
| p231_029_N1_snr_p0.wav | 0.31216082345 | 0.05171862221 | 0.061554923267 | 0.049167258131 | -0.008173235994 | 0.026182773045 | 0.0064666445056 | 0.9985441616 | -0.0001995419615 | 0.010035843018 |
| p231_029_N2_snr_m18.wav | 0.29355899084 | 0.22093333181 | 0.2824942091 | 0.2080843237 | -0.043769765234 | 0.14910040776 | 0.020876874797 | 0.99794877941 | -0.0013090413496 | 0.062869918638 |
| p231_029_N2_snr_p0.wav | 0.41994259003 | 0.033316650914 | 0.035585737616 | 0.030817460167 | -0.0059512676405 | 0.014171621983 | -0.0041924987672 | 0.99881358318 | 5.9267115575e-05 | 0.0022098195866 |
| p231_032_N1_snr_m18.wav | 0.16283640086 | 0.66047681792 | 0.84584172947 | 0.65318795402 | -0.044761882837 | 0.27488867722 | 0.18085086669 | 0.97855492947 | -0.022912097689 | 0.20827700924 |
| p231_032_N1_snr_p0.wav | 0.15957962686 | 0.056868975452 | 0.079021964351 | 0.056040381648 | -0.0051923533095 | 0.032537695518 | 0.0062610275322 | 0.9988919357 | -0.00011045832019 | 0.022263447219 |
| p231_032_N2_snr_m18.wav | 0.16031967707 | 0.24367411737 | 0.31993885724 | 0.24078877691 | -0.017997419399 | 0.11225957866 | 0.030911509449 | 0.99441739079 | -0.0024784275338 | 0.078743167395 |
| p231_032_N2_snr_p0.wav | 0.29476749692 | 0.03747840108 | 0.045477578089 | 0.035785122726 | -0.0057444540923 | 0.019488085194 | -0.0051527284651 | 0.99869704539 | 5.7112307273e-05 | 0.0079420647017 |

## Independent verification and preservation

Independent validator does not import original analysis/grouping functions. It decodes original72PCM again, directly constructs scalar residual squares from F+alpha(B−F)−S in memory at endpoints/per-case/common alpha, and directly computes residual-energy differences for gain/rest. It independently rebuilds21 groups and all32/21 summaries, verifies exact metadata/size/hashes/headers, P-normalization, alpha cases, quadratic identity, part contributions/orthogonality and the original saved SI-SDR references. Mathematical specification shared, implementation separate; no new optimization grid/bootstrap or saved audio. Numerical tolerance is floating identity checking, not a scientific selection threshold.

497 original/source/config/plan files hash-identical pre/post/current. Source72 header/size/hash locks rechecked after main; validator verifies them anew. Original metrics SHA577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04 and input registry cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998 preserved. Main audit metrics SHA 06603c8fef1a7ee7b5a7bed89b159f6f97036e06e7d04c241524190892fe3e17. All old reports/metadata/input/model outputs preserved. Current git outside newdir unchanged. No old artifact rewritten or newmodel_started.

## Limits and disposition

Eight exploratory train sentences/four speakers, two historical noises, repeated conditions; no independent confirmation/global uncertainty claim. Normalized MSE is amplitude-sensitive and not WER, SI-SDR or naturalness. Pointwise non-gain residual is not a phone classifier. Quantized saved PCM is the fixed observation domain, not preclip raw. Clean-aware optimal alpha does not establish an observable selector, generalization or innovation; historical oracle/anchor routes remain closed. Worker stops after this audit and independent validation, without outputblend/listening/ASR/quality/seed/window extension or training. Lead decides the next scientific question separately.
