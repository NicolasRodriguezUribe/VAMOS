# Final Semantic Prototype SBX Confirmatory Findings

## 1. Core supported claim
The strongest supported claim is that semantic prototype adaptation over a fixed `SBX_LIKE` family is the best-supported method in the current line: it beats `fixed_sbx` in 9/14 host-problem cases, beats the broader `adaptive_hierarchical_joint` reference overall, and concentrates most of its quality signal on the ZCAT-first suite rather than on the small classical anchor.

## 2. Experimental setup actually run
Hosts: ['nsgaii', 'moead']. ZCAT-first problems: ['zcat1', 'zcat6', 'zcat14', 'zcat19']. Anchor problems: ['zdt1', 'zdt3', 'dtlz2']. Variants: ['fixed_sbx', 'semantic_prototype_sbx', 'adaptive_hierarchical_joint', 'adaptive_hierarchical_joint_no_regime']. Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Population size: 40. Max evaluations: 800.

## 3. Main result vs fixed SBX
Overall, `semantic_prototype_sbx` went 9 wins / 5 losses / 0 ties against `fixed_sbx`, with mean/median HV gap 0.3655 / 0.0169 and mean/median runtime ratio 2.7442x / 1.8330x. Prototype behavior stayed adaptive rather than collapsing to one intent: median dominant prototype share was 0.4222, median prototype switches were 84.0, and mean early-to-late prototype shift TVD was 0.1147.

## 4. Host-wise interpretation
The observed effect is asymmetric by host. On MOEA/D, `semantic_prototype_sbx` went 7/7 against `fixed_sbx` with mean HV gap 0.9162. On NSGA-II, it only went 2/7 with mean HV gap -0.1852. Based only on observed evidence, this suggests that the prototype controller is especially useful in MOEA/D, while on NSGA-II the fixed SBX baseline remains harder to beat. The fact that `semantic_prototype_sbx` still beats the full hierarchical reference overall and on NSGA-II indicates that the weak NSGA-II result is more consistent with a strong fixed baseline than with the prototype policy being uniformly ineffective.

## 5. ZCAT-first interpretation
ZCAT is clearly more revealing than the anchor suite for this method. On ZCAT, `semantic_prototype_sbx` went 5/8 against `fixed_sbx` with mean/median HV gap 0.6353 / 0.5000. On the anchor, it went 4/6 with mean/median HV gap 0.0058 / 0.0020. The anchor signal is nearly flat, while ZCAT carries most of the positive evidence, which supports a ZCAT-first paper framing rather than a classical-suite-first framing.

## 6. Why the broader hierarchical story is not the main paper
`semantic_prototype_sbx` remains better than `adaptive_hierarchical_joint` overall, with 8 wins / 6 losses / 0 ties and mean/median HV gap 0.2238 / 0.0620. Meanwhile, the hierarchical no-regime reference stays close, with hierarchical vs no-regime mean HV gap only 0.0793. That means the main gains do not require a family-switching or regime-centric story.

## 7. Overhead interpretation
The remaining weakness is runtime overhead versus `fixed_sbx`, not absence of quality signal. The overall median runtime ratio versus `fixed_sbx` is 1.8330x. The controller-side shares are comparatively small: control=0.0961, decode=0.0636, trace=0.0020. The larger measured cost is host-side, with host-pipeline share=0.5166 and survival/update share=0.2847. This makes the overhead look more like a host-level engineering problem than a trace/controller tax, but it is not something that disappears just by disabling logging.

## 8. Final recommendation
WEAK_GO_TEVC_IF_STRONGLY_REFRAMED