# Semantic Prototype SBX Confirmatory Findings

Setup run: hosts=['nsgaii', 'moead'], zcat_problems=['zcat1', 'zcat6', 'zcat14', 'zcat19'], anchor_problems=['zdt1', 'zdt3', 'dtlz2'], variants=['fixed_sbx', 'semantic_prototype_sbx', 'adaptive_hierarchical_joint', 'adaptive_hierarchical_joint_no_regime'], seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], population_size=40, max_evaluations=800.

## A. Core claim supported
The strongest evidence-backed claim is not that the method wins uniformly on every host, but that semantic prototype adaptation on top of fixed `SBX_LIKE` produces a real quality signal overall, beating `fixed_sbx` in 9/14 host-problem cases with mean/median HV gap 0.3655 / 0.0169. The confirmatory run therefore supports a prototype-centric paper story more directly than a broader family-switching story, while also showing that the current evidence is host-asymmetric rather than uniformly reproduced.

## B. Main quantitative results
Against `fixed_sbx`, `semantic_prototype_sbx` went 9 wins / 5 losses / 0 ties overall, with mean/median runtime ratio 2.6497x / 1.9517x.
ZCAT-first contrast: wins=5/8, mean HV gap=0.6353, median runtime ratio=1.8859x. Anchor contrast: wins=4/6, mean HV gap=0.0058, median runtime ratio=2.5376x.
Host contrast: NSGA-II mean HV gap vs `fixed_sbx`=-0.1852 over 7 cases; MOEA/D mean HV gap=0.9162 over 7 cases. This means the fixed-family prototype signal is strong on MOEA/D but not yet confirmed against `fixed_sbx` on NSGA-II.
Against the full hierarchical reference, `semantic_prototype_sbx` went 8 wins / 6 losses / 0 ties, with mean/median HV gap 0.2238 / 0.0620.

## C. Why not the broader hierarchical story
The full hierarchical controller is not the main story in this confirmatory run because the prototype-SBX method already matches or exceeds it (overall mean HV gap vs hierarchical 0.2238), while the no-regime ablation remains close (hierarchical vs no-regime mean HV gap 0.0793). That keeps family switching and regime-awareness in the role of references rather than flagship claims.

## D. Practical implication
The cleanest positioning is a semantic parameter-control method implemented as a lightweight host-agnostic adaptation layer: the host keeps a strong fixed operator family, while online control adapts interpretable semantic intent prototypes over time.

## E. Final recommendation
WEAK_GO_TEVC_IF_STRONGLY_REFRAMED

Prototype decisiveness remains moderate rather than collapsed, with median dominant prototype share 0.4222 and median prototype switches 84.0000. Phase dynamics are still real: early dominant prototype=exploratory, late dominant prototype=exploratory, mean early-to-late prototype shift TVD=0.1147. Runtime overhead remains mostly host-side rather than trace/controller-side, with control share median=0.0784, decode share median=0.0521, trace share median=0.0023, and host-pipeline share median=0.5197.