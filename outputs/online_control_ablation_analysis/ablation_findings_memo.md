# Online Control Ablation Findings

## Setup actually run
- zcat: hosts=['nsgaii', 'moead'], problems=['zcat1', 'zcat6', 'zcat14', 'zcat19'], variants=['fixed_sbx', 'fixed_de', 'adaptive_flat_operator', 'adaptive_flat_parameter', 'adaptive_hierarchical_joint', 'adaptive_hierarchical_joint_no_regime', 'adaptive_hierarchical_joint_fixed_family_sbx', 'adaptive_hierarchical_joint_fixed_family_de'], seeds=[0, 1, 2, 3, 4], population_size=40, max_evaluations=800, n_var=30
- anchor: hosts=['nsgaii', 'moead'], problems=['zdt1', 'zdt3', 'dtlz2'], variants=['fixed_sbx', 'fixed_de', 'adaptive_flat_operator', 'adaptive_flat_parameter', 'adaptive_hierarchical_joint', 'adaptive_hierarchical_joint_no_regime', 'adaptive_hierarchical_joint_fixed_family_sbx', 'adaptive_hierarchical_joint_fixed_family_de'], seeds=[0, 1, 2, 3, 4], population_size=40, max_evaluations=800, n_var=12

## A. Is the gain mostly prototype-driven?
`adaptive_hierarchical_joint_fixed_family_sbx` vs best fixed: wins=7/14, mean HV gap=0.5074, median HV gap=0.0135. `adaptive_hierarchical_joint` vs fixed-family-SBX ablation: wins=8/14, mean HV gap=-0.2515. `adaptive_hierarchical_joint_fixed_family_sbx` vs `adaptive_flat_parameter`: wins=0/14, mean HV gap=0.0000. This points to prototype adaptation on top of SBX as the main positive signal, not to family switching.

## B. Does regime-awareness matter on the tested suites?
`adaptive_hierarchical_joint` vs `adaptive_hierarchical_joint_no_regime`: wins=6/14, losses=5, ties=3, mean HV gap=0.0385, median HV gap=0.0000. The regime signal is present but weak relative to the prototype ablation.

## C. Does family adaptation matter beyond SBX-fixed prototype adaptation?
`adaptive_hierarchical_joint` vs `adaptive_hierarchical_joint_fixed_family_sbx`: wins=8/14, mean HV gap=-0.2515, median HV gap=0.0536. On this ablation, family switching does not add reliable value beyond SBX-fixed prototype adaptation.

## D. Does ZCAT reveal stronger heterogeneity than the anchor suite?
Hierarchical vs best fixed mean HV gap: ZCAT=0.3302, anchor=0.1567. Hierarchical phase family shift mean: ZCAT=0.1165, anchor=0.1156. Hierarchical phase intent shift mean: ZCAT=0.2182, anchor=0.2881. ZCAT amplifies the quality gap to best fixed, but not every heterogeneity proxy becomes stronger than on the anchor suite.

## E. Where does the runtime overhead come from?
Hierarchical control total share of runtime: median=0.0789, mean=0.0737. Decode share median=0.0434, trace share median=0.0020, evaluation share median=0.0536. The overhead is not dominated by tracing; the largest measured cost sits in host-side survival/update work, with decode the main controller-side cost.

## F. Decisiveness and concentration
Median dominant family share=0.8109. Median dominant prototype share=0.3554.

## Recommended paper direction
WEAK_GO_PIVOT_TO_PROTOTYPE_STORY

The recommendation is based on the fixed-family SBX ablation, the no-regime ablation, the ZCAT-vs-anchor sensitivity comparison, and the measured runtime profile rather than on headline quality alone.