# Online-Control Medium Pilot Findings

## A. Setup actually run
- Hosts: `nsgaii`, `moead`
- Problems: `zdt1`, `zdt2`, `zdt3`, `dtlz2`
- Policies: `fixed_sbx`, `fixed_de`, `adaptive_flat_operator`, `adaptive_flat_parameter`, `adaptive_hierarchical_joint`
- Seeds: `0-9` (10 seeds)
- Pilot budget: population size `40`, `800` evaluations, `n_var=12`, engine `numpy`
- Transfer budget: source `400` evaluations, target `400` evaluations, same hosts/problems/seeds
- Deviations from requested scope: none in host/problem/policy/seed coverage. A small runtime fix was required so MOEA/D pilot/transfer runs provide packaged weight-vector metadata for `dtlz2`.

## B. Best-fixed comparison
`adaptive_hierarchical_joint` beat the best fixed baseline in **7/8** `(host, problem)` cases, lost **1**, and tied **0**. The mean HV gap vs best fixed was **0.239**, the median HV gap was **0.242**, the mean runtime ratio was **2.942x**, and the median runtime ratio was **2.505x**.

The wins were not confined to a single host: **4/4** MOEA/D cases and **3/4** NSGA-II cases were wins. They were concentrated in the ZDT family (**6/6** ZDT host-problem cases), while `dtlz2` was mixed: a small MOEA/D win and one NSGA-II loss.

## C. Flat-vs-hierarchical comparison
Against `adaptive_flat_operator`, the hierarchical policy went **6/8** wins, **2** losses, **0** ties, with mean/median HV gaps of **0.110 / 0.089** and mean/median runtime ratios of **1.023x / 1.028x**.

Against `adaptive_flat_parameter`, the hierarchical policy went **7/8** wins, **1** losses, **0** ties, with mean/median HV gaps of **0.210 / 0.217** and mean/median runtime ratios of **1.036x / 1.022x**.

Overall, hierarchy looks meaningfully better on quality than both flat adaptive variants, but only by a moderate margin over `adaptive_flat_operator` and mainly outside `dtlz2`. The pairwise runtime overhead relative to the flat adaptive variants is small (roughly `1.02x-1.04x`), so the hierarchy-vs-flat tradeoff is much cleaner than the hierarchy-vs-fixed tradeoff.

## D. Concentration / decisiveness
For `adaptive_hierarchical_joint`, family choice is fairly decisive while prototype choice remains diffuse. The median dominant family share was **0.820**, median dominant prototype share was **0.369**, median regime concentration was **0.969**, median family switch count was **30.1**, and median prototype switch count was **110.1**.

The regime router was effectively locked into `EXPAND` on this unconstrained suite: mean regime shares were `repair=0.000`, `expand=0.967`, `refine=0.033`. So the current quality gains are coming much more from family/intent adaptation than from meaningful regime transitions.

## E. Dynamic heterogeneity
The fixed-policy landscape is not heterogeneous: the best fixed winner was always `fixed_sbx` across all **8/8** host-problem cases. That weakens the case that adaptation is exploiting host/problem-specific fixed-family winners.

There is still some dynamic structure inside adaptive control. The adaptive winner set had **2** members, with `adaptive_hierarchical_joint` winning every ZDT case and `adaptive_flat_operator` taking `dtlz2`. For the hierarchical policy, mean early-to-late family shift TVD was **0.129** and mean early-to-late prototype shift TVD was **0.296**, so prototype dynamics are more pronounced than family dynamics. Phase averages also show the family mix staying strongly DE-like while intent shares move: DE share goes from **0.743** early to **0.872** late, and exploratory intent rises from **0.332** to **0.486** while local-refine intent falls from **0.213** to **0.082**.

The honest read is that adaptation has something real to exploit, but in this pilot it is mostly operator-family plus prototype control inside a nearly constant `EXPAND` regime, not rich regime-level heterogeneity.

## F. Transfer
Warm-start transfer was **harmful overall and noisy**. Across **80** transfer comparisons, warm-start won **32**, lost **48**, and tied **0**, with mean/median HV deltas of **-0.116 / -0.133** and mean/median runtime ratios of **0.969x / 0.983x**.

Direction matters. `nsgaii -> moead` was close to neutral (**mean HV delta -0.018**, **21/40** wins), but `moead -> nsgaii` was clearly harmful (**mean HV delta -0.215**, **11/40** wins). By problem, `dtlz2` was mildly positive, while `zdt1-zdt3` were negative, especially `zdt3`.

## G. Honest go/no-go verdict
**WEAK_GO**

The quality signal is strong enough to keep going: `adaptive_hierarchical_joint` beat the best fixed baseline in **7/8** host-problem cases and beat both flat adaptive variants in **6/8** cases, with clear gains on the ZDT suite and only modest pairwise overhead vs the flat adaptive policies. But this is not a clean `GO` yet because the wall-clock cost is high (median **2.505x** vs best fixed), the regime layer is barely exercised on this unconstrained pilot (`EXPAND` about **0.967** of steps), and cross-host warm-start transfer is negative overall (**mean HV delta -0.116**). The research direction looks promising, but the next step should focus on reducing overhead and demonstrating genuine regime-level value, not just adding more control complexity.
