# GCES 2-objective ZCAT survival-only ablation

This report describes a survival-only ablation. All four algorithms reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the GCES variants.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, gces-noComp, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- |
| zcat1 | 0.981731 | 0.986292 | 0.983564 | 0.983460 |
| zcat2 | 0.991373 | 0.993167 | 0.992147 | 0.992425 |
| zcat3 | 0.981568 | 0.985894 | 0.983868 | 0.983779 |
| zcat4 | 0.981568 | 0.985894 | 0.983869 | 0.983779 |
| zcat5 | 0.994124 | 0.995465 | 0.995305 | 0.995301 |
| zcat6 | 0.994481 | 0.995720 | 0.995249 | 0.995187 |
| zcat7 | 0.989018 | 0.991583 | 0.991373 | 0.990975 |
| zcat8 | 0.967947 | 0.972198 | 0.975612 | 0.976081 |
| zcat9 | 0.967941 | 0.972193 | 0.975607 | 0.976076 |
| zcat10 | 0.946007 | 0.950280 | 0.956072 | 0.955414 |
| zcat11 | 0.988725 | 0.991733 | 0.990634 | 0.990728 |
| zcat12 | 0.988428 | 0.991270 | 0.990488 | 0.990517 |
| zcat13 | 1.015344 | 1.017635 | 1.017026 | 1.016928 |
| zcat14 | 0.975055 | 0.980733 | 0.978889 | 0.979122 |
| zcat15 | 0.988429 | 0.991271 | 0.990488 | 0.990518 |
| zcat16 | 0.986793 | 0.989628 | 0.989972 | 0.990009 |
| zcat17 | 0.994120 | 0.995461 | 0.995301 | 0.995298 |
| zcat18 | 0.989018 | 0.991582 | 0.991372 | 0.990975 |
| zcat19 | 0.984635 | 0.988662 | 0.986068 | 0.986111 |
| zcat20 | 0.989016 | 0.991581 | 0.991371 | 0.990974 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- |
| zcat1 | 0.007839 | 0.005979 | 0.007352 | 0.007221 |
| zcat2 | 0.006222 | 0.004981 | 0.005553 | 0.005492 |
| zcat3 | 0.007943 | 0.006109 | 0.007186 | 0.007079 |
| zcat4 | 0.007915 | 0.006094 | 0.007167 | 0.007126 |
| zcat5 | 0.004491 | 0.003509 | 0.003990 | 0.003888 |
| zcat6 | 0.002882 | 0.002351 | 0.002859 | 0.002888 |
| zcat7 | 0.004714 | 0.003763 | 0.003903 | 0.004071 |
| zcat8 | 0.006383 | 0.005494 | 0.004859 | 0.005167 |
| zcat9 | 0.006371 | 0.005497 | 0.004900 | 0.005144 |
| zcat10 | 0.003497 | 0.003385 | 0.003441 | 0.003392 |
| zcat11 | 0.005266 | 0.003917 | 0.004331 | 0.004304 |
| zcat12 | 0.005507 | 0.004130 | 0.004439 | 0.004442 |
| zcat13 | 0.004038 | 0.003005 | 0.003090 | 0.003206 |
| zcat14 | 0.007444 | 0.005902 | 0.006471 | 0.006312 |
| zcat15 | 0.005556 | 0.004134 | 0.004431 | 0.004436 |
| zcat16 | 0.004940 | 0.003826 | 0.003782 | 0.003720 |
| zcat17 | 0.004507 | 0.003512 | 0.004004 | 0.003880 |
| zcat18 | 0.004708 | 0.003744 | 0.003917 | 0.004068 |
| zcat19 | 0.006491 | 0.004908 | 0.006049 | 0.006027 |
| zcat20 | 0.004717 | 0.003763 | 0.003911 | 0.004053 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | gces vs nsgaii | 19/0/2 | 16/0/5 | 16/3/2 |
| zcat1 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | gces-noGeo vs nsgaii | 19/0/2 | 17/0/4 | 17/2/2 |
| zcat1 | gces vs gces-noComp | 1/0/20 | 0/0/21 | 0/1/20 |
| zcat1 | gces vs gces-noGeo | 11/0/10 | 12/0/9 | 9/5/7 |
| zcat2 | gces vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat2 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | gces-noGeo vs nsgaii | 20/0/1 | 19/0/2 | 19/1/1 |
| zcat2 | gces vs gces-noComp | 1/0/20 | 0/0/21 | 0/1/20 |
| zcat2 | gces vs gces-noGeo | 12/0/9 | 15/0/6 | 12/3/6 |
| zcat3 | gces vs nsgaii | 20/0/1 | 19/0/2 | 19/1/1 |
| zcat3 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | gces vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | gces vs gces-noGeo | 10/0/11 | 11/0/10 | 9/3/9 |
| zcat4 | gces vs nsgaii | 20/0/1 | 19/0/2 | 19/1/1 |
| zcat4 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | gces vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | gces vs gces-noGeo | 10/0/11 | 12/0/9 | 9/4/8 |
| zcat5 | gces vs nsgaii | 21/0/0 | 17/0/4 | 17/4/0 |
| zcat5 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | gces-noGeo vs nsgaii | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat5 | gces vs gces-noComp | 4/0/17 | 1/0/20 | 1/3/17 |
| zcat5 | gces vs gces-noGeo | 10/0/11 | 13/0/8 | 8/7/6 |
| zcat6 | gces vs nsgaii | 19/0/2 | 8/0/13 | 8/11/2 |
| zcat6 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | gces-noGeo vs nsgaii | 18/0/3 | 9/0/12 | 7/13/1 |
| zcat6 | gces vs gces-noComp | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat6 | gces vs gces-noGeo | 8/0/13 | 9/0/12 | 4/9/8 |
| zcat7 | gces vs nsgaii | 20/0/1 | 18/0/3 | 18/2/1 |
| zcat7 | gces-noComp vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat7 | gces-noGeo vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat7 | gces vs gces-noComp | 6/0/15 | 4/0/17 | 4/2/15 |
| zcat7 | gces vs gces-noGeo | 7/0/14 | 8/0/13 | 7/1/13 |
| zcat8 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | gces vs gces-noComp | 21/0/0 | 15/0/6 | 15/6/0 |
| zcat8 | gces vs gces-noGeo | 9/0/12 | 5/0/16 | 4/6/11 |
| zcat9 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | gces vs gces-noComp | 21/0/0 | 14/0/7 | 14/7/0 |
| zcat9 | gces vs gces-noGeo | 9/0/12 | 5/0/16 | 4/6/11 |
| zcat10 | gces vs nsgaii | 21/0/0 | 12/0/9 | 12/9/0 |
| zcat10 | gces-noComp vs nsgaii | 14/0/7 | 15/0/6 | 13/3/5 |
| zcat10 | gces-noGeo vs nsgaii | 20/0/1 | 10/0/11 | 10/10/1 |
| zcat10 | gces vs gces-noComp | 20/0/1 | 9/0/12 | 8/13/0 |
| zcat10 | gces vs gces-noGeo | 11/0/10 | 8/0/13 | 6/7/8 |
| zcat11 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | gces vs gces-noComp | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat11 | gces vs gces-noGeo | 12/0/9 | 12/0/9 | 11/2/8 |
| zcat12 | gces vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | gces-noComp vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | gces-noGeo vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | gces vs gces-noComp | 3/0/18 | 4/0/17 | 3/1/17 |
| zcat12 | gces vs gces-noGeo | 11/0/10 | 10/0/11 | 10/1/10 |
| zcat13 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | gces vs gces-noComp | 1/0/20 | 3/0/18 | 1/2/18 |
| zcat13 | gces vs gces-noGeo | 8/0/13 | 10/0/11 | 8/2/11 |
| zcat14 | gces vs nsgaii | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat14 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | gces-noGeo vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat14 | gces vs gces-noComp | 2/0/19 | 3/0/18 | 1/3/17 |
| zcat14 | gces vs gces-noGeo | 14/0/7 | 14/0/7 | 12/4/5 |
| zcat15 | gces vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | gces-noComp vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | gces-noGeo vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | gces vs gces-noComp | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat15 | gces vs gces-noGeo | 11/0/10 | 10/0/11 | 10/1/10 |
| zcat16 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | gces vs gces-noComp | 13/0/8 | 12/0/9 | 12/1/8 |
| zcat16 | gces vs gces-noGeo | 11/0/10 | 12/0/9 | 11/1/9 |
| zcat17 | gces vs nsgaii | 21/0/0 | 17/0/4 | 17/4/0 |
| zcat17 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | gces-noGeo vs nsgaii | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat17 | gces vs gces-noComp | 4/0/17 | 1/0/20 | 1/3/17 |
| zcat17 | gces vs gces-noGeo | 10/0/11 | 13/0/8 | 8/7/6 |
| zcat18 | gces vs nsgaii | 20/0/1 | 18/0/3 | 18/2/1 |
| zcat18 | gces-noComp vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat18 | gces-noGeo vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat18 | gces vs gces-noComp | 6/0/15 | 3/0/18 | 3/3/15 |
| zcat18 | gces vs gces-noGeo | 7/0/14 | 8/0/13 | 7/1/13 |
| zcat19 | gces vs nsgaii | 20/0/1 | 19/0/2 | 19/1/1 |
| zcat19 | gces-noComp vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | gces-noGeo vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat19 | gces vs gces-noComp | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat19 | gces vs gces-noGeo | 12/0/9 | 10/0/11 | 10/2/9 |
| zcat20 | gces vs nsgaii | 20/0/1 | 18/0/3 | 18/2/1 |
| zcat20 | gces-noComp vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat20 | gces-noGeo vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat20 | gces vs gces-noComp | 6/0/15 | 4/0/17 | 4/2/15 |
| zcat20 | gces vs gces-noGeo | 7/0/14 | 8/0/13 | 7/1/13 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | gces vs nsgaii | 0.001729 | 19/0/2 | 0.000010 | 0.000439 | yes |
| zcat1 | gces-noComp vs nsgaii | 0.004561 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat1 | gces-noGeo vs nsgaii | 0.001833 | 19/0/2 | 0.000007 | 0.000327 | yes |
| zcat1 | gces vs gces-noComp | -0.002832 | 1/0/20 | 0.000002 | 0.000120 | yes |
| zcat1 | gces vs gces-noGeo | -0.000104 | 11/0/10 | 0.811678 | 1.000000 | no |
| zcat10 | gces vs nsgaii | 0.009407 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat10 | gces-noComp vs nsgaii | 0.004273 | 14/0/7 | 0.003753 | 0.112581 | no |
| zcat10 | gces-noGeo vs nsgaii | 0.010065 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat10 | gces vs gces-noComp | 0.005134 | 20/0/1 | 0.000003 | 0.000146 | yes |
| zcat10 | gces vs gces-noGeo | -0.000658 | 11/0/10 | 0.945745 | 1.000000 | no |
| zcat11 | gces vs nsgaii | 0.002003 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat11 | gces-noComp vs nsgaii | 0.003008 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat11 | gces-noGeo vs nsgaii | 0.001909 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat11 | gces vs gces-noComp | -0.001006 | 3/0/18 | 0.000084 | 0.003525 | yes |
| zcat11 | gces vs gces-noGeo | 0.000094 | 12/0/9 | 0.452368 | 1.000000 | no |
| zcat12 | gces vs nsgaii | 0.002089 | 20/0/1 | 0.000426 | 0.016199 | yes |
| zcat12 | gces-noComp vs nsgaii | 0.002842 | 20/0/1 | 0.000426 | 0.016199 | yes |
| zcat12 | gces-noGeo vs nsgaii | 0.002060 | 20/0/1 | 0.000426 | 0.016199 | yes |
| zcat12 | gces vs gces-noComp | -0.000753 | 3/0/18 | 0.000852 | 0.027252 | yes |
| zcat12 | gces vs gces-noGeo | 0.000029 | 11/0/10 | 0.431911 | 1.000000 | no |
| zcat13 | gces vs nsgaii | 0.001584 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat13 | gces-noComp vs nsgaii | 0.002291 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat13 | gces-noGeo vs nsgaii | 0.001682 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat13 | gces vs gces-noComp | -0.000707 | 1/0/20 | 0.000002 | 0.000120 | yes |
| zcat13 | gces vs gces-noGeo | -0.000098 | 8/0/13 | 0.355416 | 1.000000 | no |
| zcat14 | gces vs nsgaii | 0.004066 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat14 | gces-noComp vs nsgaii | 0.005678 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat14 | gces-noGeo vs nsgaii | 0.003834 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat14 | gces vs gces-noComp | -0.001612 | 2/0/19 | 0.000013 | 0.000601 | yes |
| zcat14 | gces vs gces-noGeo | 0.000232 | 14/0/7 | 0.228986 | 1.000000 | no |
| zcat15 | gces vs nsgaii | 0.002089 | 20/0/1 | 0.000426 | 0.016199 | yes |
| zcat15 | gces-noComp vs nsgaii | 0.002842 | 20/0/1 | 0.000426 | 0.016199 | yes |
| zcat15 | gces-noGeo vs nsgaii | 0.002060 | 20/0/1 | 0.000426 | 0.016199 | yes |
| zcat15 | gces vs gces-noComp | -0.000753 | 3/0/18 | 0.000852 | 0.027252 | yes |
| zcat15 | gces vs gces-noGeo | 0.000029 | 11/0/10 | 0.431911 | 1.000000 | no |
| zcat16 | gces vs nsgaii | 0.003217 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat16 | gces-noComp vs nsgaii | 0.002835 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat16 | gces-noGeo vs nsgaii | 0.003179 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat16 | gces vs gces-noComp | 0.000382 | 13/0/8 | 0.095799 | 1.000000 | no |
| zcat16 | gces vs gces-noGeo | 0.000038 | 11/0/10 | 0.373725 | 1.000000 | no |
| zcat17 | gces vs nsgaii | 0.001178 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat17 | gces-noComp vs nsgaii | 0.001342 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat17 | gces-noGeo vs nsgaii | 0.001182 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat17 | gces vs gces-noComp | -0.000163 | 4/0/17 | 0.015780 | 0.394511 | no |
| zcat17 | gces vs gces-noGeo | -0.000003 | 10/0/11 | 0.811678 | 1.000000 | no |
| zcat18 | gces vs nsgaii | 0.001957 | 20/0/1 | 0.000197 | 0.008094 | yes |
| zcat18 | gces-noComp vs nsgaii | 0.002565 | 19/0/2 | 0.004285 | 0.124261 | no |
| zcat18 | gces-noGeo vs nsgaii | 0.002354 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat18 | gces vs gces-noComp | -0.000608 | 6/0/15 | 0.031919 | 0.766068 | no |
| zcat18 | gces vs gces-noGeo | -0.000397 | 7/0/14 | 0.050192 | 1.000000 | no |
| zcat19 | gces vs nsgaii | 0.001476 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat19 | gces-noComp vs nsgaii | 0.004026 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat19 | gces-noGeo vs nsgaii | 0.001432 | 19/0/2 | 0.000007 | 0.000327 | yes |
| zcat19 | gces vs gces-noComp | -0.002550 | 1/0/20 | 0.000002 | 0.000120 | yes |
| zcat19 | gces vs gces-noGeo | 0.000043 | 12/0/9 | 0.785365 | 1.000000 | no |
| zcat2 | gces vs nsgaii | 0.001052 | 20/0/1 | 0.000003 | 0.000146 | yes |
| zcat2 | gces-noComp vs nsgaii | 0.001794 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat2 | gces-noGeo vs nsgaii | 0.000774 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat2 | gces vs gces-noComp | -0.000742 | 1/0/20 | 0.000002 | 0.000120 | yes |
| zcat2 | gces vs gces-noGeo | 0.000278 | 12/0/9 | 0.146964 | 1.000000 | no |
| zcat20 | gces vs nsgaii | 0.001958 | 20/0/1 | 0.000197 | 0.008094 | yes |
| zcat20 | gces-noComp vs nsgaii | 0.002565 | 19/0/2 | 0.004285 | 0.124261 | no |
| zcat20 | gces-noGeo vs nsgaii | 0.002355 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat20 | gces vs gces-noComp | -0.000608 | 6/0/15 | 0.031919 | 0.766068 | no |
| zcat20 | gces vs gces-noGeo | -0.000398 | 7/0/14 | 0.050192 | 1.000000 | no |
| zcat3 | gces vs nsgaii | 0.002212 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat3 | gces-noComp vs nsgaii | 0.004326 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat3 | gces-noGeo vs nsgaii | 0.002300 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat3 | gces vs gces-noComp | -0.002115 | 0/0/21 | 0.000001 | 0.000095 | yes |
| zcat3 | gces vs gces-noGeo | -0.000088 | 10/0/11 | 0.972858 | 1.000000 | no |
| zcat4 | gces vs nsgaii | 0.002211 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat4 | gces-noComp vs nsgaii | 0.004326 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat4 | gces-noGeo vs nsgaii | 0.002301 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat4 | gces vs gces-noComp | -0.002115 | 0/0/21 | 0.000001 | 0.000095 | yes |
| zcat4 | gces vs gces-noGeo | -0.000089 | 10/0/11 | 0.945745 | 1.000000 | no |
| zcat5 | gces vs nsgaii | 0.001178 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat5 | gces-noComp vs nsgaii | 0.001341 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat5 | gces-noGeo vs nsgaii | 0.001181 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat5 | gces vs gces-noComp | -0.000164 | 4/0/17 | 0.014166 | 0.368313 | no |
| zcat5 | gces vs gces-noGeo | -0.000004 | 10/0/11 | 0.811678 | 1.000000 | no |
| zcat6 | gces vs nsgaii | 0.000706 | 19/0/2 | 0.000024 | 0.001049 | yes |
| zcat6 | gces-noComp vs nsgaii | 0.001239 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat6 | gces-noGeo vs nsgaii | 0.000767 | 18/0/3 | 0.000052 | 0.002255 | yes |
| zcat6 | gces vs gces-noComp | -0.000533 | 1/0/20 | 0.000007 | 0.000327 | yes |
| zcat6 | gces vs gces-noGeo | -0.000061 | 8/0/13 | 0.494802 | 1.000000 | no |
| zcat7 | gces vs nsgaii | 0.001957 | 20/0/1 | 0.000197 | 0.008094 | yes |
| zcat7 | gces-noComp vs nsgaii | 0.002565 | 19/0/2 | 0.004285 | 0.124261 | no |
| zcat7 | gces-noGeo vs nsgaii | 0.002354 | 20/0/1 | 0.000002 | 0.000120 | yes |
| zcat7 | gces vs gces-noComp | -0.000608 | 6/0/15 | 0.031919 | 0.766068 | no |
| zcat7 | gces vs gces-noGeo | -0.000397 | 7/0/14 | 0.050192 | 1.000000 | no |
| zcat8 | gces vs nsgaii | 0.008135 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces-noComp vs nsgaii | 0.004251 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces-noGeo vs nsgaii | 0.007666 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces vs gces-noComp | 0.003883 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces vs gces-noGeo | 0.000469 | 9/0/12 | 0.918694 | 1.000000 | no |
| zcat9 | gces vs nsgaii | 0.008135 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces-noComp vs nsgaii | 0.004251 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces-noGeo vs nsgaii | 0.007666 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces vs gces-noComp | 0.003883 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces vs gces-noGeo | 0.000469 | 9/0/12 | 0.918694 | 1.000000 | no |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | gces vs nsgaii | -0.000618 | 16/0/5 | 0.003279 | 0.121313 | no |
| zcat1 | gces-noComp vs nsgaii | -0.001860 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat1 | gces-noGeo vs nsgaii | -0.000487 | 17/0/4 | 0.005542 | 0.177338 | no |
| zcat1 | gces vs gces-noComp | 0.001242 | 0/0/21 | 0.000001 | 0.000095 | yes |
| zcat1 | gces vs gces-noGeo | -0.000132 | 12/0/9 | 0.539189 | 1.000000 | no |
| zcat10 | gces vs nsgaii | -0.000105 | 12/0/9 | 0.785365 | 1.000000 | no |
| zcat10 | gces-noComp vs nsgaii | -0.000111 | 15/0/6 | 0.006281 | 0.194708 | no |
| zcat10 | gces-noGeo vs nsgaii | -0.000056 | 10/0/11 | 0.972858 | 1.000000 | no |
| zcat10 | gces vs gces-noComp | 0.000007 | 9/0/12 | 0.287734 | 1.000000 | no |
| zcat10 | gces vs gces-noGeo | -0.000049 | 8/0/13 | 0.373725 | 1.000000 | no |
| zcat11 | gces vs nsgaii | -0.000962 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat11 | gces-noComp vs nsgaii | -0.001349 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat11 | gces-noGeo vs nsgaii | -0.000936 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat11 | gces vs gces-noComp | 0.000387 | 3/0/18 | 0.000084 | 0.004951 | yes |
| zcat11 | gces vs gces-noGeo | -0.000027 | 12/0/9 | 0.452368 | 1.000000 | no |
| zcat12 | gces vs nsgaii | -0.001065 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat12 | gces-noComp vs nsgaii | -0.001377 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat12 | gces-noGeo vs nsgaii | -0.001068 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat12 | gces vs gces-noComp | 0.000313 | 4/0/17 | 0.001600 | 0.065611 | no |
| zcat12 | gces vs gces-noGeo | 0.000003 | 10/0/11 | 0.494802 | 1.000000 | no |
| zcat13 | gces vs nsgaii | -0.000832 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat13 | gces-noComp vs nsgaii | -0.001033 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat13 | gces-noGeo vs nsgaii | -0.000948 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat13 | gces vs gces-noComp | 0.000201 | 3/0/18 | 0.000354 | 0.019106 | yes |
| zcat13 | gces vs gces-noGeo | 0.000116 | 10/0/11 | 0.373725 | 1.000000 | no |
| zcat14 | gces vs nsgaii | -0.001132 | 18/0/3 | 0.000024 | 0.001502 | yes |
| zcat14 | gces-noComp vs nsgaii | -0.001541 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat14 | gces-noGeo vs nsgaii | -0.000972 | 20/0/1 | 0.000002 | 0.000134 | yes |
| zcat14 | gces vs gces-noComp | 0.000410 | 3/0/18 | 0.000018 | 0.001160 | yes |
| zcat14 | gces vs gces-noGeo | -0.000159 | 14/0/7 | 0.392584 | 1.000000 | no |
| zcat15 | gces vs nsgaii | -0.001120 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat15 | gces-noComp vs nsgaii | -0.001423 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat15 | gces-noGeo vs nsgaii | -0.001125 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat15 | gces vs gces-noComp | 0.000302 | 3/0/18 | 0.001002 | 0.042097 | yes |
| zcat15 | gces vs gces-noGeo | 0.000005 | 10/0/11 | 0.562075 | 1.000000 | no |
| zcat16 | gces vs nsgaii | -0.001220 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat16 | gces-noComp vs nsgaii | -0.001114 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat16 | gces-noGeo vs nsgaii | -0.001158 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat16 | gces vs gces-noComp | -0.000107 | 12/0/9 | 0.320457 | 1.000000 | no |
| zcat16 | gces vs gces-noGeo | -0.000062 | 12/0/9 | 0.392584 | 1.000000 | no |
| zcat17 | gces vs nsgaii | -0.000627 | 17/0/4 | 0.000426 | 0.022593 | yes |
| zcat17 | gces-noComp vs nsgaii | -0.000994 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat17 | gces-noGeo vs nsgaii | -0.000503 | 18/0/3 | 0.000041 | 0.002460 | yes |
| zcat17 | gces vs gces-noComp | 0.000368 | 1/0/20 | 0.000002 | 0.000134 | yes |
| zcat17 | gces vs gces-noGeo | -0.000124 | 13/0/8 | 0.373725 | 1.000000 | no |
| zcat18 | gces vs nsgaii | -0.000641 | 18/0/3 | 0.002151 | 0.086060 | no |
| zcat18 | gces-noComp vs nsgaii | -0.000964 | 19/0/2 | 0.008010 | 0.240297 | no |
| zcat18 | gces-noGeo vs nsgaii | -0.000792 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat18 | gces vs gces-noComp | 0.000324 | 3/0/18 | 0.003279 | 0.121313 | no |
| zcat18 | gces vs gces-noGeo | 0.000151 | 8/0/13 | 0.119342 | 1.000000 | no |
| zcat19 | gces vs nsgaii | -0.000464 | 19/0/2 | 0.000084 | 0.004951 | yes |
| zcat19 | gces-noComp vs nsgaii | -0.001583 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat19 | gces-noGeo vs nsgaii | -0.000443 | 19/0/2 | 0.000024 | 0.001502 | yes |
| zcat19 | gces vs gces-noComp | 0.001119 | 1/0/20 | 0.000002 | 0.000134 | yes |
| zcat19 | gces vs gces-noGeo | -0.000022 | 10/0/11 | 0.918694 | 1.000000 | no |
| zcat2 | gces vs nsgaii | -0.000730 | 20/0/1 | 0.000007 | 0.000434 | yes |
| zcat2 | gces-noComp vs nsgaii | -0.001242 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat2 | gces-noGeo vs nsgaii | -0.000669 | 19/0/2 | 0.000024 | 0.001502 | yes |
| zcat2 | gces vs gces-noComp | 0.000511 | 0/0/21 | 0.000001 | 0.000095 | yes |
| zcat2 | gces vs gces-noGeo | -0.000061 | 15/0/6 | 0.190687 | 1.000000 | no |
| zcat20 | gces vs nsgaii | -0.000664 | 18/0/3 | 0.002482 | 0.096814 | no |
| zcat20 | gces-noComp vs nsgaii | -0.000954 | 19/0/2 | 0.008010 | 0.240297 | no |
| zcat20 | gces-noGeo vs nsgaii | -0.000806 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat20 | gces vs gces-noComp | 0.000290 | 4/0/17 | 0.003753 | 0.123839 | no |
| zcat20 | gces vs gces-noGeo | 0.000142 | 8/0/13 | 0.119342 | 1.000000 | no |
| zcat3 | gces vs nsgaii | -0.000864 | 19/0/2 | 0.000197 | 0.011055 | yes |
| zcat3 | gces-noComp vs nsgaii | -0.001834 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat3 | gces-noGeo vs nsgaii | -0.000757 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat3 | gces vs gces-noComp | 0.000970 | 0/0/21 | 0.000001 | 0.000095 | yes |
| zcat3 | gces vs gces-noGeo | -0.000107 | 11/0/10 | 0.945745 | 1.000000 | no |
| zcat4 | gces vs nsgaii | -0.000788 | 19/0/2 | 0.000293 | 0.016103 | yes |
| zcat4 | gces-noComp vs nsgaii | -0.001821 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat4 | gces-noGeo vs nsgaii | -0.000748 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat4 | gces vs gces-noComp | 0.001032 | 0/0/21 | 0.000001 | 0.000095 | yes |
| zcat4 | gces vs gces-noGeo | -0.000041 | 12/0/9 | 0.811678 | 1.000000 | no |
| zcat5 | gces vs nsgaii | -0.000603 | 17/0/4 | 0.000852 | 0.036620 | yes |
| zcat5 | gces-noComp vs nsgaii | -0.000981 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat5 | gces-noGeo vs nsgaii | -0.000501 | 18/0/3 | 0.000084 | 0.004951 | yes |
| zcat5 | gces vs gces-noComp | 0.000378 | 1/0/20 | 0.000002 | 0.000134 | yes |
| zcat5 | gces vs gces-noGeo | -0.000102 | 13/0/8 | 0.373725 | 1.000000 | no |
| zcat6 | gces vs nsgaii | 0.000006 | 8/0/13 | 0.811678 | 1.000000 | no |
| zcat6 | gces-noComp vs nsgaii | -0.000531 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat6 | gces-noGeo vs nsgaii | -0.000023 | 9/0/12 | 0.785365 | 1.000000 | no |
| zcat6 | gces vs gces-noComp | 0.000537 | 1/0/20 | 0.000002 | 0.000134 | yes |
| zcat6 | gces vs gces-noGeo | 0.000029 | 9/0/12 | 0.838194 | 1.000000 | no |
| zcat7 | gces vs nsgaii | -0.000643 | 18/0/3 | 0.002482 | 0.096814 | no |
| zcat7 | gces-noComp vs nsgaii | -0.000951 | 19/0/2 | 0.008010 | 0.240297 | no |
| zcat7 | gces-noGeo vs nsgaii | -0.000811 | 20/0/1 | 0.000426 | 0.022593 | yes |
| zcat7 | gces vs gces-noComp | 0.000308 | 4/0/17 | 0.003279 | 0.121313 | no |
| zcat7 | gces vs gces-noGeo | 0.000168 | 8/0/13 | 0.128078 | 1.000000 | no |
| zcat8 | gces vs nsgaii | -0.001216 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces-noComp vs nsgaii | -0.000889 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces-noGeo vs nsgaii | -0.001523 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat8 | gces vs gces-noComp | -0.000327 | 15/0/6 | 0.003279 | 0.121313 | no |
| zcat8 | gces vs gces-noGeo | 0.000307 | 5/0/16 | 0.119342 | 1.000000 | no |
| zcat9 | gces vs nsgaii | -0.001227 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces-noComp vs nsgaii | -0.000874 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces-noGeo vs nsgaii | -0.001470 | 21/0/0 | 0.000001 | 0.000095 | yes |
| zcat9 | gces vs gces-noComp | -0.000353 | 14/0/7 | 0.010125 | 0.273379 | no |
| zcat9 | gces vs gces-noGeo | 0.000244 | 5/0/16 | 0.137283 | 1.000000 | no |
