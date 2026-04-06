# NSGA-II farthest 2-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across nsga2_farthest and the GCES-family variants.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, nsga2-farthest, gces-noComp, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-farthest vs nsgaii, nsga2-farthest vs gces, nsga2-farthest vs gces-noComp, nsga2-farthest vs gces-noGeo

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 0.981731 | 0.986382 | 0.986292 | 0.983564 | 0.983460 |
| zcat2 | 0.991373 | 0.992855 | 0.993167 | 0.992147 | 0.992425 |
| zcat3 | 0.981568 | 0.986511 | 0.985894 | 0.983868 | 0.983779 |
| zcat4 | 0.981568 | 0.986512 | 0.985894 | 0.983869 | 0.983779 |
| zcat5 | 0.994124 | 0.995304 | 0.995465 | 0.995305 | 0.995301 |
| zcat6 | 0.994481 | 0.995075 | 0.995720 | 0.995249 | 0.995187 |
| zcat7 | 0.989018 | 0.990921 | 0.991583 | 0.991373 | 0.990975 |
| zcat8 | 0.967947 | 0.974321 | 0.972198 | 0.975612 | 0.976081 |
| zcat9 | 0.967941 | 0.974316 | 0.972193 | 0.975607 | 0.976076 |
| zcat10 | 0.946007 | 0.948984 | 0.950280 | 0.956072 | 0.955414 |
| zcat11 | 0.988725 | 0.991531 | 0.991733 | 0.990634 | 0.990728 |
| zcat12 | 0.988428 | 0.991547 | 0.991270 | 0.990488 | 0.990517 |
| zcat13 | 1.015344 | 1.017381 | 1.017635 | 1.017026 | 1.016928 |
| zcat14 | 0.975055 | 0.980920 | 0.980733 | 0.978889 | 0.979122 |
| zcat15 | 0.988429 | 0.991547 | 0.991271 | 0.990488 | 0.990518 |
| zcat16 | 0.986793 | 0.988881 | 0.989628 | 0.989972 | 0.990009 |
| zcat17 | 0.994120 | 0.995300 | 0.995461 | 0.995301 | 0.995298 |
| zcat18 | 0.989018 | 0.990921 | 0.991582 | 0.991372 | 0.990975 |
| zcat19 | 0.984635 | 0.988134 | 0.988662 | 0.986068 | 0.986111 |
| zcat20 | 0.989016 | 0.990919 | 0.991581 | 0.991371 | 0.990974 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 0.007839 | 0.005936 | 0.005979 | 0.007352 | 0.007221 |
| zcat2 | 0.006222 | 0.005140 | 0.004981 | 0.005553 | 0.005492 |
| zcat3 | 0.007943 | 0.005845 | 0.006109 | 0.007186 | 0.007079 |
| zcat4 | 0.007915 | 0.005859 | 0.006094 | 0.007167 | 0.007126 |
| zcat5 | 0.004491 | 0.003606 | 0.003509 | 0.003990 | 0.003888 |
| zcat6 | 0.002882 | 0.002565 | 0.002351 | 0.002859 | 0.002888 |
| zcat7 | 0.004714 | 0.004049 | 0.003763 | 0.003903 | 0.004071 |
| zcat8 | 0.006383 | 0.005100 | 0.005494 | 0.004859 | 0.005167 |
| zcat9 | 0.006371 | 0.005079 | 0.005497 | 0.004900 | 0.005144 |
| zcat10 | 0.003497 | 0.003351 | 0.003385 | 0.003441 | 0.003392 |
| zcat11 | 0.005266 | 0.004013 | 0.003917 | 0.004331 | 0.004304 |
| zcat12 | 0.005507 | 0.004031 | 0.004130 | 0.004439 | 0.004442 |
| zcat13 | 0.004038 | 0.003146 | 0.003005 | 0.003090 | 0.003206 |
| zcat14 | 0.007444 | 0.005750 | 0.005902 | 0.006471 | 0.006312 |
| zcat15 | 0.005556 | 0.004046 | 0.004134 | 0.004431 | 0.004436 |
| zcat16 | 0.004940 | 0.004199 | 0.003826 | 0.003782 | 0.003720 |
| zcat17 | 0.004507 | 0.003599 | 0.003512 | 0.004004 | 0.003880 |
| zcat18 | 0.004708 | 0.004051 | 0.003744 | 0.003917 | 0.004068 |
| zcat19 | 0.006491 | 0.005111 | 0.004908 | 0.006049 | 0.006027 |
| zcat20 | 0.004717 | 0.004054 | 0.003763 | 0.003911 | 0.004053 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 1.326 | 12.397 | 5.529 | 4.555 | 5.669 |
| zcat2 | 2.056 | 19.822 | 8.362 | 4.835 | 6.033 |
| zcat3 | 2.081 | 21.100 | 8.422 | 4.554 | 5.672 |
| zcat4 | 1.936 | 19.771 | 8.289 | 4.311 | 5.461 |
| zcat5 | 1.994 | 17.996 | 7.861 | 4.586 | 5.675 |
| zcat6 | 2.008 | 16.893 | 7.482 | 4.306 | 5.213 |
| zcat7 | 2.046 | 19.218 | 8.927 | 4.959 | 5.914 |
| zcat8 | 2.154 | 20.086 | 8.377 | 4.920 | 6.066 |
| zcat9 | 2.141 | 20.250 | 8.396 | 4.903 | 6.079 |
| zcat10 | 2.102 | 17.383 | 7.743 | 4.737 | 5.841 |
| zcat11 | 2.116 | 19.913 | 8.832 | 4.125 | 4.902 |
| zcat12 | 2.042 | 19.479 | 8.948 | 4.634 | 5.390 |
| zcat13 | 2.089 | 19.275 | 9.291 | 4.867 | 5.682 |
| zcat14 | 2.046 | 20.349 | 8.472 | 4.719 | 5.834 |
| zcat15 | 2.024 | 20.005 | 9.147 | 4.602 | 5.470 |
| zcat16 | 2.097 | 19.404 | 8.798 | 4.223 | 4.938 |
| zcat17 | 2.172 | 19.085 | 8.098 | 4.753 | 5.835 |
| zcat18 | 2.114 | 19.171 | 8.684 | 4.767 | 5.671 |
| zcat19 | 2.098 | 18.688 | 7.906 | 4.091 | 5.095 |
| zcat20 | 1.931 | 17.990 | 8.644 | 4.717 | 5.630 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-farthest vs gces | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat1 | nsga2-farthest vs gces-noComp | 12/0/9 | 13/0/8 | 12/1/8 |
| zcat1 | nsga2-farthest vs gces-noGeo | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat2 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-farthest vs gces | 16/0/5 | 18/0/3 | 16/2/3 |
| zcat2 | nsga2-farthest vs gces-noComp | 8/0/13 | 10/0/11 | 7/4/10 |
| zcat2 | nsga2-farthest vs gces-noGeo | 19/0/2 | 20/0/1 | 19/1/1 |
| zcat3 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-farthest vs gces-noComp | 11/0/10 | 11/0/10 | 11/0/10 |
| zcat3 | nsga2-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs gces-noComp | 11/0/10 | 11/0/10 | 11/0/10 |
| zcat4 | nsga2-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-farthest vs gces | 12/0/9 | 15/0/6 | 12/3/6 |
| zcat5 | nsga2-farthest vs gces-noComp | 6/0/15 | 7/0/14 | 5/3/13 |
| zcat5 | nsga2-farthest vs gces-noGeo | 10/0/11 | 19/0/2 | 10/9/2 |
| zcat6 | nsga2-farthest vs nsgaii | 19/0/2 | 21/0/0 | 19/2/0 |
| zcat6 | nsga2-farthest vs gces | 12/0/9 | 19/0/2 | 12/7/2 |
| zcat6 | nsga2-farthest vs gces-noComp | 1/0/20 | 1/0/20 | 0/2/19 |
| zcat6 | nsga2-farthest vs gces-noGeo | 9/0/12 | 20/0/1 | 9/11/1 |
| zcat7 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat7 | nsga2-farthest vs gces | 9/0/12 | 11/0/10 | 9/2/10 |
| zcat7 | nsga2-farthest vs gces-noComp | 6/0/15 | 6/0/15 | 5/2/14 |
| zcat7 | nsga2-farthest vs gces-noGeo | 7/0/14 | 6/0/15 | 5/3/13 |
| zcat8 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-farthest vs gces | 3/0/18 | 12/0/9 | 3/9/9 |
| zcat8 | nsga2-farthest vs gces-noComp | 17/0/4 | 16/0/5 | 16/1/4 |
| zcat8 | nsga2-farthest vs gces-noGeo | 4/0/17 | 5/0/16 | 1/7/13 |
| zcat9 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-farthest vs gces | 3/0/18 | 12/0/9 | 3/9/9 |
| zcat9 | nsga2-farthest vs gces-noComp | 17/0/4 | 16/0/5 | 16/1/4 |
| zcat9 | nsga2-farthest vs gces-noGeo | 4/0/17 | 6/0/15 | 2/6/13 |
| zcat10 | nsga2-farthest vs nsgaii | 14/0/7 | 13/0/8 | 12/3/6 |
| zcat10 | nsga2-farthest vs gces | 2/0/19 | 12/0/9 | 2/10/9 |
| zcat10 | nsga2-farthest vs gces-noComp | 11/0/10 | 11/0/10 | 10/2/9 |
| zcat10 | nsga2-farthest vs gces-noGeo | 1/0/20 | 13/0/8 | 1/12/8 |
| zcat11 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat11 | nsga2-farthest vs gces | 18/0/3 | 17/0/4 | 17/1/3 |
| zcat11 | nsga2-farthest vs gces-noComp | 6/0/15 | 7/0/14 | 6/1/14 |
| zcat11 | nsga2-farthest vs gces-noGeo | 19/0/2 | 18/0/3 | 18/1/2 |
| zcat12 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | nsga2-farthest vs gces | 19/0/2 | 19/0/2 | 18/2/1 |
| zcat12 | nsga2-farthest vs gces-noComp | 15/0/6 | 11/0/10 | 11/4/6 |
| zcat12 | nsga2-farthest vs gces-noGeo | 19/0/2 | 18/0/3 | 18/1/2 |
| zcat13 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-farthest vs gces | 16/0/5 | 11/0/10 | 11/5/5 |
| zcat13 | nsga2-farthest vs gces-noComp | 5/0/16 | 5/0/16 | 4/2/15 |
| zcat13 | nsga2-farthest vs gces-noGeo | 17/0/4 | 8/0/13 | 8/9/4 |
| zcat14 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-farthest vs gces | 18/0/3 | 18/0/3 | 17/2/2 |
| zcat14 | nsga2-farthest vs gces-noComp | 11/0/10 | 12/0/9 | 11/1/9 |
| zcat14 | nsga2-farthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | nsga2-farthest vs gces | 19/0/2 | 19/0/2 | 18/2/1 |
| zcat15 | nsga2-farthest vs gces-noComp | 15/0/6 | 11/0/10 | 11/4/6 |
| zcat15 | nsga2-farthest vs gces-noGeo | 19/0/2 | 18/0/3 | 18/1/2 |
| zcat16 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat16 | nsga2-farthest vs gces | 4/0/17 | 5/0/16 | 3/3/15 |
| zcat16 | nsga2-farthest vs gces-noComp | 7/0/14 | 6/0/15 | 6/1/14 |
| zcat16 | nsga2-farthest vs gces-noGeo | 5/0/16 | 3/0/18 | 3/2/16 |
| zcat17 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-farthest vs gces | 12/0/9 | 15/0/6 | 12/3/6 |
| zcat17 | nsga2-farthest vs gces-noComp | 6/0/15 | 7/0/14 | 5/3/13 |
| zcat17 | nsga2-farthest vs gces-noGeo | 10/0/11 | 19/0/2 | 10/9/2 |
| zcat18 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat18 | nsga2-farthest vs gces | 9/0/12 | 11/0/10 | 9/2/10 |
| zcat18 | nsga2-farthest vs gces-noComp | 6/0/15 | 6/0/15 | 5/2/14 |
| zcat18 | nsga2-farthest vs gces-noGeo | 7/0/14 | 6/0/15 | 5/3/13 |
| zcat19 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-farthest vs gces | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat19 | nsga2-farthest vs gces-noComp | 7/0/14 | 8/0/13 | 7/1/13 |
| zcat19 | nsga2-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat20 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat20 | nsga2-farthest vs gces | 9/0/12 | 11/0/10 | 9/2/10 |
| zcat20 | nsga2-farthest vs gces-noComp | 6/0/15 | 5/0/16 | 4/3/14 |
| zcat20 | nsga2-farthest vs gces-noGeo | 7/0/14 | 6/0/15 | 5/3/13 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0.004651 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-farthest vs gces | 0.002922 | 20/0/1 | 0.000002 | 0.000113 | yes |
| zcat1 | nsga2-farthest vs gces-noComp | 0.000090 | 12/0/9 | 0.257248 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces-noGeo | 0.002818 | 20/0/1 | 0.000002 | 0.000113 | yes |
| zcat10 | nsga2-farthest vs nsgaii | 0.002977 | 14/0/7 | 0.075980 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces | -0.006430 | 2/0/19 | 0.000005 | 0.000262 | yes |
| zcat10 | nsga2-farthest vs gces-noComp | -0.001295 | 11/0/10 | 0.539189 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces-noGeo | -0.007088 | 1/0/20 | 0.000010 | 0.000515 | yes |
| zcat11 | nsga2-farthest vs nsgaii | 0.002806 | 20/0/1 | 0.000426 | 0.019183 | yes |
| zcat11 | nsga2-farthest vs gces | 0.000804 | 18/0/3 | 0.000721 | 0.028839 | yes |
| zcat11 | nsga2-farthest vs gces-noComp | -0.000202 | 6/0/15 | 0.035056 | 1.000000 | no |
| zcat11 | nsga2-farthest vs gces-noGeo | 0.000898 | 19/0/2 | 0.001002 | 0.039090 | yes |
| zcat12 | nsga2-farthest vs nsgaii | 0.003119 | 20/0/1 | 0.000426 | 0.019183 | yes |
| zcat12 | nsga2-farthest vs gces | 0.001030 | 19/0/2 | 0.001002 | 0.039090 | yes |
| zcat12 | nsga2-farthest vs gces-noComp | 0.000277 | 15/0/6 | 0.202917 | 1.000000 | no |
| zcat12 | nsga2-farthest vs gces-noGeo | 0.001059 | 19/0/2 | 0.000426 | 0.019183 | yes |
| zcat13 | nsga2-farthest vs nsgaii | 0.002037 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-farthest vs gces | 0.000453 | 16/0/5 | 0.012691 | 0.380745 | no |
| zcat13 | nsga2-farthest vs gces-noComp | -0.000254 | 5/0/16 | 0.009016 | 0.279497 | no |
| zcat13 | nsga2-farthest vs gces-noGeo | 0.000355 | 17/0/4 | 0.001859 | 0.065055 | no |
| zcat14 | nsga2-farthest vs nsgaii | 0.005864 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-farthest vs gces | 0.001798 | 18/0/3 | 0.000067 | 0.003471 | yes |
| zcat14 | nsga2-farthest vs gces-noComp | 0.000186 | 11/0/10 | 0.585402 | 1.000000 | no |
| zcat14 | nsga2-farthest vs gces-noGeo | 0.002030 | 20/0/1 | 0.000002 | 0.000113 | yes |
| zcat15 | nsga2-farthest vs nsgaii | 0.003119 | 20/0/1 | 0.000426 | 0.019183 | yes |
| zcat15 | nsga2-farthest vs gces | 0.001030 | 19/0/2 | 0.001002 | 0.039090 | yes |
| zcat15 | nsga2-farthest vs gces-noComp | 0.000277 | 15/0/6 | 0.202917 | 1.000000 | no |
| zcat15 | nsga2-farthest vs gces-noGeo | 0.001059 | 19/0/2 | 0.000426 | 0.019183 | yes |
| zcat16 | nsga2-farthest vs nsgaii | 0.002088 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-farthest vs gces | -0.001129 | 4/0/17 | 0.000293 | 0.013468 | yes |
| zcat16 | nsga2-farthest vs gces-noComp | -0.000747 | 7/0/14 | 0.075980 | 1.000000 | no |
| zcat16 | nsga2-farthest vs gces-noGeo | -0.001091 | 5/0/16 | 0.001374 | 0.049473 | yes |
| zcat17 | nsga2-farthest vs nsgaii | 0.001181 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-farthest vs gces | 0.000002 | 12/0/9 | 0.838194 | 1.000000 | no |
| zcat17 | nsga2-farthest vs gces-noComp | -0.000161 | 6/0/15 | 0.045993 | 1.000000 | no |
| zcat17 | nsga2-farthest vs gces-noGeo | -0.000001 | 10/0/11 | 0.785365 | 1.000000 | no |
| zcat18 | nsga2-farthest vs nsgaii | 0.001903 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-farthest vs gces | -0.000054 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces-noComp | -0.000662 | 6/0/15 | 0.059507 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces-noGeo | -0.000451 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat19 | nsga2-farthest vs nsgaii | 0.003499 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-farthest vs gces | 0.002023 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-farthest vs gces-noComp | -0.000527 | 7/0/14 | 0.082195 | 1.000000 | no |
| zcat19 | nsga2-farthest vs gces-noGeo | 0.002066 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-farthest vs nsgaii | 0.001482 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-farthest vs gces | 0.000430 | 16/0/5 | 0.002482 | 0.084402 | no |
| zcat2 | nsga2-farthest vs gces-noComp | -0.000312 | 8/0/13 | 0.257248 | 1.000000 | no |
| zcat2 | nsga2-farthest vs gces-noGeo | 0.000708 | 19/0/2 | 0.000105 | 0.005350 | yes |
| zcat20 | nsga2-farthest vs nsgaii | 0.001904 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-farthest vs gces | -0.000054 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces-noComp | -0.000662 | 6/0/15 | 0.054693 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces-noGeo | -0.000452 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat3 | nsga2-farthest vs nsgaii | 0.004944 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-farthest vs gces | 0.002732 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-farthest vs gces-noComp | 0.000617 | 11/0/10 | 0.228986 | 1.000000 | no |
| zcat3 | nsga2-farthest vs gces-noGeo | 0.002644 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-farthest vs nsgaii | 0.004944 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-farthest vs gces | 0.002733 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-farthest vs gces-noComp | 0.000618 | 11/0/10 | 0.228986 | 1.000000 | no |
| zcat4 | nsga2-farthest vs gces-noGeo | 0.002644 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-farthest vs nsgaii | 0.001180 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-farthest vs gces | 0.000002 | 12/0/9 | 0.811678 | 1.000000 | no |
| zcat5 | nsga2-farthest vs gces-noComp | -0.000161 | 6/0/15 | 0.045993 | 1.000000 | no |
| zcat5 | nsga2-farthest vs gces-noGeo | -0.000001 | 10/0/11 | 0.759288 | 1.000000 | no |
| zcat6 | nsga2-farthest vs nsgaii | 0.000593 | 19/0/2 | 0.000013 | 0.000708 | yes |
| zcat6 | nsga2-farthest vs gces | -0.000113 | 12/0/9 | 0.682714 | 1.000000 | no |
| zcat6 | nsga2-farthest vs gces-noComp | -0.000646 | 1/0/20 | 0.000002 | 0.000113 | yes |
| zcat6 | nsga2-farthest vs gces-noGeo | -0.000174 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat7 | nsga2-farthest vs nsgaii | 0.001903 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-farthest vs gces | -0.000054 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces-noComp | -0.000662 | 6/0/15 | 0.059507 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces-noGeo | -0.000451 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat8 | nsga2-farthest vs nsgaii | 0.006375 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-farthest vs gces | -0.001760 | 3/0/18 | 0.000161 | 0.007736 | yes |
| zcat8 | nsga2-farthest vs gces-noComp | 0.002123 | 17/0/4 | 0.004879 | 0.161007 | no |
| zcat8 | nsga2-farthest vs gces-noGeo | -0.001291 | 4/0/17 | 0.000105 | 0.005350 | yes |
| zcat9 | nsga2-farthest vs nsgaii | 0.006375 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-farthest vs gces | -0.001760 | 3/0/18 | 0.000161 | 0.007736 | yes |
| zcat9 | nsga2-farthest vs gces-noComp | 0.002123 | 17/0/4 | 0.004879 | 0.161007 | no |
| zcat9 | nsga2-farthest vs gces-noGeo | -0.001291 | 4/0/17 | 0.000105 | 0.005350 | yes |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | -0.001903 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-farthest vs gces | -0.001285 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-farthest vs gces-noComp | -0.000043 | 13/0/8 | 0.228986 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces-noGeo | -0.001417 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-farthest vs nsgaii | -0.000146 | 13/0/8 | 0.103214 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces | -0.000041 | 12/0/9 | 0.452368 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces-noComp | -0.000035 | 11/0/10 | 0.918694 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces-noGeo | -0.000090 | 13/0/8 | 0.272210 | 1.000000 | no |
| zcat11 | nsga2-farthest vs nsgaii | -0.001253 | 20/0/1 | 0.000426 | 0.019609 | yes |
| zcat11 | nsga2-farthest vs gces | -0.000291 | 17/0/4 | 0.003753 | 0.135098 | no |
| zcat11 | nsga2-farthest vs gces-noComp | 0.000096 | 7/0/14 | 0.038438 | 1.000000 | no |
| zcat11 | nsga2-farthest vs gces-noGeo | -0.000317 | 18/0/3 | 0.002857 | 0.108574 | no |
| zcat12 | nsga2-farthest vs nsgaii | -0.001476 | 20/0/1 | 0.000426 | 0.019609 | yes |
| zcat12 | nsga2-farthest vs gces | -0.000412 | 19/0/2 | 0.000852 | 0.035769 | yes |
| zcat12 | nsga2-farthest vs gces-noComp | -0.000099 | 11/0/10 | 0.411982 | 1.000000 | no |
| zcat12 | nsga2-farthest vs gces-noGeo | -0.000408 | 18/0/3 | 0.001002 | 0.040092 | yes |
| zcat13 | nsga2-farthest vs nsgaii | -0.000892 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-farthest vs gces | -0.000060 | 11/0/10 | 0.473334 | 1.000000 | no |
| zcat13 | nsga2-farthest vs gces-noComp | 0.000141 | 5/0/16 | 0.023854 | 0.715628 | no |
| zcat13 | nsga2-farthest vs gces-noGeo | 0.000056 | 8/0/13 | 0.759288 | 1.000000 | no |
| zcat14 | nsga2-farthest vs nsgaii | -0.001693 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-farthest vs gces | -0.000561 | 18/0/3 | 0.000041 | 0.002091 | yes |
| zcat14 | nsga2-farthest vs gces-noComp | -0.000152 | 12/0/9 | 0.516761 | 1.000000 | no |
| zcat14 | nsga2-farthest vs gces-noGeo | -0.000721 | 20/0/1 | 0.000007 | 0.000381 | yes |
| zcat15 | nsga2-farthest vs nsgaii | -0.001511 | 20/0/1 | 0.000426 | 0.019609 | yes |
| zcat15 | nsga2-farthest vs gces | -0.000391 | 19/0/2 | 0.000852 | 0.035769 | yes |
| zcat15 | nsga2-farthest vs gces-noComp | -0.000088 | 11/0/10 | 0.494802 | 1.000000 | no |
| zcat15 | nsga2-farthest vs gces-noGeo | -0.000386 | 18/0/3 | 0.001176 | 0.045859 | yes |
| zcat16 | nsga2-farthest vs nsgaii | -0.000741 | 20/0/1 | 0.000426 | 0.019609 | yes |
| zcat16 | nsga2-farthest vs gces | 0.000479 | 5/0/16 | 0.000293 | 0.014053 | yes |
| zcat16 | nsga2-farthest vs gces-noComp | 0.000372 | 6/0/15 | 0.011347 | 0.363098 | no |
| zcat16 | nsga2-farthest vs gces-noGeo | 0.000417 | 3/0/18 | 0.000354 | 0.016629 | yes |
| zcat17 | nsga2-farthest vs nsgaii | -0.000908 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-farthest vs gces | -0.000281 | 15/0/6 | 0.005542 | 0.182879 | no |
| zcat17 | nsga2-farthest vs gces-noComp | 0.000087 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat17 | nsga2-farthest vs gces-noGeo | -0.000405 | 19/0/2 | 0.000010 | 0.000525 | yes |
| zcat18 | nsga2-farthest vs nsgaii | -0.000657 | 20/0/1 | 0.000002 | 0.000116 | yes |
| zcat18 | nsga2-farthest vs gces | -0.000017 | 11/0/10 | 0.411982 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces-noComp | 0.000307 | 6/0/15 | 0.050192 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces-noGeo | 0.000134 | 6/0/15 | 0.242842 | 1.000000 | no |
| zcat19 | nsga2-farthest vs nsgaii | -0.001380 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-farthest vs gces | -0.000916 | 20/0/1 | 0.000002 | 0.000116 | yes |
| zcat19 | nsga2-farthest vs gces-noComp | 0.000203 | 8/0/13 | 0.059507 | 1.000000 | no |
| zcat19 | nsga2-farthest vs gces-noGeo | -0.000938 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-farthest vs nsgaii | -0.001082 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-farthest vs gces | -0.000352 | 18/0/3 | 0.000067 | 0.003271 | yes |
| zcat2 | nsga2-farthest vs gces-noComp | 0.000159 | 10/0/11 | 0.539189 | 1.000000 | no |
| zcat2 | nsga2-farthest vs gces-noGeo | -0.000413 | 20/0/1 | 0.000052 | 0.002623 | yes |
| zcat20 | nsga2-farthest vs nsgaii | -0.000662 | 20/0/1 | 0.000002 | 0.000116 | yes |
| zcat20 | nsga2-farthest vs gces | 0.000001 | 11/0/10 | 0.392584 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces-noComp | 0.000291 | 5/0/16 | 0.050192 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces-noGeo | 0.000143 | 6/0/15 | 0.257248 | 1.000000 | no |
| zcat3 | nsga2-farthest vs nsgaii | -0.002098 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-farthest vs gces | -0.001234 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-farthest vs gces-noComp | -0.000264 | 11/0/10 | 0.190687 | 1.000000 | no |
| zcat3 | nsga2-farthest vs gces-noGeo | -0.001341 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-farthest vs nsgaii | -0.002056 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-farthest vs gces | -0.001267 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-farthest vs gces-noComp | -0.000235 | 11/0/10 | 0.178988 | 1.000000 | no |
| zcat4 | nsga2-farthest vs gces-noGeo | -0.001308 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-farthest vs nsgaii | -0.000884 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-farthest vs gces | -0.000281 | 15/0/6 | 0.004285 | 0.145685 | no |
| zcat5 | nsga2-farthest vs gces-noComp | 0.000097 | 7/0/14 | 0.128078 | 1.000000 | no |
| zcat5 | nsga2-farthest vs gces-noGeo | -0.000384 | 19/0/2 | 0.000007 | 0.000381 | yes |
| zcat6 | nsga2-farthest vs nsgaii | -0.000317 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-farthest vs gces | -0.000323 | 19/0/2 | 0.000024 | 0.001240 | yes |
| zcat6 | nsga2-farthest vs gces-noComp | 0.000214 | 1/0/20 | 0.000013 | 0.000721 | yes |
| zcat6 | nsga2-farthest vs gces-noGeo | -0.000294 | 20/0/1 | 0.000018 | 0.000960 | yes |
| zcat7 | nsga2-farthest vs nsgaii | -0.000665 | 20/0/1 | 0.000002 | 0.000116 | yes |
| zcat7 | nsga2-farthest vs gces | -0.000022 | 11/0/10 | 0.320457 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces-noComp | 0.000286 | 6/0/15 | 0.054693 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces-noGeo | 0.000146 | 6/0/15 | 0.287734 | 1.000000 | no |
| zcat8 | nsga2-farthest vs nsgaii | -0.001283 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-farthest vs gces | -0.000067 | 12/0/9 | 0.864887 | 1.000000 | no |
| zcat8 | nsga2-farthest vs gces-noComp | -0.000394 | 16/0/5 | 0.003279 | 0.121313 | no |
| zcat8 | nsga2-farthest vs gces-noGeo | 0.000241 | 5/0/16 | 0.019473 | 0.603665 | no |
| zcat9 | nsga2-farthest vs nsgaii | -0.001291 | 21/0/0 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-farthest vs gces | -0.000065 | 12/0/9 | 0.864887 | 1.000000 | no |
| zcat9 | nsga2-farthest vs gces-noComp | -0.000418 | 16/0/5 | 0.003753 | 0.135098 | no |
| zcat9 | nsga2-farthest vs gces-noGeo | 0.000179 | 6/0/15 | 0.070137 | 1.000000 | no |
