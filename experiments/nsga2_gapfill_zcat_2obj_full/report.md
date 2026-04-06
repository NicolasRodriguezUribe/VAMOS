# nsga2-gapfill 2-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across nsga2_gapfill and the GCES-family variants.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, nsga2-gapfill, gces-noComp, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-gapfill vs nsgaii, nsga2-gapfill vs gces, nsga2-gapfill vs gces-noComp, nsga2-gapfill vs gces-noGeo

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-gapfill | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 0.981731 | 0.980283 | 0.986292 | 0.983564 | 0.983460 |
| zcat2 | 0.991373 | 0.989838 | 0.993167 | 0.992147 | 0.992425 |
| zcat3 | 0.981568 | 0.979399 | 0.985894 | 0.983868 | 0.983779 |
| zcat4 | 0.981568 | 0.979399 | 0.985894 | 0.983869 | 0.983779 |
| zcat5 | 0.994124 | 0.990416 | 0.995465 | 0.995305 | 0.995301 |
| zcat6 | 0.994481 | 0.992957 | 0.995720 | 0.995249 | 0.995187 |
| zcat7 | 0.989018 | 0.982105 | 0.991583 | 0.991373 | 0.990975 |
| zcat8 | 0.967947 | 0.957786 | 0.972198 | 0.975612 | 0.976081 |
| zcat9 | 0.967941 | 0.957781 | 0.972193 | 0.975607 | 0.976076 |
| zcat10 | 0.946007 | 0.899104 | 0.950280 | 0.956072 | 0.955414 |
| zcat11 | 0.988725 | 0.984382 | 0.991733 | 0.990634 | 0.990728 |
| zcat12 | 0.988428 | 0.953799 | 0.991270 | 0.990488 | 0.990517 |
| zcat13 | 1.015344 | 0.983922 | 1.017635 | 1.017026 | 1.016928 |
| zcat14 | 0.975055 | 0.967447 | 0.980733 | 0.978889 | 0.979122 |
| zcat15 | 0.988429 | 0.953800 | 0.991271 | 0.990488 | 0.990518 |
| zcat16 | 0.986793 | 0.942661 | 0.989628 | 0.989972 | 0.990009 |
| zcat17 | 0.994120 | 0.990411 | 0.995461 | 0.995301 | 0.995298 |
| zcat18 | 0.989018 | 0.982105 | 0.991582 | 0.991372 | 0.990975 |
| zcat19 | 0.984635 | 0.980747 | 0.988662 | 0.986068 | 0.986111 |
| zcat20 | 0.989016 | 0.982102 | 0.991581 | 0.991371 | 0.990974 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-gapfill | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 0.007839 | 0.008284 | 0.005979 | 0.007352 | 0.007221 |
| zcat2 | 0.006222 | 0.007037 | 0.004981 | 0.005553 | 0.005492 |
| zcat3 | 0.007943 | 0.008796 | 0.006109 | 0.007186 | 0.007079 |
| zcat4 | 0.007915 | 0.008731 | 0.006094 | 0.007167 | 0.007126 |
| zcat5 | 0.004491 | 0.011271 | 0.003509 | 0.003990 | 0.003888 |
| zcat6 | 0.002882 | 0.003996 | 0.002351 | 0.002859 | 0.002888 |
| zcat7 | 0.004714 | 0.009472 | 0.003763 | 0.003903 | 0.004071 |
| zcat8 | 0.006383 | 0.008109 | 0.005494 | 0.004859 | 0.005167 |
| zcat9 | 0.006371 | 0.008151 | 0.005497 | 0.004900 | 0.005144 |
| zcat10 | 0.003497 | 0.009476 | 0.003385 | 0.003441 | 0.003392 |
| zcat11 | 0.005266 | 0.006750 | 0.003917 | 0.004331 | 0.004304 |
| zcat12 | 0.005507 | 0.016034 | 0.004130 | 0.004439 | 0.004442 |
| zcat13 | 0.004038 | 0.010460 | 0.003005 | 0.003090 | 0.003206 |
| zcat14 | 0.007444 | 0.009458 | 0.005902 | 0.006471 | 0.006312 |
| zcat15 | 0.005556 | 0.016004 | 0.004134 | 0.004431 | 0.004436 |
| zcat16 | 0.004940 | 0.019531 | 0.003826 | 0.003782 | 0.003720 |
| zcat17 | 0.004507 | 0.011119 | 0.003512 | 0.004004 | 0.003880 |
| zcat18 | 0.004708 | 0.009562 | 0.003744 | 0.003917 | 0.004068 |
| zcat19 | 0.006491 | 0.008860 | 0.004908 | 0.006049 | 0.006027 |
| zcat20 | 0.004717 | 0.009520 | 0.003763 | 0.003911 | 0.004053 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-gapfill | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 2.856 | 19.224 | 11.664 | 6.338 | 7.844 |
| zcat2 | 2.835 | 17.157 | 9.930 | 5.638 | 7.023 |
| zcat3 | 2.469 | 16.742 | 9.851 | 5.295 | 6.474 |
| zcat4 | 2.459 | 16.303 | 10.056 | 5.327 | 6.567 |
| zcat5 | 2.435 | 15.230 | 9.421 | 5.567 | 6.777 |
| zcat6 | 2.485 | 13.969 | 8.849 | 4.875 | 5.970 |
| zcat7 | 2.401 | 14.982 | 10.155 | 5.515 | 6.574 |
| zcat8 | 2.405 | 15.321 | 9.463 | 5.521 | 6.859 |
| zcat9 | 2.427 | 15.408 | 10.076 | 5.891 | 7.302 |
| zcat10 | 2.600 | 14.520 | 9.406 | 5.816 | 6.970 |
| zcat11 | 2.575 | 16.387 | 10.453 | 4.921 | 5.741 |
| zcat12 | 2.456 | 15.965 | 10.640 | 5.308 | 6.361 |
| zcat13 | 2.568 | 16.239 | 10.999 | 5.863 | 6.799 |
| zcat14 | 2.526 | 16.769 | 10.156 | 5.568 | 7.064 |
| zcat15 | 2.484 | 16.060 | 10.514 | 5.255 | 6.291 |
| zcat16 | 2.485 | 15.190 | 10.287 | 4.971 | 5.856 |
| zcat17 | 2.533 | 15.699 | 10.010 | 6.096 | 7.237 |
| zcat18 | 2.683 | 16.156 | 10.914 | 5.889 | 6.865 |
| zcat19 | 2.539 | 16.313 | 10.049 | 4.995 | 6.417 |
| zcat20 | 2.448 | 15.535 | 10.619 | 5.824 | 7.013 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-gapfill vs nsgaii | 6/0/15 | 5/0/16 | 5/1/15 |
| zcat1 | nsga2-gapfill vs gces | 1/0/20 | 3/0/18 | 1/2/18 |
| zcat1 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat2 | nsga2-gapfill vs nsgaii | 1/0/20 | 1/0/20 | 0/2/19 |
| zcat2 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-gapfill vs nsgaii | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat3 | nsga2-gapfill vs gces | 0/0/21 | 2/0/19 | 0/2/19 |
| zcat3 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-gapfill vs nsgaii | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat4 | nsga2-gapfill vs gces | 0/0/21 | 2/0/19 | 0/2/19 |
| zcat4 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-gapfill vs nsgaii | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat7 | nsga2-gapfill vs gces | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat7 | nsga2-gapfill vs gces-noComp | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat7 | nsga2-gapfill vs gces-noGeo | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat8 | nsga2-gapfill vs nsgaii | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat8 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat8 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat8 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-gapfill vs nsgaii | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat9 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat11 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-gapfill vs nsgaii | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat14 | nsga2-gapfill vs gces | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat14 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-gapfill vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-gapfill vs nsgaii | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat18 | nsga2-gapfill vs gces | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat18 | nsga2-gapfill vs gces-noComp | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat18 | nsga2-gapfill vs gces-noGeo | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat19 | nsga2-gapfill vs nsgaii | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat19 | nsga2-gapfill vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-gapfill vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-gapfill vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat20 | nsga2-gapfill vs nsgaii | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat20 | nsga2-gapfill vs gces | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat20 | nsga2-gapfill vs gces-noComp | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat20 | nsga2-gapfill vs gces-noGeo | 1/0/20 | 2/0/19 | 1/1/19 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-gapfill vs nsgaii | -0.001448 | 6/0/15 | 0.001002 | 0.003806 | yes |
| zcat1 | nsga2-gapfill vs gces | -0.003177 | 1/0/20 | 0.000007 | 0.000113 | yes |
| zcat1 | nsga2-gapfill vs gces-noComp | -0.006009 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-gapfill vs gces-noGeo | -0.003281 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs nsgaii | -0.046903 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs gces | -0.056310 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs gces-noComp | -0.051176 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs gces-noGeo | -0.056969 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs nsgaii | -0.004343 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs gces | -0.006346 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs gces-noComp | -0.007351 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs gces-noGeo | -0.006252 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs nsgaii | -0.034628 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs gces | -0.036717 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs gces-noComp | -0.037471 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs gces-noGeo | -0.036688 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs nsgaii | -0.031423 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs gces | -0.033007 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs gces-noComp | -0.033713 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs gces-noGeo | -0.033104 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-gapfill vs nsgaii | -0.007608 | 3/0/18 | 0.000024 | 0.000334 | yes |
| zcat14 | nsga2-gapfill vs gces | -0.011675 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat14 | nsga2-gapfill vs gces-noComp | -0.013287 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-gapfill vs gces-noGeo | -0.011442 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs nsgaii | -0.034629 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs gces | -0.036718 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs gces-noComp | -0.037471 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs gces-noGeo | -0.036688 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs nsgaii | -0.044132 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs gces | -0.047349 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs gces-noComp | -0.046967 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs gces-noGeo | -0.047311 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs nsgaii | -0.003709 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs gces | -0.004887 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs gces-noComp | -0.005050 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs gces-noGeo | -0.004890 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-gapfill vs nsgaii | -0.006913 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat18 | nsga2-gapfill vs gces | -0.008870 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat18 | nsga2-gapfill vs gces-noComp | -0.009478 | 1/0/20 | 0.000354 | 0.003806 | yes |
| zcat18 | nsga2-gapfill vs gces-noGeo | -0.009267 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat19 | nsga2-gapfill vs nsgaii | -0.003888 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat19 | nsga2-gapfill vs gces | -0.005364 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-gapfill vs gces-noComp | -0.007915 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-gapfill vs gces-noGeo | -0.005321 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs nsgaii | -0.001535 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs gces | -0.002588 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs gces-noComp | -0.003329 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs gces-noGeo | -0.002309 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-gapfill vs nsgaii | -0.006914 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat20 | nsga2-gapfill vs gces | -0.008872 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat20 | nsga2-gapfill vs gces-noComp | -0.009479 | 1/0/20 | 0.000354 | 0.003806 | yes |
| zcat20 | nsga2-gapfill vs gces-noGeo | -0.009269 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat3 | nsga2-gapfill vs nsgaii | -0.002169 | 2/0/19 | 0.000007 | 0.000113 | yes |
| zcat3 | nsga2-gapfill vs gces | -0.004380 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-gapfill vs gces-noComp | -0.006495 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-gapfill vs gces-noGeo | -0.004469 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-gapfill vs nsgaii | -0.002169 | 2/0/19 | 0.000007 | 0.000113 | yes |
| zcat4 | nsga2-gapfill vs gces | -0.004381 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-gapfill vs gces-noComp | -0.006496 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-gapfill vs gces-noGeo | -0.004470 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs nsgaii | -0.003708 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs gces | -0.004886 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs gces-noComp | -0.005050 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs gces-noGeo | -0.004889 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs nsgaii | -0.001524 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs gces | -0.002230 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs gces-noComp | -0.002763 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs gces-noGeo | -0.002291 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-gapfill vs nsgaii | -0.006913 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat7 | nsga2-gapfill vs gces | -0.008870 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat7 | nsga2-gapfill vs gces-noComp | -0.009478 | 1/0/20 | 0.000354 | 0.003806 | yes |
| zcat7 | nsga2-gapfill vs gces-noGeo | -0.009267 | 1/0/20 | 0.000293 | 0.003806 | yes |
| zcat8 | nsga2-gapfill vs nsgaii | -0.010161 | 2/0/19 | 0.000005 | 0.000091 | yes |
| zcat8 | nsga2-gapfill vs gces | -0.018295 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-gapfill vs gces-noComp | -0.014412 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-gapfill vs gces-noGeo | -0.017826 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-gapfill vs nsgaii | -0.010161 | 2/0/19 | 0.000005 | 0.000091 | yes |
| zcat9 | nsga2-gapfill vs gces | -0.018295 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-gapfill vs gces-noComp | -0.014412 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-gapfill vs gces-noGeo | -0.017826 | 0/0/21 | 0.000001 | 0.000076 | yes |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-gapfill vs nsgaii | 0.000445 | 5/0/16 | 0.001859 | 0.004600 | yes |
| zcat1 | nsga2-gapfill vs gces | 0.001063 | 3/0/18 | 0.000067 | 0.001001 | yes |
| zcat1 | nsga2-gapfill vs gces-noComp | 0.002305 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-gapfill vs gces-noGeo | 0.000932 | 1/0/20 | 0.000084 | 0.001175 | yes |
| zcat10 | nsga2-gapfill vs nsgaii | 0.005980 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs gces | 0.006084 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs gces-noComp | 0.006091 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-gapfill vs gces-noGeo | 0.006036 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs nsgaii | 0.001483 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs gces | 0.002446 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs gces-noComp | 0.002832 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-gapfill vs gces-noGeo | 0.002419 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs nsgaii | 0.010527 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs gces | 0.011592 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs gces-noComp | 0.011905 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-gapfill vs gces-noGeo | 0.011595 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs nsgaii | 0.006423 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs gces | 0.007254 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs gces-noComp | 0.007455 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-gapfill vs gces-noGeo | 0.007370 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-gapfill vs nsgaii | 0.002014 | 3/0/18 | 0.000052 | 0.000839 | yes |
| zcat14 | nsga2-gapfill vs gces | 0.003146 | 2/0/19 | 0.000005 | 0.000110 | yes |
| zcat14 | nsga2-gapfill vs gces-noComp | 0.003556 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-gapfill vs gces-noGeo | 0.002987 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs nsgaii | 0.010448 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs gces | 0.011568 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs gces-noComp | 0.011871 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-gapfill vs gces-noGeo | 0.011573 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs nsgaii | 0.014591 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs gces | 0.015811 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs gces-noComp | 0.015705 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-gapfill vs gces-noGeo | 0.015749 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs nsgaii | 0.006613 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs gces | 0.007239 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs gces-noComp | 0.007607 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-gapfill vs gces-noGeo | 0.007115 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-gapfill vs nsgaii | 0.004853 | 1/0/20 | 0.000354 | 0.004600 | yes |
| zcat18 | nsga2-gapfill vs gces | 0.005494 | 2/0/19 | 0.000426 | 0.004600 | yes |
| zcat18 | nsga2-gapfill vs gces-noComp | 0.005818 | 1/0/20 | 0.000426 | 0.004600 | yes |
| zcat18 | nsga2-gapfill vs gces-noGeo | 0.005645 | 2/0/19 | 0.000426 | 0.004600 | yes |
| zcat19 | nsga2-gapfill vs nsgaii | 0.002368 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat19 | nsga2-gapfill vs gces | 0.002833 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-gapfill vs gces-noComp | 0.003952 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-gapfill vs gces-noGeo | 0.002811 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs nsgaii | 0.000814 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs gces | 0.001545 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs gces-noComp | 0.002056 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-gapfill vs gces-noGeo | 0.001483 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-gapfill vs nsgaii | 0.004804 | 1/0/20 | 0.000354 | 0.004600 | yes |
| zcat20 | nsga2-gapfill vs gces | 0.005468 | 2/0/19 | 0.000426 | 0.004600 | yes |
| zcat20 | nsga2-gapfill vs gces-noComp | 0.005757 | 1/0/20 | 0.000426 | 0.004600 | yes |
| zcat20 | nsga2-gapfill vs gces-noGeo | 0.005610 | 2/0/19 | 0.000426 | 0.004600 | yes |
| zcat3 | nsga2-gapfill vs nsgaii | 0.000853 | 2/0/19 | 0.000031 | 0.000566 | yes |
| zcat3 | nsga2-gapfill vs gces | 0.001717 | 2/0/19 | 0.000010 | 0.000191 | yes |
| zcat3 | nsga2-gapfill vs gces-noComp | 0.002687 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-gapfill vs gces-noGeo | 0.001610 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-gapfill vs nsgaii | 0.000817 | 2/0/19 | 0.000031 | 0.000566 | yes |
| zcat4 | nsga2-gapfill vs gces | 0.001605 | 2/0/19 | 0.000010 | 0.000191 | yes |
| zcat4 | nsga2-gapfill vs gces-noComp | 0.002637 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-gapfill vs gces-noGeo | 0.001565 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs nsgaii | 0.006781 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs gces | 0.007384 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs gces-noComp | 0.007762 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-gapfill vs gces-noGeo | 0.007281 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs nsgaii | 0.001114 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs gces | 0.001108 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs gces-noComp | 0.001644 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-gapfill vs gces-noGeo | 0.001137 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-gapfill vs nsgaii | 0.004758 | 1/0/20 | 0.000354 | 0.004600 | yes |
| zcat7 | nsga2-gapfill vs gces | 0.005401 | 2/0/19 | 0.000426 | 0.004600 | yes |
| zcat7 | nsga2-gapfill vs gces-noComp | 0.005709 | 1/0/20 | 0.000426 | 0.004600 | yes |
| zcat7 | nsga2-gapfill vs gces-noGeo | 0.005569 | 2/0/19 | 0.000426 | 0.004600 | yes |
| zcat8 | nsga2-gapfill vs nsgaii | 0.001726 | 2/0/19 | 0.000005 | 0.000110 | yes |
| zcat8 | nsga2-gapfill vs gces | 0.002942 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-gapfill vs gces-noComp | 0.002615 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-gapfill vs gces-noGeo | 0.003249 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-gapfill vs nsgaii | 0.001780 | 2/0/19 | 0.000005 | 0.000110 | yes |
| zcat9 | nsga2-gapfill vs gces | 0.003007 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-gapfill vs gces-noComp | 0.002654 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-gapfill vs gces-noGeo | 0.003250 | 0/0/21 | 0.000001 | 0.000076 | yes |
