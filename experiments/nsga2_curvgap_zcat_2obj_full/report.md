# nsga2-curvgap 2-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across nsga2_curvgap and the GCES-family variants.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, nsga2-curvgap, gces-noComp, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-curvgap vs nsgaii, nsga2-curvgap vs gces, nsga2-curvgap vs gces-noComp, nsga2-curvgap vs gces-noGeo

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-curvgap | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 0.981731 | 0.626635 | 0.986292 | 0.983564 | 0.983460 |
| zcat2 | 0.991373 | 0.921425 | 0.993167 | 0.992147 | 0.992425 |
| zcat3 | 0.981568 | 0.450056 | 0.985894 | 0.983868 | 0.983779 |
| zcat4 | 0.981568 | 0.450069 | 0.985894 | 0.983869 | 0.983779 |
| zcat5 | 0.994124 | 0.848305 | 0.995465 | 0.995305 | 0.995301 |
| zcat6 | 0.994481 | 0.989292 | 0.995720 | 0.995249 | 0.995187 |
| zcat7 | 0.989018 | 0.930441 | 0.991583 | 0.991373 | 0.990975 |
| zcat8 | 0.967947 | 0.893205 | 0.972198 | 0.975612 | 0.976081 |
| zcat9 | 0.967941 | 0.893200 | 0.972193 | 0.975607 | 0.976076 |
| zcat10 | 0.946007 | 0.582130 | 0.950280 | 0.956072 | 0.955414 |
| zcat11 | 0.988725 | 0.950883 | 0.991733 | 0.990634 | 0.990728 |
| zcat12 | 0.988428 | 0.929045 | 0.991270 | 0.990488 | 0.990517 |
| zcat13 | 1.015344 | 0.981089 | 1.017635 | 1.017026 | 1.016928 |
| zcat14 | 0.975055 | 0.779081 | 0.980733 | 0.978889 | 0.979122 |
| zcat15 | 0.988429 | 0.929045 | 0.991271 | 0.990488 | 0.990518 |
| zcat16 | 0.986793 | 0.956245 | 0.989628 | 0.989972 | 0.990009 |
| zcat17 | 0.994120 | 0.848281 | 0.995461 | 0.995301 | 0.995298 |
| zcat18 | 0.989018 | 0.930441 | 0.991582 | 0.991372 | 0.990975 |
| zcat19 | 0.984635 | 0.814002 | 0.988662 | 0.986068 | 0.986111 |
| zcat20 | 0.989016 | 0.930424 | 0.991581 | 0.991371 | 0.990974 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-curvgap | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 0.007839 | 0.160457 | 0.005979 | 0.007352 | 0.007221 |
| zcat2 | 0.006222 | 0.045089 | 0.004981 | 0.005553 | 0.005492 |
| zcat3 | 0.007943 | 0.220627 | 0.006109 | 0.007186 | 0.007079 |
| zcat4 | 0.007915 | 0.222040 | 0.006094 | 0.007167 | 0.007126 |
| zcat5 | 0.004491 | 0.173565 | 0.003509 | 0.003990 | 0.003888 |
| zcat6 | 0.002882 | 0.008268 | 0.002351 | 0.002859 | 0.002888 |
| zcat7 | 0.004714 | 0.049299 | 0.003763 | 0.003903 | 0.004071 |
| zcat8 | 0.006383 | 0.020894 | 0.005494 | 0.004859 | 0.005167 |
| zcat9 | 0.006371 | 0.020456 | 0.005497 | 0.004900 | 0.005144 |
| zcat10 | 0.003497 | 0.043366 | 0.003385 | 0.003441 | 0.003392 |
| zcat11 | 0.005266 | 0.019941 | 0.003917 | 0.004331 | 0.004304 |
| zcat12 | 0.005507 | 0.030675 | 0.004130 | 0.004439 | 0.004442 |
| zcat13 | 0.004038 | 0.012530 | 0.003005 | 0.003090 | 0.003206 |
| zcat14 | 0.007444 | 0.067470 | 0.005902 | 0.006471 | 0.006312 |
| zcat15 | 0.005556 | 0.031521 | 0.004134 | 0.004431 | 0.004436 |
| zcat16 | 0.004940 | 0.014652 | 0.003826 | 0.003782 | 0.003720 |
| zcat17 | 0.004507 | 0.172014 | 0.003512 | 0.004004 | 0.003880 |
| zcat18 | 0.004708 | 0.048974 | 0.003744 | 0.003917 | 0.004068 |
| zcat19 | 0.006491 | 0.070882 | 0.004908 | 0.006049 | 0.006027 |
| zcat20 | 0.004717 | 0.049214 | 0.003763 | 0.003911 | 0.004053 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-curvgap | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- |
| zcat1 | 2.854 | 20.797 | 11.576 | 6.283 | 7.825 |
| zcat2 | 2.823 | 18.820 | 9.933 | 5.606 | 7.015 |
| zcat3 | 2.458 | 17.793 | 9.948 | 5.172 | 6.598 |
| zcat4 | 2.342 | 17.757 | 10.079 | 5.250 | 6.583 |
| zcat5 | 2.458 | 17.669 | 9.483 | 5.545 | 6.674 |
| zcat6 | 2.507 | 16.567 | 8.766 | 4.947 | 6.068 |
| zcat7 | 2.379 | 17.657 | 10.106 | 5.557 | 6.512 |
| zcat8 | 2.455 | 17.559 | 9.494 | 5.479 | 6.985 |
| zcat9 | 2.575 | 18.606 | 10.022 | 5.880 | 7.290 |
| zcat10 | 2.585 | 17.158 | 9.472 | 5.756 | 6.873 |
| zcat11 | 2.510 | 18.395 | 10.218 | 4.931 | 5.857 |
| zcat12 | 2.472 | 18.677 | 10.936 | 5.501 | 6.428 |
| zcat13 | 2.528 | 18.618 | 11.208 | 5.911 | 6.703 |
| zcat14 | 2.465 | 18.321 | 10.157 | 5.481 | 6.859 |
| zcat15 | 2.435 | 18.525 | 10.626 | 5.425 | 6.410 |
| zcat16 | 2.514 | 18.661 | 10.501 | 5.335 | 6.161 |
| zcat17 | 2.605 | 18.463 | 9.817 | 5.786 | 6.745 |
| zcat18 | 2.507 | 18.399 | 10.422 | 5.615 | 6.842 |
| zcat19 | 2.466 | 18.626 | 10.242 | 5.227 | 6.651 |
| zcat20 | 2.595 | 19.159 | 11.282 | 6.095 | 7.425 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat8 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat8 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat8 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat8 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-curvgap vs gces | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat12 | nsga2-curvgap vs gces-noComp | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat12 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat13 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-curvgap vs gces | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat15 | nsga2-curvgap vs gces-noComp | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat15 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat16 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat19 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat20 | nsga2-curvgap vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat20 | nsga2-curvgap vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat20 | nsga2-curvgap vs gces-noComp | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat20 | nsga2-curvgap vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-curvgap vs nsgaii | -0.355096 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-curvgap vs gces | -0.356825 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-curvgap vs gces-noComp | -0.359657 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-curvgap vs gces-noGeo | -0.356929 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs nsgaii | -0.363877 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs gces | -0.373283 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs gces-noComp | -0.368149 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs gces-noGeo | -0.373942 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs nsgaii | -0.037842 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs gces | -0.039844 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs gces-noComp | -0.040850 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs gces-noGeo | -0.039750 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-curvgap vs nsgaii | -0.059383 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-curvgap vs gces | -0.061472 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-curvgap vs gces-noComp | -0.062225 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-curvgap vs gces-noGeo | -0.061443 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs nsgaii | -0.034255 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs gces | -0.035839 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs gces-noComp | -0.036546 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs gces-noGeo | -0.035937 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs nsgaii | -0.195974 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs gces | -0.200041 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs gces-noComp | -0.201653 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs gces-noGeo | -0.199809 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-curvgap vs nsgaii | -0.059384 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-curvgap vs gces | -0.061473 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-curvgap vs gces-noComp | -0.062226 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-curvgap vs gces-noGeo | -0.061443 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs nsgaii | -0.030548 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs gces | -0.033765 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs gces-noComp | -0.033383 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs gces-noGeo | -0.033727 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs nsgaii | -0.145839 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs gces | -0.147017 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs gces-noComp | -0.147180 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs gces-noGeo | -0.147020 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs nsgaii | -0.058577 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs gces | -0.060534 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs gces-noComp | -0.061141 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs gces-noGeo | -0.060931 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs nsgaii | -0.170633 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs gces | -0.172109 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs gces-noComp | -0.174659 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs gces-noGeo | -0.172066 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs nsgaii | -0.069948 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs gces | -0.071000 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs gces-noComp | -0.071742 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs gces-noGeo | -0.070722 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs nsgaii | -0.058592 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs gces | -0.060550 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs gces-noComp | -0.061157 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs gces-noGeo | -0.060947 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs nsgaii | -0.531512 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs gces | -0.533724 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs gces-noComp | -0.535838 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs gces-noGeo | -0.533812 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs nsgaii | -0.531499 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs gces | -0.533710 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs gces-noComp | -0.535825 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs gces-noGeo | -0.533799 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs nsgaii | -0.145819 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs gces | -0.146996 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs gces-noComp | -0.147160 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs gces-noGeo | -0.147000 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs nsgaii | -0.005190 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs gces | -0.005895 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs gces-noComp | -0.006428 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs gces-noGeo | -0.005957 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs nsgaii | -0.058577 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs gces | -0.060534 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs gces-noComp | -0.061142 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs gces-noGeo | -0.060931 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs nsgaii | -0.074742 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs gces | -0.082876 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs gces-noComp | -0.078993 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs gces-noGeo | -0.082407 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs nsgaii | -0.074741 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs gces | -0.082876 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs gces-noComp | -0.078993 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs gces-noGeo | -0.082407 | 0/0/21 | 0.000001 | 0.000076 | yes |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-curvgap vs nsgaii | 0.152618 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-curvgap vs gces | 0.153236 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-curvgap vs gces-noComp | 0.154478 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat1 | nsga2-curvgap vs gces-noGeo | 0.153104 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs nsgaii | 0.039869 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs gces | 0.039974 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs gces-noComp | 0.039981 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat10 | nsga2-curvgap vs gces-noGeo | 0.039925 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs nsgaii | 0.014675 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs gces | 0.015637 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs gces-noComp | 0.016024 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat11 | nsga2-curvgap vs gces-noGeo | 0.015610 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-curvgap vs nsgaii | 0.025168 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat12 | nsga2-curvgap vs gces | 0.026232 | 1/0/20 | 0.000161 | 0.000967 | yes |
| zcat12 | nsga2-curvgap vs gces-noComp | 0.026545 | 1/0/20 | 0.000354 | 0.001415 | yes |
| zcat12 | nsga2-curvgap vs gces-noGeo | 0.026236 | 1/0/20 | 0.000354 | 0.001415 | yes |
| zcat13 | nsga2-curvgap vs nsgaii | 0.008492 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs gces | 0.009323 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs gces-noComp | 0.009525 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat13 | nsga2-curvgap vs gces-noGeo | 0.009440 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs nsgaii | 0.060027 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs gces | 0.061158 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs gces-noComp | 0.061568 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat14 | nsga2-curvgap vs gces-noGeo | 0.060999 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-curvgap vs nsgaii | 0.025965 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat15 | nsga2-curvgap vs gces | 0.027085 | 1/0/20 | 0.000161 | 0.000967 | yes |
| zcat15 | nsga2-curvgap vs gces-noComp | 0.027387 | 1/0/20 | 0.000354 | 0.001415 | yes |
| zcat15 | nsga2-curvgap vs gces-noGeo | 0.027090 | 1/0/20 | 0.000354 | 0.001415 | yes |
| zcat16 | nsga2-curvgap vs nsgaii | 0.009712 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs gces | 0.010932 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs gces-noComp | 0.010825 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat16 | nsga2-curvgap vs gces-noGeo | 0.010870 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs nsgaii | 0.167507 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs gces | 0.168134 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs gces-noComp | 0.168501 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat17 | nsga2-curvgap vs gces-noGeo | 0.168010 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs nsgaii | 0.044266 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs gces | 0.044906 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs gces-noComp | 0.045230 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat18 | nsga2-curvgap vs gces-noGeo | 0.045057 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs nsgaii | 0.064391 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs gces | 0.064855 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs gces-noComp | 0.065975 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat19 | nsga2-curvgap vs gces-noGeo | 0.064834 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs nsgaii | 0.038866 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs gces | 0.039597 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs gces-noComp | 0.040108 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat2 | nsga2-curvgap vs gces-noGeo | 0.039535 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs nsgaii | 0.044497 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs gces | 0.045161 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs gces-noComp | 0.045451 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat20 | nsga2-curvgap vs gces-noGeo | 0.045303 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs nsgaii | 0.212684 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs gces | 0.213548 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs gces-noComp | 0.214518 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat3 | nsga2-curvgap vs gces-noGeo | 0.213441 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs nsgaii | 0.214126 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs gces | 0.214914 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs gces-noComp | 0.215946 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat4 | nsga2-curvgap vs gces-noGeo | 0.214874 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs nsgaii | 0.169074 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs gces | 0.169677 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs gces-noComp | 0.170056 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat5 | nsga2-curvgap vs gces-noGeo | 0.169575 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs nsgaii | 0.005386 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs gces | 0.005380 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs gces-noComp | 0.005917 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat6 | nsga2-curvgap vs gces-noGeo | 0.005409 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs nsgaii | 0.044585 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs gces | 0.045228 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs gces-noComp | 0.045536 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat7 | nsga2-curvgap vs gces-noGeo | 0.045396 | 1/0/20 | 0.000002 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs nsgaii | 0.014512 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs gces | 0.015728 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs gces-noComp | 0.015400 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat8 | nsga2-curvgap vs gces-noGeo | 0.016035 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs nsgaii | 0.014085 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs gces | 0.015312 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs gces-noComp | 0.014959 | 0/0/21 | 0.000001 | 0.000076 | yes |
| zcat9 | nsga2-curvgap vs gces-noGeo | 0.015556 | 0/0/21 | 0.000001 | 0.000076 | yes |
