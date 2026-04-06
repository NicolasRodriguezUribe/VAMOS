# Farthest-family 2-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the farthest-derived selectors and the GCES-family variants.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, nsga2-farthest, nsga2-hvfarthest, nsga2-refcover-farthest, nsga2-hvref-farthest, gces-noComp, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-farthest vs nsgaii, nsga2-farthest vs gces-noGeo, nsga2-farthest vs gces, nsga2-hvfarthest vs nsgaii, nsga2-hvfarthest vs nsga2-farthest, nsga2-hvfarthest vs gces-noGeo, nsga2-hvfarthest vs gces, nsga2-refcover-farthest vs nsgaii, nsga2-refcover-farthest vs nsga2-farthest, nsga2-refcover-farthest vs gces-noGeo, nsga2-refcover-farthest vs gces, nsga2-hvref-farthest vs nsgaii, nsga2-hvref-farthest vs nsga2-farthest, nsga2-hvref-farthest vs gces-noGeo, nsga2-hvref-farthest vs gces, nsga2-hvref-farthest vs nsga2-hvfarthest, nsga2-hvref-farthest vs nsga2-refcover-farthest

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-refcover-farthest | nsga2-hvref-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.981731 | 0.986382 | 0.989015 | 0.980510 | 0.987852 | 0.986292 | 0.983564 | 0.983460 |
| zcat2 | 0.991373 | 0.992855 | 0.995017 | 0.988535 | 0.992044 | 0.993167 | 0.992147 | 0.992425 |
| zcat3 | 0.981568 | 0.986511 | 0.989030 | 0.982152 | 0.987765 | 0.985894 | 0.983868 | 0.983779 |
| zcat4 | 0.981568 | 0.986512 | 0.989031 | 0.982152 | 0.987766 | 0.985894 | 0.983869 | 0.983779 |
| zcat5 | 0.994124 | 0.995304 | 0.996976 | 0.992553 | 0.995431 | 0.995465 | 0.995305 | 0.995301 |
| zcat6 | 0.994481 | 0.995075 | 0.996839 | 0.989260 | 0.996213 | 0.995720 | 0.995249 | 0.995187 |
| zcat7 | 0.989018 | 0.990921 | 0.994036 | 0.988285 | 0.992334 | 0.991583 | 0.991373 | 0.990975 |
| zcat8 | 0.967947 | 0.974321 | 0.982211 | 0.977128 | 0.980875 | 0.972198 | 0.975612 | 0.976081 |
| zcat9 | 0.967941 | 0.974316 | 0.982206 | 0.977123 | 0.980870 | 0.972193 | 0.975607 | 0.976076 |
| zcat10 | 0.946007 | 0.948984 | 0.970217 | 0.504835 | 0.964817 | 0.950280 | 0.956072 | 0.955414 |
| zcat11 | 0.988725 | 0.991531 | 0.993710 | 0.988087 | 0.992473 | 0.991733 | 0.990634 | 0.990728 |
| zcat12 | 0.988428 | 0.991547 | 0.993469 | 0.987508 | 0.992280 | 0.991270 | 0.990488 | 0.990517 |
| zcat13 | 1.015344 | 1.017381 | 1.019075 | 1.014793 | 1.017757 | 1.017635 | 1.017026 | 1.016928 |
| zcat14 | 0.975055 | 0.980920 | 0.985545 | 0.984281 | 0.985270 | 0.980733 | 0.978889 | 0.979122 |
| zcat15 | 0.988429 | 0.991547 | 0.993469 | 0.987509 | 0.992281 | 0.991271 | 0.990488 | 0.990518 |
| zcat16 | 0.986793 | 0.988881 | 0.992716 | 0.984815 | 0.991877 | 0.989628 | 0.989972 | 0.990009 |
| zcat17 | 0.994120 | 0.995300 | 0.996972 | 0.992549 | 0.995427 | 0.995461 | 0.995301 | 0.995298 |
| zcat18 | 0.989018 | 0.990921 | 0.994035 | 0.988285 | 0.992334 | 0.991582 | 0.991372 | 0.990975 |
| zcat19 | 0.984635 | 0.988134 | 0.991257 | 0.983442 | 0.989714 | 0.988662 | 0.986068 | 0.986111 |
| zcat20 | 0.989016 | 0.990919 | 0.994035 | 0.988283 | 0.992333 | 0.991581 | 0.991371 | 0.990974 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-refcover-farthest | nsga2-hvref-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.007839 | 0.005936 | 0.004830 | 0.008405 | 0.005365 | 0.005979 | 0.007352 | 0.007221 |
| zcat2 | 0.006222 | 0.005140 | 0.003699 | 0.008170 | 0.005742 | 0.004981 | 0.005553 | 0.005492 |
| zcat3 | 0.007943 | 0.005845 | 0.004799 | 0.007671 | 0.005336 | 0.006109 | 0.007186 | 0.007079 |
| zcat4 | 0.007915 | 0.005859 | 0.004805 | 0.007661 | 0.005331 | 0.006094 | 0.007167 | 0.007126 |
| zcat5 | 0.004491 | 0.003606 | 0.002481 | 0.005539 | 0.003510 | 0.003509 | 0.003990 | 0.003888 |
| zcat6 | 0.002882 | 0.002565 | 0.001773 | 0.005267 | 0.002062 | 0.002351 | 0.002859 | 0.002888 |
| zcat7 | 0.004714 | 0.004049 | 0.002745 | 0.005146 | 0.003443 | 0.003763 | 0.003903 | 0.004071 |
| zcat8 | 0.006383 | 0.005100 | 0.003643 | 0.004501 | 0.003875 | 0.005494 | 0.004859 | 0.005167 |
| zcat9 | 0.006371 | 0.005079 | 0.003648 | 0.004508 | 0.003876 | 0.005497 | 0.004900 | 0.005144 |
| zcat10 | 0.003497 | 0.003351 | 0.002042 | 0.029279 | 0.002395 | 0.003385 | 0.003441 | 0.003392 |
| zcat11 | 0.005266 | 0.004013 | 0.003034 | 0.005649 | 0.003607 | 0.003917 | 0.004331 | 0.004304 |
| zcat12 | 0.005507 | 0.004031 | 0.003145 | 0.005916 | 0.003674 | 0.004130 | 0.004439 | 0.004442 |
| zcat13 | 0.004038 | 0.003146 | 0.002345 | 0.004297 | 0.002946 | 0.003005 | 0.003090 | 0.003206 |
| zcat14 | 0.007444 | 0.005750 | 0.004455 | 0.004799 | 0.004503 | 0.005902 | 0.006471 | 0.006312 |
| zcat15 | 0.005556 | 0.004046 | 0.003151 | 0.005937 | 0.003677 | 0.004134 | 0.004431 | 0.004436 |
| zcat16 | 0.004940 | 0.004199 | 0.002736 | 0.005311 | 0.003059 | 0.003826 | 0.003782 | 0.003720 |
| zcat17 | 0.004507 | 0.003599 | 0.002482 | 0.005550 | 0.003514 | 0.003512 | 0.004004 | 0.003880 |
| zcat18 | 0.004708 | 0.004051 | 0.002743 | 0.005157 | 0.003454 | 0.003744 | 0.003917 | 0.004068 |
| zcat19 | 0.006491 | 0.005111 | 0.003941 | 0.007123 | 0.004486 | 0.004908 | 0.006049 | 0.006027 |
| zcat20 | 0.004717 | 0.004054 | 0.002760 | 0.005176 | 0.003453 | 0.003763 | 0.003911 | 0.004053 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-refcover-farthest | nsga2-hvref-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 3.155 | 32.409 | 100.174 | 64.977 | 155.102 | 12.595 | 6.858 | 9.372 |
| zcat2 | 3.474 | 29.700 | 95.563 | 57.191 | 133.790 | 11.852 | 6.732 | 8.310 |
| zcat3 | 2.966 | 37.549 | 128.015 | 84.043 | 198.833 | 15.726 | 8.047 | 9.921 |
| zcat4 | 3.868 | 41.482 | 126.797 | 90.858 | 201.643 | 16.007 | 8.368 | 9.980 |
| zcat5 | 3.727 | 35.959 | 116.620 | 76.624 | 167.497 | 14.118 | 8.780 | 10.420 |
| zcat6 | 3.871 | 31.501 | 98.451 | 68.026 | 159.856 | 13.221 | 7.576 | 9.109 |
| zcat7 | 3.557 | 34.570 | 112.355 | 75.336 | 177.216 | 15.522 | 8.324 | 9.795 |
| zcat8 | 3.609 | 35.276 | 111.865 | 81.792 | 184.870 | 14.322 | 8.288 | 10.348 |
| zcat9 | 3.657 | 35.174 | 113.309 | 81.305 | 186.801 | 14.216 | 8.412 | 10.223 |
| zcat10 | 3.658 | 31.047 | 98.935 | 70.364 | 163.843 | 13.391 | 8.184 | 10.097 |
| zcat11 | 3.810 | 35.678 | 116.654 | 75.533 | 185.170 | 15.370 | 6.992 | 8.355 |
| zcat12 | 3.695 | 36.294 | 114.013 | 76.842 | 181.792 | 15.931 | 8.215 | 9.403 |
| zcat13 | 3.678 | 35.476 | 113.637 | 76.015 | 175.142 | 15.949 | 9.070 | 9.794 |
| zcat14 | 3.837 | 37.300 | 126.086 | 91.274 | 206.369 | 15.693 | 8.680 | 11.289 |
| zcat15 | 3.835 | 37.917 | 123.239 | 81.970 | 191.199 | 17.491 | 8.825 | 9.517 |
| zcat16 | 4.096 | 38.755 | 123.450 | 85.579 | 194.937 | 16.892 | 8.249 | 9.609 |
| zcat17 | 3.836 | 37.297 | 122.389 | 77.994 | 186.731 | 15.792 | 9.816 | 11.347 |
| zcat18 | 3.979 | 39.656 | 122.343 | 82.174 | 188.212 | 16.016 | 8.731 | 10.217 |
| zcat19 | 3.705 | 37.982 | 122.996 | 80.568 | 193.950 | 15.144 | 7.594 | 9.498 |
| zcat20 | 3.794 | 36.003 | 116.611 | 78.368 | 180.483 | 16.580 | 9.547 | 11.337 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-farthest vs gces-noGeo | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat1 | nsga2-farthest vs gces | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat1 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-refcover-farthest vs nsgaii | 5/0/16 | 5/0/16 | 5/0/16 |
| zcat1 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-refcover-farthest vs gces-noGeo | 4/0/17 | 5/0/16 | 4/1/16 |
| zcat1 | nsga2-refcover-farthest vs gces | 3/0/18 | 2/0/19 | 2/1/18 |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 18/0/3 | 18/0/3 | 18/0/3 |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-farthest vs gces-noGeo | 19/0/2 | 20/0/1 | 19/1/1 |
| zcat2 | nsga2-farthest vs gces | 16/0/5 | 18/0/3 | 16/2/3 |
| zcat2 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-refcover-farthest vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-hvref-farthest vs nsgaii | 18/0/3 | 18/0/3 | 18/0/3 |
| zcat2 | nsga2-hvref-farthest vs nsga2-farthest | 3/0/18 | 2/0/19 | 2/1/18 |
| zcat2 | nsga2-hvref-farthest vs gces-noGeo | 7/0/14 | 7/0/14 | 6/2/13 |
| zcat2 | nsga2-hvref-farthest vs gces | 6/0/15 | 5/0/16 | 3/5/13 |
| zcat2 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-refcover-farthest vs nsgaii | 14/0/7 | 14/0/7 | 14/0/7 |
| zcat3 | nsga2-refcover-farthest vs nsga2-farthest | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat3 | nsga2-refcover-farthest vs gces-noGeo | 8/0/13 | 8/0/13 | 8/0/13 |
| zcat3 | nsga2-refcover-farthest vs gces | 6/0/15 | 8/0/13 | 6/2/13 |
| zcat3 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs nsga2-farthest | 18/0/3 | 18/0/3 | 18/0/3 |
| zcat3 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-refcover-farthest vs nsgaii | 14/0/7 | 14/0/7 | 14/0/7 |
| zcat4 | nsga2-refcover-farthest vs nsga2-farthest | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat4 | nsga2-refcover-farthest vs gces-noGeo | 8/0/13 | 8/0/13 | 8/0/13 |
| zcat4 | nsga2-refcover-farthest vs gces | 6/0/15 | 7/0/14 | 6/1/14 |
| zcat4 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs nsga2-farthest | 18/0/3 | 18/0/3 | 18/0/3 |
| zcat4 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-farthest vs gces-noGeo | 10/0/11 | 19/0/2 | 10/9/2 |
| zcat5 | nsga2-farthest vs gces | 12/0/9 | 15/0/6 | 12/3/6 |
| zcat5 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-refcover-farthest vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvref-farthest vs nsga2-farthest | 14/0/7 | 14/0/7 | 13/2/6 |
| zcat5 | nsga2-hvref-farthest vs gces-noGeo | 13/0/8 | 20/0/1 | 13/7/1 |
| zcat5 | nsga2-hvref-farthest vs gces | 14/0/7 | 19/0/2 | 14/5/2 |
| zcat5 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-farthest vs nsgaii | 19/0/2 | 21/0/0 | 19/2/0 |
| zcat6 | nsga2-farthest vs gces-noGeo | 9/0/12 | 20/0/1 | 9/11/1 |
| zcat6 | nsga2-farthest vs gces | 12/0/9 | 19/0/2 | 12/7/2 |
| zcat6 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-refcover-farthest vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat7 | nsga2-farthest vs gces-noGeo | 7/0/14 | 6/0/15 | 5/3/13 |
| zcat7 | nsga2-farthest vs gces | 9/0/12 | 11/0/10 | 9/2/10 |
| zcat7 | nsga2-hvfarthest vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat7 | nsga2-hvfarthest vs nsga2-farthest | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat7 | nsga2-hvfarthest vs gces-noGeo | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat7 | nsga2-hvfarthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat7 | nsga2-refcover-farthest vs nsgaii | 3/0/18 | 4/0/17 | 2/3/16 |
| zcat7 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat7 | nsga2-refcover-farthest vs gces | 0/0/21 | 2/0/19 | 0/2/19 |
| zcat7 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-hvref-farthest vs nsga2-farthest | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat7 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-hvref-farthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat7 | nsga2-hvref-farthest vs nsga2-hvfarthest | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat7 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-farthest vs gces-noGeo | 4/0/17 | 5/0/16 | 1/7/13 |
| zcat8 | nsga2-farthest vs gces | 3/0/18 | 12/0/9 | 3/9/9 |
| zcat8 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-refcover-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-refcover-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-refcover-farthest vs gces-noGeo | 17/0/4 | 19/0/2 | 17/2/2 |
| zcat8 | nsga2-refcover-farthest vs gces | 18/0/3 | 19/0/2 | 17/3/1 |
| zcat8 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs nsga2-hvfarthest | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat8 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-farthest vs gces-noGeo | 4/0/17 | 6/0/15 | 2/6/13 |
| zcat9 | nsga2-farthest vs gces | 3/0/18 | 12/0/9 | 3/9/9 |
| zcat9 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-refcover-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-refcover-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-refcover-farthest vs gces-noGeo | 17/0/4 | 19/0/2 | 17/2/2 |
| zcat9 | nsga2-refcover-farthest vs gces | 18/0/3 | 19/0/2 | 17/3/1 |
| zcat9 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs nsga2-hvfarthest | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat9 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-farthest vs nsgaii | 14/0/7 | 13/0/8 | 12/3/6 |
| zcat10 | nsga2-farthest vs gces-noGeo | 1/0/20 | 13/0/8 | 1/12/8 |
| zcat10 | nsga2-farthest vs gces | 2/0/19 | 12/0/9 | 2/10/9 |
| zcat10 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-refcover-farthest vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat11 | nsga2-farthest vs gces-noGeo | 19/0/2 | 18/0/3 | 18/1/2 |
| zcat11 | nsga2-farthest vs gces | 18/0/3 | 17/0/4 | 17/1/3 |
| zcat11 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-refcover-farthest vs nsgaii | 4/0/17 | 5/0/16 | 4/1/16 |
| zcat11 | nsga2-refcover-farthest vs nsga2-farthest | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat11 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | nsga2-farthest vs gces-noGeo | 19/0/2 | 18/0/3 | 18/1/2 |
| zcat12 | nsga2-farthest vs gces | 19/0/2 | 19/0/2 | 18/2/1 |
| zcat12 | nsga2-hvfarthest vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat12 | nsga2-hvfarthest vs nsga2-farthest | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat12 | nsga2-hvfarthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | nsga2-hvfarthest vs gces | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat12 | nsga2-refcover-farthest vs nsgaii | 5/0/16 | 4/0/17 | 4/1/16 |
| zcat12 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvref-farthest vs nsga2-farthest | 17/0/4 | 18/0/3 | 17/1/3 |
| zcat12 | nsga2-hvref-farthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat12 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvref-farthest vs nsga2-hvfarthest | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat12 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-farthest vs gces-noGeo | 17/0/4 | 8/0/13 | 8/9/4 |
| zcat13 | nsga2-farthest vs gces | 16/0/5 | 11/0/10 | 11/5/5 |
| zcat13 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-refcover-farthest vs nsgaii | 6/0/15 | 7/0/14 | 5/3/13 |
| zcat13 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvref-farthest vs nsga2-farthest | 16/0/5 | 15/0/6 | 15/1/5 |
| zcat13 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 15/0/6 | 15/6/0 |
| zcat13 | nsga2-hvref-farthest vs gces | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat13 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-farthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat14 | nsga2-farthest vs gces | 18/0/3 | 18/0/3 | 17/2/2 |
| zcat14 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-refcover-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-refcover-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-refcover-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-refcover-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs nsga2-hvfarthest | 5/0/16 | 7/0/14 | 4/4/13 |
| zcat14 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | nsga2-farthest vs gces-noGeo | 19/0/2 | 18/0/3 | 18/1/2 |
| zcat15 | nsga2-farthest vs gces | 19/0/2 | 19/0/2 | 18/2/1 |
| zcat15 | nsga2-hvfarthest vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat15 | nsga2-hvfarthest vs nsga2-farthest | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat15 | nsga2-hvfarthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | nsga2-hvfarthest vs gces | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat15 | nsga2-refcover-farthest vs nsgaii | 5/0/16 | 4/0/17 | 4/1/16 |
| zcat15 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-hvref-farthest vs nsga2-farthest | 17/0/4 | 18/0/3 | 17/1/3 |
| zcat15 | nsga2-hvref-farthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat15 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-hvref-farthest vs nsga2-hvfarthest | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat15 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat16 | nsga2-farthest vs gces-noGeo | 5/0/16 | 3/0/18 | 3/2/16 |
| zcat16 | nsga2-farthest vs gces | 4/0/17 | 5/0/16 | 3/3/15 |
| zcat16 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-refcover-farthest vs nsgaii | 8/0/13 | 8/0/13 | 8/0/13 |
| zcat16 | nsga2-refcover-farthest vs nsga2-farthest | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat16 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-refcover-farthest vs gces | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat16 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-farthest vs gces-noGeo | 10/0/11 | 19/0/2 | 10/9/2 |
| zcat17 | nsga2-farthest vs gces | 12/0/9 | 15/0/6 | 12/3/6 |
| zcat17 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-refcover-farthest vs nsgaii | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-refcover-farthest vs gces | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvref-farthest vs nsga2-farthest | 14/0/7 | 13/0/8 | 12/3/6 |
| zcat17 | nsga2-hvref-farthest vs gces-noGeo | 13/0/8 | 20/0/1 | 13/7/1 |
| zcat17 | nsga2-hvref-farthest vs gces | 14/0/7 | 19/0/2 | 14/5/2 |
| zcat17 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat18 | nsga2-farthest vs gces-noGeo | 7/0/14 | 6/0/15 | 5/3/13 |
| zcat18 | nsga2-farthest vs gces | 9/0/12 | 11/0/10 | 9/2/10 |
| zcat18 | nsga2-hvfarthest vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat18 | nsga2-hvfarthest vs nsga2-farthest | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat18 | nsga2-hvfarthest vs gces-noGeo | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat18 | nsga2-hvfarthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat18 | nsga2-refcover-farthest vs nsgaii | 3/0/18 | 4/0/17 | 2/3/16 |
| zcat18 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat18 | nsga2-refcover-farthest vs gces | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat18 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs nsga2-farthest | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat18 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat18 | nsga2-hvref-farthest vs nsga2-hvfarthest | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat18 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-farthest vs gces | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat19 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-refcover-farthest vs nsgaii | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat19 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-refcover-farthest vs gces-noGeo | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat19 | nsga2-refcover-farthest vs gces | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat19 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat20 | nsga2-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat20 | nsga2-farthest vs gces-noGeo | 7/0/14 | 6/0/15 | 5/3/13 |
| zcat20 | nsga2-farthest vs gces | 9/0/12 | 11/0/10 | 9/2/10 |
| zcat20 | nsga2-hvfarthest vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat20 | nsga2-hvfarthest vs nsga2-farthest | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat20 | nsga2-hvfarthest vs gces-noGeo | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat20 | nsga2-hvfarthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat20 | nsga2-refcover-farthest vs nsgaii | 3/0/18 | 3/0/18 | 1/4/16 |
| zcat20 | nsga2-refcover-farthest vs nsga2-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat20 | nsga2-refcover-farthest vs gces-noGeo | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat20 | nsga2-refcover-farthest vs gces | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat20 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat20 | nsga2-hvref-farthest vs nsga2-farthest | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat20 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat20 | nsga2-hvref-farthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat20 | nsga2-hvref-farthest vs nsga2-hvfarthest | 3/0/18 | 3/0/18 | 3/0/18 |
| zcat20 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 21/0/0 | 21/0/0 | 21/0/0 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0.004651 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-farthest vs gces-noGeo | 0.002818 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat1 | nsga2-farthest vs gces | 0.002922 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs nsgaii | 0.007284 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 0.002633 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 0.005451 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs gces | 0.005555 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-refcover-farthest vs nsgaii | -0.001221 | 5/0/16 | 0.012691 | 0.418819 | no |
| zcat1 | nsga2-refcover-farthest vs nsga2-farthest | -0.005872 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-refcover-farthest vs gces-noGeo | -0.003054 | 4/0/17 | 0.000161 | 0.013861 | yes |
| zcat1 | nsga2-refcover-farthest vs gces | -0.002950 | 3/0/18 | 0.000105 | 0.009651 | yes |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0.006121 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0.001470 | 18/0/3 | 0.000052 | 0.004930 | yes |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0.004288 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs gces | 0.004392 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001163 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.007342 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-farthest vs nsgaii | 0.002977 | 14/0/7 | 0.075980 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces-noGeo | -0.007088 | 1/0/20 | 0.000010 | 0.000963 | yes |
| zcat10 | nsga2-farthest vs gces | -0.006430 | 2/0/19 | 0.000005 | 0.000496 | yes |
| zcat10 | nsga2-hvfarthest vs nsgaii | 0.024210 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvfarthest vs nsga2-farthest | 0.021233 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvfarthest vs gces-noGeo | 0.014145 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvfarthest vs gces | 0.014803 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs nsgaii | -0.441172 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs nsga2-farthest | -0.444150 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs gces-noGeo | -0.451237 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs gces | -0.450579 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsgaii | 0.018810 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-farthest | 0.015833 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs gces-noGeo | 0.008745 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs gces | 0.009403 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.005400 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.459982 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-farthest vs nsgaii | 0.002806 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat11 | nsga2-farthest vs gces-noGeo | 0.000898 | 19/0/2 | 0.001002 | 0.058134 | no |
| zcat11 | nsga2-farthest vs gces | 0.000804 | 18/0/3 | 0.000721 | 0.043259 | yes |
| zcat11 | nsga2-hvfarthest vs nsgaii | 0.004985 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvfarthest vs nsga2-farthest | 0.002179 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvfarthest vs gces-noGeo | 0.003077 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvfarthest vs gces | 0.002982 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-refcover-farthest vs nsgaii | -0.000638 | 4/0/17 | 0.000721 | 0.043259 | yes |
| zcat11 | nsga2-refcover-farthest vs nsga2-farthest | -0.003444 | 1/0/20 | 0.000426 | 0.033677 | yes |
| zcat11 | nsga2-refcover-farthest vs gces-noGeo | -0.002547 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-refcover-farthest vs gces | -0.002641 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsgaii | 0.003747 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-farthest | 0.000941 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs gces-noGeo | 0.001839 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs gces | 0.001745 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001238 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.004386 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-farthest vs nsgaii | 0.003119 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat12 | nsga2-farthest vs gces-noGeo | 0.001059 | 19/0/2 | 0.000426 | 0.033677 | yes |
| zcat12 | nsga2-farthest vs gces | 0.001030 | 19/0/2 | 0.001002 | 0.058134 | no |
| zcat12 | nsga2-hvfarthest vs nsgaii | 0.005041 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat12 | nsga2-hvfarthest vs nsga2-farthest | 0.001922 | 19/0/2 | 0.006281 | 0.282640 | no |
| zcat12 | nsga2-hvfarthest vs gces-noGeo | 0.002981 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat12 | nsga2-hvfarthest vs gces | 0.002952 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat12 | nsga2-refcover-farthest vs nsgaii | -0.000919 | 5/0/16 | 0.001176 | 0.064673 | no |
| zcat12 | nsga2-refcover-farthest vs nsga2-farthest | -0.004038 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-refcover-farthest vs gces-noGeo | -0.002979 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-refcover-farthest vs gces | -0.003008 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs nsgaii | 0.003852 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-farthest | 0.000733 | 17/0/4 | 0.000041 | 0.003937 | yes |
| zcat12 | nsga2-hvref-farthest vs gces-noGeo | 0.001792 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs gces | 0.001763 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001189 | 2/0/19 | 0.008010 | 0.344426 | no |
| zcat12 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.004771 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-farthest vs nsgaii | 0.002037 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-farthest vs gces-noGeo | 0.000355 | 17/0/4 | 0.001859 | 0.091077 | no |
| zcat13 | nsga2-farthest vs gces | 0.000453 | 16/0/5 | 0.012691 | 0.418819 | no |
| zcat13 | nsga2-hvfarthest vs nsgaii | 0.003731 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvfarthest vs nsga2-farthest | 0.001694 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvfarthest vs gces-noGeo | 0.002049 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvfarthest vs gces | 0.002146 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-refcover-farthest vs nsgaii | -0.000551 | 6/0/15 | 0.010125 | 0.344426 | no |
| zcat13 | nsga2-refcover-farthest vs nsga2-farthest | -0.002588 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-refcover-farthest vs gces-noGeo | -0.002233 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-refcover-farthest vs gces | -0.002135 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsgaii | 0.002413 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-farthest | 0.000376 | 16/0/5 | 0.002151 | 0.103271 | no |
| zcat13 | nsga2-hvref-farthest vs gces-noGeo | 0.000731 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs gces | 0.000829 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001318 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.002964 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-farthest vs nsgaii | 0.005864 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-farthest vs gces-noGeo | 0.002030 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat14 | nsga2-farthest vs gces | 0.001798 | 18/0/3 | 0.000067 | 0.006208 | yes |
| zcat14 | nsga2-hvfarthest vs nsgaii | 0.010489 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvfarthest vs nsga2-farthest | 0.004625 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvfarthest vs gces-noGeo | 0.006655 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvfarthest vs gces | 0.006423 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs nsgaii | 0.009225 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs nsga2-farthest | 0.003361 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs gces-noGeo | 0.005391 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs gces | 0.005159 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs nsgaii | 0.010215 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-farthest | 0.004350 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs gces-noGeo | 0.006380 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs gces | 0.006148 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.000275 | 5/0/16 | 0.004879 | 0.224434 | no |
| zcat14 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.000989 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-farthest vs nsgaii | 0.003119 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat15 | nsga2-farthest vs gces-noGeo | 0.001059 | 19/0/2 | 0.000426 | 0.033677 | yes |
| zcat15 | nsga2-farthest vs gces | 0.001030 | 19/0/2 | 0.001002 | 0.058134 | no |
| zcat15 | nsga2-hvfarthest vs nsgaii | 0.005041 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat15 | nsga2-hvfarthest vs nsga2-farthest | 0.001922 | 19/0/2 | 0.006281 | 0.282640 | no |
| zcat15 | nsga2-hvfarthest vs gces-noGeo | 0.002981 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat15 | nsga2-hvfarthest vs gces | 0.002952 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat15 | nsga2-refcover-farthest vs nsgaii | -0.000919 | 5/0/16 | 0.001176 | 0.064673 | no |
| zcat15 | nsga2-refcover-farthest vs nsga2-farthest | -0.004038 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-refcover-farthest vs gces-noGeo | -0.002979 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-refcover-farthest vs gces | -0.003008 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs nsgaii | 0.003852 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-farthest | 0.000733 | 17/0/4 | 0.000041 | 0.003937 | yes |
| zcat15 | nsga2-hvref-farthest vs gces-noGeo | 0.001792 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs gces | 0.001763 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001189 | 2/0/19 | 0.008010 | 0.344426 | no |
| zcat15 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.004772 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-farthest vs nsgaii | 0.002088 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-farthest vs gces-noGeo | -0.001091 | 5/0/16 | 0.001374 | 0.072835 | no |
| zcat16 | nsga2-farthest vs gces | -0.001129 | 4/0/17 | 0.000293 | 0.023422 | yes |
| zcat16 | nsga2-hvfarthest vs nsgaii | 0.005923 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvfarthest vs nsga2-farthest | 0.003835 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvfarthest vs gces-noGeo | 0.002744 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvfarthest vs gces | 0.002706 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-refcover-farthest vs nsgaii | -0.001978 | 8/0/13 | 0.054693 | 1.000000 | no |
| zcat16 | nsga2-refcover-farthest vs nsga2-farthest | -0.004066 | 1/0/20 | 0.000002 | 0.000324 | yes |
| zcat16 | nsga2-refcover-farthest vs gces-noGeo | -0.005157 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-refcover-farthest vs gces | -0.005195 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsgaii | 0.005084 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-farthest | 0.002996 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs gces-noGeo | 0.001905 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs gces | 0.001867 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.000839 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.007062 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-farthest vs nsgaii | 0.001181 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-farthest vs gces-noGeo | -0.000001 | 10/0/11 | 0.785365 | 1.000000 | no |
| zcat17 | nsga2-farthest vs gces | 0.000002 | 12/0/9 | 0.838194 | 1.000000 | no |
| zcat17 | nsga2-hvfarthest vs nsgaii | 0.002853 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvfarthest vs nsga2-farthest | 0.001672 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvfarthest vs gces-noGeo | 0.001671 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvfarthest vs gces | 0.001674 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs nsgaii | -0.001571 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs nsga2-farthest | -0.002752 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs gces-noGeo | -0.002753 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs gces | -0.002750 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs nsgaii | 0.001308 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-farthest | 0.000127 | 14/0/7 | 0.050192 | 1.000000 | no |
| zcat17 | nsga2-hvref-farthest vs gces-noGeo | 0.000126 | 13/0/8 | 0.026331 | 0.763597 | no |
| zcat17 | nsga2-hvref-farthest vs gces | 0.000129 | 14/0/7 | 0.128078 | 1.000000 | no |
| zcat17 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001545 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.002879 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-farthest vs nsgaii | 0.001903 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-farthest vs gces-noGeo | -0.000451 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces | -0.000054 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat18 | nsga2-hvfarthest vs nsgaii | 0.005017 | 19/0/2 | 0.000510 | 0.034695 | yes |
| zcat18 | nsga2-hvfarthest vs nsga2-farthest | 0.003114 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat18 | nsga2-hvfarthest vs gces-noGeo | 0.002663 | 19/0/2 | 0.000510 | 0.034695 | yes |
| zcat18 | nsga2-hvfarthest vs gces | 0.003060 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat18 | nsga2-refcover-farthest vs nsgaii | -0.000733 | 3/0/18 | 0.000197 | 0.016385 | yes |
| zcat18 | nsga2-refcover-farthest vs nsga2-farthest | -0.002636 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-refcover-farthest vs gces-noGeo | -0.003087 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-refcover-farthest vs gces | -0.002690 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs nsgaii | 0.003316 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-farthest | 0.001413 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs gces-noGeo | 0.000962 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs gces | 0.001359 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001701 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat18 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.004049 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-farthest vs nsgaii | 0.003499 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-farthest vs gces-noGeo | 0.002066 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-farthest vs gces | 0.002023 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs nsgaii | 0.006622 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs nsga2-farthest | 0.003123 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs gces-noGeo | 0.005189 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs gces | 0.005146 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-refcover-farthest vs nsgaii | -0.001193 | 3/0/18 | 0.001374 | 0.072835 | no |
| zcat19 | nsga2-refcover-farthest vs nsga2-farthest | -0.004692 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-refcover-farthest vs gces-noGeo | -0.002626 | 2/0/19 | 0.000010 | 0.000963 | yes |
| zcat19 | nsga2-refcover-farthest vs gces | -0.002669 | 1/0/20 | 0.000002 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsgaii | 0.005078 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-farthest | 0.001580 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs gces-noGeo | 0.003646 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs gces | 0.003603 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001543 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.006272 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-farthest vs nsgaii | 0.001482 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-farthest vs gces-noGeo | 0.000708 | 19/0/2 | 0.000105 | 0.009651 | yes |
| zcat2 | nsga2-farthest vs gces | 0.000430 | 16/0/5 | 0.002482 | 0.116673 | no |
| zcat2 | nsga2-hvfarthest vs nsgaii | 0.003644 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvfarthest vs nsga2-farthest | 0.002162 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvfarthest vs gces-noGeo | 0.002870 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvfarthest vs gces | 0.002592 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs nsgaii | -0.002838 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs nsga2-farthest | -0.004320 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs gces-noGeo | -0.003612 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs gces | -0.003890 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvref-farthest vs nsgaii | 0.000671 | 18/0/3 | 0.000013 | 0.001322 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-farthest | -0.000811 | 3/0/18 | 0.000013 | 0.001322 | yes |
| zcat2 | nsga2-hvref-farthest vs gces-noGeo | -0.000103 | 7/0/14 | 0.064650 | 1.000000 | no |
| zcat2 | nsga2-hvref-farthest vs gces | -0.000381 | 6/0/15 | 0.042080 | 1.000000 | no |
| zcat2 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.002973 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.003509 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-farthest vs nsgaii | 0.001904 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-farthest vs gces-noGeo | -0.000452 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces | -0.000054 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat20 | nsga2-hvfarthest vs nsgaii | 0.005019 | 19/0/2 | 0.000510 | 0.034695 | yes |
| zcat20 | nsga2-hvfarthest vs nsga2-farthest | 0.003115 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat20 | nsga2-hvfarthest vs gces-noGeo | 0.002663 | 19/0/2 | 0.000510 | 0.034695 | yes |
| zcat20 | nsga2-hvfarthest vs gces | 0.003061 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat20 | nsga2-refcover-farthest vs nsgaii | -0.000732 | 3/0/18 | 0.000197 | 0.016385 | yes |
| zcat20 | nsga2-refcover-farthest vs nsga2-farthest | -0.002636 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-refcover-farthest vs gces-noGeo | -0.003088 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-refcover-farthest vs gces | -0.002690 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs nsgaii | 0.003317 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-farthest | 0.001414 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs gces-noGeo | 0.000962 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs gces | 0.001359 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001702 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat20 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.004050 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-farthest vs nsgaii | 0.004944 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-farthest vs gces-noGeo | 0.002644 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-farthest vs gces | 0.002732 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs nsgaii | 0.007463 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs nsga2-farthest | 0.002519 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs gces-noGeo | 0.005162 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs gces | 0.005251 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-refcover-farthest vs nsgaii | 0.000584 | 14/0/7 | 0.516761 | 1.000000 | no |
| zcat3 | nsga2-refcover-farthest vs nsga2-farthest | -0.004360 | 1/0/20 | 0.000005 | 0.000496 | yes |
| zcat3 | nsga2-refcover-farthest vs gces-noGeo | -0.001716 | 8/0/13 | 0.031919 | 0.861826 | no |
| zcat3 | nsga2-refcover-farthest vs gces | -0.001628 | 6/0/15 | 0.015780 | 0.489194 | no |
| zcat3 | nsga2-hvref-farthest vs nsgaii | 0.006198 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-farthest | 0.001254 | 18/0/3 | 0.000105 | 0.009651 | yes |
| zcat3 | nsga2-hvref-farthest vs gces-noGeo | 0.003897 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs gces | 0.003986 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001265 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.005614 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-farthest vs nsgaii | 0.004944 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-farthest vs gces-noGeo | 0.002644 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-farthest vs gces | 0.002733 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs nsgaii | 0.007463 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs nsga2-farthest | 0.002518 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs gces-noGeo | 0.005162 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs gces | 0.005251 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-refcover-farthest vs nsgaii | 0.000584 | 14/0/7 | 0.516761 | 1.000000 | no |
| zcat4 | nsga2-refcover-farthest vs nsga2-farthest | -0.004360 | 1/0/20 | 0.000005 | 0.000496 | yes |
| zcat4 | nsga2-refcover-farthest vs gces-noGeo | -0.001717 | 8/0/13 | 0.031919 | 0.861826 | no |
| zcat4 | nsga2-refcover-farthest vs gces | -0.001627 | 6/0/15 | 0.015780 | 0.489194 | no |
| zcat4 | nsga2-hvref-farthest vs nsgaii | 0.006198 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-farthest | 0.001254 | 18/0/3 | 0.000105 | 0.009651 | yes |
| zcat4 | nsga2-hvref-farthest vs gces-noGeo | 0.003897 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs gces | 0.003987 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001265 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.005614 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-farthest vs nsgaii | 0.001180 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-farthest vs gces-noGeo | -0.000001 | 10/0/11 | 0.759288 | 1.000000 | no |
| zcat5 | nsga2-farthest vs gces | 0.000002 | 12/0/9 | 0.811678 | 1.000000 | no |
| zcat5 | nsga2-hvfarthest vs nsgaii | 0.002852 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvfarthest vs nsga2-farthest | 0.001672 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvfarthest vs gces-noGeo | 0.001671 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvfarthest vs gces | 0.001674 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs nsgaii | -0.001571 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs nsga2-farthest | -0.002751 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs gces-noGeo | -0.002752 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs gces | -0.002748 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs nsgaii | 0.001307 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-farthest | 0.000127 | 14/0/7 | 0.050192 | 1.000000 | no |
| zcat5 | nsga2-hvref-farthest vs gces-noGeo | 0.000126 | 13/0/8 | 0.029015 | 0.812408 | no |
| zcat5 | nsga2-hvref-farthest vs gces | 0.000130 | 14/0/7 | 0.128078 | 1.000000 | no |
| zcat5 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001545 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.002878 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-farthest vs nsgaii | 0.000593 | 19/0/2 | 0.000013 | 0.001322 | yes |
| zcat6 | nsga2-farthest vs gces-noGeo | -0.000174 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat6 | nsga2-farthest vs gces | -0.000113 | 12/0/9 | 0.682714 | 1.000000 | no |
| zcat6 | nsga2-hvfarthest vs nsgaii | 0.002357 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvfarthest vs nsga2-farthest | 0.001764 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvfarthest vs gces-noGeo | 0.001590 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvfarthest vs gces | 0.001652 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs nsgaii | -0.005222 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs nsga2-farthest | -0.005815 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs gces-noGeo | -0.005989 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs gces | -0.005928 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsgaii | 0.001731 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-farthest | 0.001138 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs gces-noGeo | 0.000964 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs gces | 0.001025 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.000626 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.006953 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-farthest vs nsgaii | 0.001903 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-farthest vs gces-noGeo | -0.000451 | 7/0/14 | 0.119342 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces | -0.000054 | 9/0/12 | 0.682714 | 1.000000 | no |
| zcat7 | nsga2-hvfarthest vs nsgaii | 0.005017 | 19/0/2 | 0.000510 | 0.034695 | yes |
| zcat7 | nsga2-hvfarthest vs nsga2-farthest | 0.003114 | 19/0/2 | 0.008010 | 0.344426 | no |
| zcat7 | nsga2-hvfarthest vs gces-noGeo | 0.002663 | 19/0/2 | 0.000510 | 0.034695 | yes |
| zcat7 | nsga2-hvfarthest vs gces | 0.003060 | 20/0/1 | 0.000426 | 0.033677 | yes |
| zcat7 | nsga2-refcover-farthest vs nsgaii | -0.000733 | 3/0/18 | 0.000197 | 0.016385 | yes |
| zcat7 | nsga2-refcover-farthest vs nsga2-farthest | -0.002636 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-refcover-farthest vs gces-noGeo | -0.003087 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-refcover-farthest vs gces | -0.002690 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs nsgaii | 0.003316 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-farthest | 0.001413 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs gces-noGeo | 0.000962 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs gces | 0.001359 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001701 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat7 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.004049 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-farthest vs nsgaii | 0.006375 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-farthest vs gces-noGeo | -0.001291 | 4/0/17 | 0.000105 | 0.009651 | yes |
| zcat8 | nsga2-farthest vs gces | -0.001760 | 3/0/18 | 0.000161 | 0.013861 | yes |
| zcat8 | nsga2-hvfarthest vs nsgaii | 0.014265 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvfarthest vs nsga2-farthest | 0.007890 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvfarthest vs gces-noGeo | 0.006599 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvfarthest vs gces | 0.006130 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-refcover-farthest vs nsgaii | 0.009182 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-refcover-farthest vs nsga2-farthest | 0.002807 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-refcover-farthest vs gces-noGeo | 0.001516 | 17/0/4 | 0.001374 | 0.072835 | no |
| zcat8 | nsga2-refcover-farthest vs gces | 0.001047 | 18/0/3 | 0.000510 | 0.034695 | yes |
| zcat8 | nsga2-hvref-farthest vs nsgaii | 0.012928 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-farthest | 0.006554 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs gces-noGeo | 0.005263 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs gces | 0.004794 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001336 | 1/0/20 | 0.000003 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.003747 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-farthest vs nsgaii | 0.006375 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-farthest vs gces-noGeo | -0.001291 | 4/0/17 | 0.000105 | 0.009651 | yes |
| zcat9 | nsga2-farthest vs gces | -0.001760 | 3/0/18 | 0.000161 | 0.013861 | yes |
| zcat9 | nsga2-hvfarthest vs nsgaii | 0.014265 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvfarthest vs nsga2-farthest | 0.007890 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvfarthest vs gces-noGeo | 0.006599 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvfarthest vs gces | 0.006130 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-refcover-farthest vs nsgaii | 0.009182 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-refcover-farthest vs nsga2-farthest | 0.002807 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-refcover-farthest vs gces-noGeo | 0.001516 | 17/0/4 | 0.001374 | 0.072835 | no |
| zcat9 | nsga2-refcover-farthest vs gces | 0.001047 | 18/0/3 | 0.000510 | 0.034695 | yes |
| zcat9 | nsga2-hvref-farthest vs nsgaii | 0.012928 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-farthest | 0.006554 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs gces-noGeo | 0.005263 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs gces | 0.004794 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.001336 | 1/0/20 | 0.000003 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.003747 | 21/0/0 | 0.000001 | 0.000324 | yes |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | -0.001903 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-farthest vs gces-noGeo | -0.001417 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-farthest vs gces | -0.001285 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs nsgaii | -0.003009 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | -0.001106 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | -0.002522 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvfarthest vs gces | -0.002391 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-refcover-farthest vs nsgaii | 0.000566 | 5/0/16 | 0.010125 | 0.333750 | no |
| zcat1 | nsga2-refcover-farthest vs nsga2-farthest | 0.002469 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-refcover-farthest vs gces-noGeo | 0.001053 | 5/0/16 | 0.001002 | 0.060139 | no |
| zcat1 | nsga2-refcover-farthest vs gces | 0.001184 | 2/0/19 | 0.000241 | 0.018820 | yes |
| zcat1 | nsga2-hvref-farthest vs nsgaii | -0.002475 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | -0.000571 | 18/0/3 | 0.000084 | 0.007050 | yes |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | -0.001988 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs gces | -0.001856 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000534 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.003040 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-farthest vs nsgaii | -0.000146 | 13/0/8 | 0.103214 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces-noGeo | -0.000090 | 13/0/8 | 0.272210 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces | -0.000041 | 12/0/9 | 0.452368 | 1.000000 | no |
| zcat10 | nsga2-hvfarthest vs nsgaii | -0.001455 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvfarthest vs nsga2-farthest | -0.001309 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvfarthest vs gces-noGeo | -0.001399 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvfarthest vs gces | -0.001350 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs nsgaii | 0.025783 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs nsga2-farthest | 0.025929 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs gces-noGeo | 0.025839 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-refcover-farthest vs gces | 0.025888 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsgaii | -0.001102 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-farthest | -0.000956 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs gces-noGeo | -0.001046 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs gces | -0.000997 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000353 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.026884 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-farthest vs nsgaii | -0.001253 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat11 | nsga2-farthest vs gces-noGeo | -0.000317 | 18/0/3 | 0.002857 | 0.157146 | no |
| zcat11 | nsga2-farthest vs gces | -0.000291 | 17/0/4 | 0.003753 | 0.202646 | no |
| zcat11 | nsga2-hvfarthest vs nsgaii | -0.002233 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvfarthest vs nsga2-farthest | -0.000980 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvfarthest vs gces-noGeo | -0.001297 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvfarthest vs gces | -0.001271 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-refcover-farthest vs nsgaii | 0.000383 | 5/0/16 | 0.001374 | 0.076958 | no |
| zcat11 | nsga2-refcover-farthest vs nsga2-farthest | 0.001635 | 1/0/20 | 0.000426 | 0.031119 | yes |
| zcat11 | nsga2-refcover-farthest vs gces-noGeo | 0.001318 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-refcover-farthest vs gces | 0.001345 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsgaii | -0.001660 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-farthest | -0.000407 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs gces-noGeo | -0.000724 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs gces | -0.000697 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000573 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002042 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-farthest vs nsgaii | -0.001476 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat12 | nsga2-farthest vs gces-noGeo | -0.000408 | 18/0/3 | 0.001002 | 0.060139 | no |
| zcat12 | nsga2-farthest vs gces | -0.000412 | 19/0/2 | 0.000852 | 0.052801 | no |
| zcat12 | nsga2-hvfarthest vs nsgaii | -0.002362 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat12 | nsga2-hvfarthest vs nsga2-farthest | -0.000886 | 19/0/2 | 0.006281 | 0.307764 | no |
| zcat12 | nsga2-hvfarthest vs gces-noGeo | -0.001294 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat12 | nsga2-hvfarthest vs gces | -0.001298 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat12 | nsga2-refcover-farthest vs nsgaii | 0.000409 | 4/0/17 | 0.000293 | 0.022544 | yes |
| zcat12 | nsga2-refcover-farthest vs nsga2-farthest | 0.001885 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-refcover-farthest vs gces-noGeo | 0.001477 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-refcover-farthest vs gces | 0.001474 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs nsgaii | -0.001833 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-farthest | -0.000357 | 18/0/3 | 0.000024 | 0.002193 | yes |
| zcat12 | nsga2-hvref-farthest vs gces-noGeo | -0.000765 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs gces | -0.000769 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000529 | 2/0/19 | 0.008010 | 0.333750 | no |
| zcat12 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002242 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-farthest vs nsgaii | -0.000892 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-farthest vs gces-noGeo | 0.000056 | 8/0/13 | 0.759288 | 1.000000 | no |
| zcat13 | nsga2-farthest vs gces | -0.000060 | 11/0/10 | 0.473334 | 1.000000 | no |
| zcat13 | nsga2-hvfarthest vs nsgaii | -0.001693 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvfarthest vs nsga2-farthest | -0.000801 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvfarthest vs gces-noGeo | -0.000745 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvfarthest vs gces | -0.000861 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-refcover-farthest vs nsgaii | 0.000259 | 7/0/14 | 0.137283 | 1.000000 | no |
| zcat13 | nsga2-refcover-farthest vs nsga2-farthest | 0.001151 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-refcover-farthest vs gces-noGeo | 0.001207 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-refcover-farthest vs gces | 0.001091 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsgaii | -0.001092 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-farthest | -0.000200 | 15/0/6 | 0.004285 | 0.222813 | no |
| zcat13 | nsga2-hvref-farthest vs gces-noGeo | -0.000144 | 15/0/6 | 0.003753 | 0.202646 | no |
| zcat13 | nsga2-hvref-farthest vs gces | -0.000260 | 18/0/3 | 0.000052 | 0.004616 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000601 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.001351 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-farthest vs nsgaii | -0.001693 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-farthest vs gces-noGeo | -0.000721 | 20/0/1 | 0.000007 | 0.000701 | yes |
| zcat14 | nsga2-farthest vs gces | -0.000561 | 18/0/3 | 0.000041 | 0.003650 | yes |
| zcat14 | nsga2-hvfarthest vs nsgaii | -0.002988 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvfarthest vs nsga2-farthest | -0.001295 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvfarthest vs gces-noGeo | -0.002016 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvfarthest vs gces | -0.001857 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs nsgaii | -0.002645 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs nsga2-farthest | -0.000951 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs gces-noGeo | -0.001672 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-refcover-farthest vs gces | -0.001513 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs nsgaii | -0.002941 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-farthest | -0.001248 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs gces-noGeo | -0.001968 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs gces | -0.001809 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000048 | 7/0/14 | 0.029015 | 0.841423 | no |
| zcat14 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.000296 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-farthest vs nsgaii | -0.001511 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat15 | nsga2-farthest vs gces-noGeo | -0.000386 | 18/0/3 | 0.001176 | 0.067025 | no |
| zcat15 | nsga2-farthest vs gces | -0.000391 | 19/0/2 | 0.000852 | 0.052801 | no |
| zcat15 | nsga2-hvfarthest vs nsgaii | -0.002405 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat15 | nsga2-hvfarthest vs nsga2-farthest | -0.000894 | 19/0/2 | 0.006281 | 0.307764 | no |
| zcat15 | nsga2-hvfarthest vs gces-noGeo | -0.001280 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat15 | nsga2-hvfarthest vs gces | -0.001285 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat15 | nsga2-refcover-farthest vs nsgaii | 0.000381 | 4/0/17 | 0.000354 | 0.026536 | yes |
| zcat15 | nsga2-refcover-farthest vs nsga2-farthest | 0.001891 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-refcover-farthest vs gces-noGeo | 0.001506 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-refcover-farthest vs gces | 0.001501 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs nsgaii | -0.001879 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-farthest | -0.000368 | 18/0/3 | 0.000018 | 0.001703 | yes |
| zcat15 | nsga2-hvref-farthest vs gces-noGeo | -0.000754 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs gces | -0.000759 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000526 | 2/0/19 | 0.008010 | 0.333750 | no |
| zcat15 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002260 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-farthest vs nsgaii | -0.000741 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat16 | nsga2-farthest vs gces-noGeo | 0.000417 | 3/0/18 | 0.000354 | 0.026536 | yes |
| zcat16 | nsga2-farthest vs gces | 0.000479 | 5/0/16 | 0.000293 | 0.022544 | yes |
| zcat16 | nsga2-hvfarthest vs nsgaii | -0.002204 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvfarthest vs nsga2-farthest | -0.001462 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvfarthest vs gces-noGeo | -0.001045 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvfarthest vs gces | -0.000983 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-refcover-farthest vs nsgaii | 0.000371 | 8/0/13 | 0.303816 | 1.000000 | no |
| zcat16 | nsga2-refcover-farthest vs nsga2-farthest | 0.001112 | 2/0/19 | 0.001002 | 0.060139 | no |
| zcat16 | nsga2-refcover-farthest vs gces-noGeo | 0.001529 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-refcover-farthest vs gces | 0.001591 | 1/0/20 | 0.000002 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsgaii | -0.001881 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-farthest | -0.001140 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs gces-noGeo | -0.000723 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs gces | -0.000661 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000323 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002252 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-farthest vs nsgaii | -0.000908 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-farthest vs gces-noGeo | -0.000405 | 19/0/2 | 0.000010 | 0.000973 | yes |
| zcat17 | nsga2-farthest vs gces | -0.000281 | 15/0/6 | 0.005542 | 0.277090 | no |
| zcat17 | nsga2-hvfarthest vs nsgaii | -0.002024 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvfarthest vs nsga2-farthest | -0.001117 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvfarthest vs gces-noGeo | -0.001522 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvfarthest vs gces | -0.001398 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs nsgaii | 0.001044 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs nsga2-farthest | 0.001951 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs gces-noGeo | 0.001546 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-refcover-farthest vs gces | 0.001670 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs nsgaii | -0.000992 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-farthest | -0.000085 | 13/0/8 | 0.070137 | 1.000000 | no |
| zcat17 | nsga2-hvref-farthest vs gces-noGeo | -0.000490 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs gces | -0.000366 | 19/0/2 | 0.000010 | 0.000973 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.001032 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002036 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-farthest vs nsgaii | -0.000657 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat18 | nsga2-farthest vs gces-noGeo | 0.000134 | 6/0/15 | 0.242842 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces | -0.000017 | 11/0/10 | 0.411982 | 1.000000 | no |
| zcat18 | nsga2-hvfarthest vs nsgaii | -0.001965 | 19/0/2 | 0.007101 | 0.333750 | no |
| zcat18 | nsga2-hvfarthest vs nsga2-farthest | -0.001308 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat18 | nsga2-hvfarthest vs gces-noGeo | -0.001174 | 19/0/2 | 0.007101 | 0.333750 | no |
| zcat18 | nsga2-hvfarthest vs gces | -0.001325 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat18 | nsga2-refcover-farthest vs nsgaii | 0.000449 | 4/0/17 | 0.000105 | 0.008497 | yes |
| zcat18 | nsga2-refcover-farthest vs nsga2-farthest | 0.001107 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-refcover-farthest vs gces-noGeo | 0.001241 | 1/0/20 | 0.000003 | 0.000329 | yes |
| zcat18 | nsga2-refcover-farthest vs gces | 0.001090 | 1/0/20 | 0.000003 | 0.000329 | yes |
| zcat18 | nsga2-hvref-farthest vs nsgaii | -0.001254 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-farthest | -0.000596 | 20/0/1 | 0.000005 | 0.000539 | yes |
| zcat18 | nsga2-hvref-farthest vs gces-noGeo | -0.000462 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs gces | -0.000613 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000712 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat18 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.001703 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-farthest vs nsgaii | -0.001380 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-farthest vs gces-noGeo | -0.000938 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-farthest vs gces | -0.000916 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs nsgaii | -0.002550 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs nsga2-farthest | -0.001169 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs gces-noGeo | -0.002107 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvfarthest vs gces | -0.002086 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-refcover-farthest vs nsgaii | 0.000632 | 3/0/18 | 0.000426 | 0.031119 | yes |
| zcat19 | nsga2-refcover-farthest vs nsga2-farthest | 0.002013 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-refcover-farthest vs gces-noGeo | 0.001075 | 2/0/19 | 0.000024 | 0.002193 | yes |
| zcat19 | nsga2-refcover-farthest vs gces | 0.001096 | 2/0/19 | 0.000010 | 0.000973 | yes |
| zcat19 | nsga2-hvref-farthest vs nsgaii | -0.002005 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-farthest | -0.000624 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs gces-noGeo | -0.001562 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs gces | -0.001541 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000545 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002637 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-farthest vs nsgaii | -0.001082 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-farthest vs gces-noGeo | -0.000413 | 20/0/1 | 0.000052 | 0.004616 | yes |
| zcat2 | nsga2-farthest vs gces | -0.000352 | 18/0/3 | 0.000067 | 0.005741 | yes |
| zcat2 | nsga2-hvfarthest vs nsgaii | -0.002523 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvfarthest vs nsga2-farthest | -0.001440 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvfarthest vs gces-noGeo | -0.001854 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvfarthest vs gces | -0.001793 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs nsgaii | 0.001948 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs nsga2-farthest | 0.003030 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs gces-noGeo | 0.002617 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-refcover-farthest vs gces | 0.002678 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvref-farthest vs nsgaii | -0.000481 | 18/0/3 | 0.000197 | 0.015595 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-farthest | 0.000602 | 2/0/19 | 0.000010 | 0.000973 | yes |
| zcat2 | nsga2-hvref-farthest vs gces-noGeo | 0.000188 | 7/0/14 | 0.103214 | 1.000000 | no |
| zcat2 | nsga2-hvref-farthest vs gces | 0.000250 | 5/0/16 | 0.023854 | 0.715628 | no |
| zcat2 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.002042 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002429 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-farthest vs nsgaii | -0.000662 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat20 | nsga2-farthest vs gces-noGeo | 0.000143 | 6/0/15 | 0.257248 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces | 0.000001 | 11/0/10 | 0.392584 | 1.000000 | no |
| zcat20 | nsga2-hvfarthest vs nsgaii | -0.001957 | 19/0/2 | 0.007101 | 0.333750 | no |
| zcat20 | nsga2-hvfarthest vs nsga2-farthest | -0.001294 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat20 | nsga2-hvfarthest vs gces-noGeo | -0.001151 | 19/0/2 | 0.007101 | 0.333750 | no |
| zcat20 | nsga2-hvfarthest vs gces | -0.001293 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat20 | nsga2-refcover-farthest vs nsgaii | 0.000459 | 3/0/18 | 0.000067 | 0.005741 | yes |
| zcat20 | nsga2-refcover-farthest vs nsga2-farthest | 0.001122 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-refcover-farthest vs gces-noGeo | 0.001265 | 1/0/20 | 0.000002 | 0.000324 | yes |
| zcat20 | nsga2-refcover-farthest vs gces | 0.001123 | 1/0/20 | 0.000002 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs nsgaii | -0.001264 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-farthest | -0.000601 | 20/0/1 | 0.000005 | 0.000539 | yes |
| zcat20 | nsga2-hvref-farthest vs gces-noGeo | -0.000458 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs gces | -0.000600 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000693 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat20 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.001723 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-farthest vs nsgaii | -0.002098 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-farthest vs gces-noGeo | -0.001341 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-farthest vs gces | -0.001234 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs nsgaii | -0.003144 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs nsga2-farthest | -0.001046 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs gces-noGeo | -0.002387 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvfarthest vs gces | -0.002280 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-refcover-farthest vs nsgaii | -0.000272 | 14/0/7 | 0.633297 | 1.000000 | no |
| zcat3 | nsga2-refcover-farthest vs nsga2-farthest | 0.001826 | 1/0/20 | 0.000005 | 0.000539 | yes |
| zcat3 | nsga2-refcover-farthest vs gces-noGeo | 0.000484 | 8/0/13 | 0.050192 | 1.000000 | no |
| zcat3 | nsga2-refcover-farthest vs gces | 0.000591 | 8/0/13 | 0.038438 | 1.000000 | no |
| zcat3 | nsga2-hvref-farthest vs nsgaii | -0.002607 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-farthest | -0.000509 | 18/0/3 | 0.000105 | 0.008497 | yes |
| zcat3 | nsga2-hvref-farthest vs gces-noGeo | -0.001850 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs gces | -0.001743 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000537 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002334 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-farthest vs nsgaii | -0.002056 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-farthest vs gces-noGeo | -0.001308 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-farthest vs gces | -0.001267 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs nsgaii | -0.003109 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs nsga2-farthest | -0.001054 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs gces-noGeo | -0.002362 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvfarthest vs gces | -0.002321 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-refcover-farthest vs nsgaii | -0.000253 | 14/0/7 | 0.609149 | 1.000000 | no |
| zcat4 | nsga2-refcover-farthest vs nsga2-farthest | 0.001803 | 1/0/20 | 0.000005 | 0.000539 | yes |
| zcat4 | nsga2-refcover-farthest vs gces-noGeo | 0.000495 | 8/0/13 | 0.045993 | 1.000000 | no |
| zcat4 | nsga2-refcover-farthest vs gces | 0.000535 | 7/0/14 | 0.038438 | 1.000000 | no |
| zcat4 | nsga2-hvref-farthest vs nsgaii | -0.002583 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-farthest | -0.000528 | 18/0/3 | 0.000084 | 0.007050 | yes |
| zcat4 | nsga2-hvref-farthest vs gces-noGeo | -0.001836 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs gces | -0.001795 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000526 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002330 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-farthest vs nsgaii | -0.000884 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-farthest vs gces-noGeo | -0.000384 | 19/0/2 | 0.000007 | 0.000701 | yes |
| zcat5 | nsga2-farthest vs gces | -0.000281 | 15/0/6 | 0.004285 | 0.222813 | no |
| zcat5 | nsga2-hvfarthest vs nsgaii | -0.002010 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvfarthest vs nsga2-farthest | -0.001125 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvfarthest vs gces-noGeo | -0.001509 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvfarthest vs gces | -0.001407 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs nsgaii | 0.001049 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs nsga2-farthest | 0.001933 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs gces-noGeo | 0.001549 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-refcover-farthest vs gces | 0.001652 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs nsgaii | -0.000981 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-farthest | -0.000097 | 14/0/7 | 0.054693 | 1.000000 | no |
| zcat5 | nsga2-hvref-farthest vs gces-noGeo | -0.000480 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs gces | -0.000378 | 19/0/2 | 0.000005 | 0.000539 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.001029 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.002030 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-farthest vs nsgaii | -0.000317 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-farthest vs gces-noGeo | -0.000294 | 20/0/1 | 0.000018 | 0.001703 | yes |
| zcat6 | nsga2-farthest vs gces | -0.000323 | 19/0/2 | 0.000024 | 0.002193 | yes |
| zcat6 | nsga2-hvfarthest vs nsgaii | -0.001109 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvfarthest vs nsga2-farthest | -0.000792 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvfarthest vs gces-noGeo | -0.001086 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvfarthest vs gces | -0.001115 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs nsgaii | 0.002385 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs nsga2-farthest | 0.002702 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs gces-noGeo | 0.002408 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-refcover-farthest vs gces | 0.002379 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsgaii | -0.000820 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-farthest | -0.000503 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs gces-noGeo | -0.000797 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs gces | -0.000826 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000289 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.003205 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-farthest vs nsgaii | -0.000665 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat7 | nsga2-farthest vs gces-noGeo | 0.000146 | 6/0/15 | 0.287734 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces | -0.000022 | 11/0/10 | 0.320457 | 1.000000 | no |
| zcat7 | nsga2-hvfarthest vs nsgaii | -0.001969 | 19/0/2 | 0.007101 | 0.333750 | no |
| zcat7 | nsga2-hvfarthest vs nsga2-farthest | -0.001304 | 19/0/2 | 0.008010 | 0.333750 | no |
| zcat7 | nsga2-hvfarthest vs gces-noGeo | -0.001158 | 19/0/2 | 0.007101 | 0.333750 | no |
| zcat7 | nsga2-hvfarthest vs gces | -0.001326 | 20/0/1 | 0.000426 | 0.031119 | yes |
| zcat7 | nsga2-refcover-farthest vs nsgaii | 0.000431 | 4/0/17 | 0.000084 | 0.007050 | yes |
| zcat7 | nsga2-refcover-farthest vs nsga2-farthest | 0.001096 | 0/0/21 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-refcover-farthest vs gces-noGeo | 0.001242 | 1/0/20 | 0.000002 | 0.000324 | yes |
| zcat7 | nsga2-refcover-farthest vs gces | 0.001074 | 2/0/19 | 0.000005 | 0.000539 | yes |
| zcat7 | nsga2-hvref-farthest vs nsgaii | -0.001271 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-farthest | -0.000606 | 20/0/1 | 0.000005 | 0.000539 | yes |
| zcat7 | nsga2-hvref-farthest vs gces-noGeo | -0.000460 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs gces | -0.000628 | 20/0/1 | 0.000002 | 0.000324 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000698 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat7 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.001703 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-farthest vs nsgaii | -0.001283 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-farthest vs gces-noGeo | 0.000241 | 5/0/16 | 0.019473 | 0.603665 | no |
| zcat8 | nsga2-farthest vs gces | -0.000067 | 12/0/9 | 0.864887 | 1.000000 | no |
| zcat8 | nsga2-hvfarthest vs nsgaii | -0.002739 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvfarthest vs nsga2-farthest | -0.001457 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvfarthest vs gces-noGeo | -0.001216 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvfarthest vs gces | -0.001523 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-refcover-farthest vs nsgaii | -0.001882 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-refcover-farthest vs nsga2-farthest | -0.000599 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-refcover-farthest vs gces-noGeo | -0.000358 | 19/0/2 | 0.000010 | 0.000973 | yes |
| zcat8 | nsga2-refcover-farthest vs gces | -0.000666 | 19/0/2 | 0.000007 | 0.000701 | yes |
| zcat8 | nsga2-hvref-farthest vs nsgaii | -0.002507 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-farthest | -0.001225 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs gces-noGeo | -0.000984 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs gces | -0.001291 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000232 | 2/0/19 | 0.000010 | 0.000973 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.000626 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-farthest vs nsgaii | -0.001291 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-farthest vs gces-noGeo | 0.000179 | 6/0/15 | 0.070137 | 1.000000 | no |
| zcat9 | nsga2-farthest vs gces | -0.000065 | 12/0/9 | 0.864887 | 1.000000 | no |
| zcat9 | nsga2-hvfarthest vs nsgaii | -0.002723 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvfarthest vs nsga2-farthest | -0.001431 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvfarthest vs gces-noGeo | -0.001252 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvfarthest vs gces | -0.001496 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-refcover-farthest vs nsgaii | -0.001863 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-refcover-farthest vs nsga2-farthest | -0.000572 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-refcover-farthest vs gces-noGeo | -0.000393 | 19/0/2 | 0.000013 | 0.001268 | yes |
| zcat9 | nsga2-refcover-farthest vs gces | -0.000636 | 19/0/2 | 0.000005 | 0.000539 | yes |
| zcat9 | nsga2-hvref-farthest vs nsgaii | -0.002495 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-farthest | -0.001204 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs gces-noGeo | -0.001025 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs gces | -0.001269 | 21/0/0 | 0.000001 | 0.000324 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000227 | 2/0/19 | 0.000010 | 0.000973 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.000632 | 21/0/0 | 0.000001 | 0.000324 | yes |
