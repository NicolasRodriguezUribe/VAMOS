# Farthest-family 3-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the farthest-derived selectors, nsga2_sector_farthest, and the GCES-family variants.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, nsga2-farthest, nsga2-hvfarthest, nsga2-hvref-farthest, nsga2-sector-farthest, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 3
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-farthest vs nsgaii, gces-noGeo vs nsgaii, gces vs nsgaii, nsga2-farthest vs gces-noGeo, nsga2-farthest vs gces, nsga2-hvfarthest vs nsgaii, nsga2-hvfarthest vs nsga2-farthest, nsga2-hvfarthest vs gces-noGeo, nsga2-hvfarthest vs gces, nsga2-hvref-farthest vs nsgaii, nsga2-hvref-farthest vs nsga2-farthest, nsga2-hvref-farthest vs gces-noGeo, nsga2-hvref-farthest vs gces, nsga2-hvref-farthest vs nsga2-hvfarthest, nsga2-sector-farthest vs nsgaii, nsga2-sector-farthest vs nsga2-farthest, nsga2-sector-farthest vs gces-noGeo, nsga2-sector-farthest vs gces, nsga2-sector-farthest vs nsga2-hvref-farthest

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-hvref-farthest | nsga2-sector-farthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.843857 | 0.908630 | 0.956191 | 0.951775 | 0.907674 | 0.905625 | 0.903931 |
| zcat2 | 0.967643 | 0.981385 | 0.991305 | 0.988295 | 0.980214 | 0.979205 | 0.979673 |
| zcat3 | 0.616807 | 0.758866 | 0.918696 | 0.890654 | 0.774409 | 0.764038 | 0.754387 |
| zcat4 | 0.663852 | 0.771065 | 0.912332 | 0.887765 | 0.794711 | 0.778310 | 0.766102 |
| zcat5 | 0.925084 | 0.950595 | 0.982911 | 0.973709 | 0.957641 | 0.949605 | 0.948556 |
| zcat6 | 0.714138 | 0.809214 | 0.924995 | 0.919430 | 0.823956 | 0.807183 | 0.803477 |
| zcat7 | 0.812443 | 0.873552 | 0.963666 | 0.935834 | 0.886932 | 0.871833 | 0.870513 |
| zcat8 | 0.000000 | 0.169852 | 0.589909 | 0.351575 | 0.182958 | 0.165722 | 0.163794 |
| zcat9 | 0.587150 | 0.683339 | 0.892737 | 0.847880 | 0.662244 | 0.660819 | 0.675360 |
| zcat10 | 0.278869 | 0.462590 | 0.793201 | 0.689838 | 0.402802 | 0.403585 | 0.445590 |
| zcat11 | 0.671832 | 0.802241 | 0.939614 | 0.914329 | 0.816668 | 0.794090 | 0.787566 |
| zcat12 | 0.166753 | 0.580975 | 0.861370 | 0.766757 | 0.584988 | 0.564427 | 0.563015 |
| zcat13 | 0.734491 | 0.887477 | 0.978678 | 0.962672 | 0.892228 | 0.887956 | 0.882945 |
| zcat14 | 0.831262 | 0.823481 | 0.975999 | 0.930774 | 0.846613 | 0.826586 | 0.835323 |
| zcat15 | 0.849868 | 0.888981 | 0.988191 | 0.957588 | 0.890449 | 0.887681 | 0.891440 |
| zcat16 | 0.931061 | 0.915318 | 1.030890 | 1.010222 | 0.925386 | 0.937626 | 0.934716 |
| zcat17 | 0.875149 | 0.941901 | 0.993734 | 0.981214 | 0.952189 | 0.940532 | 0.947913 |
| zcat18 | 0.819354 | 0.874419 | 0.973417 | 0.949202 | 0.884391 | 0.866835 | 0.878956 |
| zcat19 | 0.625848 | 0.790252 | 0.944867 | 0.908706 | 0.803449 | 0.792874 | 0.771933 |
| zcat20 | 0.840227 | 0.880796 | 0.973465 | 0.944623 | 0.886755 | 0.891661 | 0.875639 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-hvref-farthest | nsga2-sector-farthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.138152 | 0.105392 | 0.065076 | 0.068295 | 0.108629 | 0.106790 | 0.108038 |
| zcat2 | 0.099635 | 0.064855 | 0.036939 | 0.046699 | 0.068221 | 0.066579 | 0.063568 |
| zcat3 | 0.261556 | 0.152509 | 0.065833 | 0.079687 | 0.140402 | 0.150151 | 0.157952 |
| zcat4 | 0.283423 | 0.184373 | 0.085325 | 0.098322 | 0.170825 | 0.179053 | 0.183295 |
| zcat5 | 0.135642 | 0.089035 | 0.048892 | 0.062749 | 0.089420 | 0.089472 | 0.090489 |
| zcat6 | 0.164386 | 0.092914 | 0.048388 | 0.050721 | 0.093782 | 0.093500 | 0.094791 |
| zcat7 | 0.200579 | 0.117564 | 0.040267 | 0.058069 | 0.104861 | 0.114768 | 0.119194 |
| zcat8 | 0.225101 | 0.128073 | 0.041743 | 0.061045 | 0.134415 | 0.133137 | 0.123106 |
| zcat9 | 0.138503 | 0.101955 | 0.043760 | 0.054939 | 0.104887 | 0.106379 | 0.104514 |
| zcat10 | 0.132805 | 0.090072 | 0.036521 | 0.046617 | 0.099074 | 0.103608 | 0.088904 |
| zcat11 | 0.217425 | 0.127511 | 0.051867 | 0.066388 | 0.122038 | 0.128879 | 0.138095 |
| zcat12 | 0.401889 | 0.158139 | 0.054282 | 0.080059 | 0.154595 | 0.168201 | 0.168163 |
| zcat13 | 0.226733 | 0.104640 | 0.040432 | 0.051713 | 0.101674 | 0.104036 | 0.106217 |
| zcat14 | 0.155090 | 0.132075 | 0.020559 | 0.055996 | 0.123898 | 0.126658 | 0.128080 |
| zcat15 | 0.265569 | 0.170903 | 0.016408 | 0.069231 | 0.174169 | 0.157795 | 0.157806 |
| zcat16 | 0.091439 | 0.103764 | 0.019241 | 0.037886 | 0.099651 | 0.098879 | 0.096276 |
| zcat17 | 0.190277 | 0.093503 | 0.037625 | 0.061421 | 0.104011 | 0.103380 | 0.091652 |
| zcat18 | 0.200197 | 0.114540 | 0.035701 | 0.049572 | 0.107132 | 0.120280 | 0.109924 |
| zcat19 | 0.311682 | 0.169911 | 0.064267 | 0.086946 | 0.164033 | 0.169104 | 0.183220 |
| zcat20 | 0.193825 | 0.116065 | 0.029988 | 0.051421 | 0.100505 | 0.110164 | 0.146093 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-hvref-farthest | nsga2-sector-farthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 4.249 | 52.693 | 192.423 | 294.052 | 19.440 | 15.126 | 20.415 |
| zcat2 | 4.522 | 45.957 | 176.870 | 252.608 | 14.211 | 17.763 | 19.315 |
| zcat3 | 4.522 | 46.122 | 157.582 | 222.702 | 16.062 | 11.655 | 14.783 |
| zcat4 | 3.624 | 39.139 | 153.293 | 230.137 | 15.979 | 11.874 | 15.580 |
| zcat5 | 3.891 | 40.573 | 147.515 | 213.902 | 16.281 | 11.980 | 15.755 |
| zcat6 | 3.593 | 36.518 | 142.026 | 213.077 | 15.130 | 11.236 | 14.185 |
| zcat7 | 3.831 | 35.803 | 148.300 | 209.536 | 15.084 | 10.976 | 13.873 |
| zcat8 | 3.969 | 34.470 | 134.944 | 190.114 | 14.896 | 10.648 | 13.420 |
| zcat9 | 3.870 | 37.129 | 139.870 | 208.357 | 15.506 | 10.834 | 14.096 |
| zcat10 | 3.879 | 34.470 | 134.896 | 195.631 | 14.404 | 10.702 | 14.395 |
| zcat11 | 3.967 | 36.959 | 156.902 | 218.358 | 15.435 | 11.025 | 14.940 |
| zcat12 | 3.900 | 38.025 | 151.625 | 210.938 | 15.097 | 11.083 | 14.643 |
| zcat13 | 4.050 | 37.215 | 149.199 | 220.467 | 15.207 | 11.043 | 14.621 |
| zcat14 | 3.910 | 36.164 | 136.226 | 196.583 | 15.121 | 10.985 | 13.744 |
| zcat15 | 3.907 | 36.161 | 128.920 | 194.002 | 14.975 | 10.476 | 13.498 |
| zcat16 | 3.823 | 36.180 | 132.714 | 197.710 | 15.244 | 10.562 | 12.938 |
| zcat17 | 3.912 | 37.515 | 140.922 | 198.724 | 15.510 | 11.411 | 14.690 |
| zcat18 | 3.892 | 35.410 | 154.761 | 207.049 | 15.252 | 10.850 | 13.976 |
| zcat19 | 4.038 | 37.048 | 153.667 | 208.690 | 15.115 | 10.936 | 14.043 |
| zcat20 | 3.769 | 35.249 | 140.764 | 191.263 | 14.716 | 9.881 | 12.152 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-farthest vs gces-noGeo | 13/0/8 | 10/0/11 | 8/7/6 |
| zcat1 | nsga2-farthest vs gces | 14/0/7 | 14/0/7 | 11/6/4 |
| zcat1 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat1 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat1 | nsga2-sector-farthest vs nsga2-farthest | 10/0/11 | 9/0/12 | 8/3/10 |
| zcat1 | nsga2-sector-farthest vs gces-noGeo | 13/0/8 | 9/0/12 | 9/4/8 |
| zcat1 | nsga2-sector-farthest vs gces | 14/0/7 | 9/0/12 | 9/5/7 |
| zcat1 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-farthest vs gces-noGeo | 18/0/3 | 12/0/9 | 12/6/3 |
| zcat2 | nsga2-farthest vs gces | 16/0/5 | 12/0/9 | 12/4/5 |
| zcat2 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat2 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat2 | nsga2-sector-farthest vs nsga2-farthest | 4/0/17 | 3/0/18 | 2/3/16 |
| zcat2 | nsga2-sector-farthest vs gces-noGeo | 15/0/6 | 5/0/16 | 5/10/6 |
| zcat2 | nsga2-sector-farthest vs gces | 12/0/9 | 7/0/14 | 7/5/9 |
| zcat2 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat3 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | gces vs nsgaii | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat3 | nsga2-farthest vs gces-noGeo | 9/0/12 | 9/0/12 | 9/0/12 |
| zcat3 | nsga2-farthest vs gces | 12/0/9 | 12/0/9 | 11/2/8 |
| zcat3 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat3 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat3 | nsga2-sector-farthest vs nsga2-farthest | 13/0/8 | 11/0/10 | 10/4/7 |
| zcat3 | nsga2-sector-farthest vs gces-noGeo | 11/0/10 | 10/0/11 | 10/1/10 |
| zcat3 | nsga2-sector-farthest vs gces | 15/0/6 | 16/0/5 | 15/1/5 |
| zcat3 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat4 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-farthest vs gces-noGeo | 11/0/10 | 11/0/10 | 10/2/9 |
| zcat4 | nsga2-farthest vs gces | 13/0/8 | 13/0/8 | 13/0/8 |
| zcat4 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-hvref-farthest vs nsga2-hvfarthest | 1/0/20 | 2/0/19 | 1/1/19 |
| zcat4 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat4 | nsga2-sector-farthest vs nsga2-farthest | 15/0/6 | 14/0/7 | 14/1/6 |
| zcat4 | nsga2-sector-farthest vs gces-noGeo | 15/0/6 | 12/0/9 | 12/3/6 |
| zcat4 | nsga2-sector-farthest vs gces | 18/0/3 | 16/0/5 | 16/2/3 |
| zcat4 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-farthest vs gces-noGeo | 14/0/7 | 11/0/10 | 10/5/6 |
| zcat5 | nsga2-farthest vs gces | 12/0/9 | 12/0/9 | 10/4/7 |
| zcat5 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat5 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat5 | nsga2-sector-farthest vs nsga2-farthest | 19/0/2 | 11/0/10 | 11/8/2 |
| zcat5 | nsga2-sector-farthest vs gces-noGeo | 20/0/1 | 11/0/10 | 11/9/1 |
| zcat5 | nsga2-sector-farthest vs gces | 21/0/0 | 14/0/7 | 14/7/0 |
| zcat5 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-farthest vs gces-noGeo | 9/0/12 | 12/0/9 | 8/5/8 |
| zcat6 | nsga2-farthest vs gces | 14/0/7 | 13/0/8 | 10/7/4 |
| zcat6 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat6 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat6 | nsga2-sector-farthest vs nsga2-farthest | 15/0/6 | 9/0/12 | 9/6/6 |
| zcat6 | nsga2-sector-farthest vs gces-noGeo | 17/0/4 | 10/0/11 | 10/7/4 |
| zcat6 | nsga2-sector-farthest vs gces | 18/0/3 | 12/0/9 | 12/6/3 |
| zcat6 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat7 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-farthest vs gces-noGeo | 10/0/11 | 9/0/12 | 7/5/9 |
| zcat7 | nsga2-farthest vs gces | 12/0/9 | 12/0/9 | 11/2/8 |
| zcat7 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat7 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat7 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat7 | nsga2-hvfarthest vs gces | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat7 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat7 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat7 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 3/0/18 | 0/3/18 |
| zcat7 | nsga2-sector-farthest vs nsgaii | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat7 | nsga2-sector-farthest vs nsga2-farthest | 13/0/8 | 17/0/4 | 12/6/3 |
| zcat7 | nsga2-sector-farthest vs gces-noGeo | 15/0/6 | 14/0/7 | 14/1/6 |
| zcat7 | nsga2-sector-farthest vs gces | 15/0/6 | 17/0/4 | 15/2/4 |
| zcat7 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat8 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-farthest vs gces-noGeo | 11/0/10 | 14/0/7 | 10/5/6 |
| zcat8 | nsga2-farthest vs gces | 13/0/8 | 13/0/8 | 10/6/5 |
| zcat8 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat8 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat8 | nsga2-sector-farthest vs nsgaii | 20/1/0 | 21/0/0 | 20/1/0 |
| zcat8 | nsga2-sector-farthest vs nsga2-farthest | 10/0/11 | 8/0/13 | 6/6/9 |
| zcat8 | nsga2-sector-farthest vs gces-noGeo | 11/0/10 | 10/0/11 | 7/7/7 |
| zcat8 | nsga2-sector-farthest vs gces | 12/0/9 | 12/0/9 | 9/6/6 |
| zcat8 | nsga2-sector-farthest vs nsga2-hvref-farthest | 1/0/20 | 0/0/21 | 0/1/20 |
| zcat9 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | gces-noGeo vs nsgaii | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat9 | gces vs nsgaii | 19/0/2 | 20/0/1 | 19/1/1 |
| zcat9 | nsga2-farthest vs gces-noGeo | 12/0/9 | 14/0/7 | 11/4/6 |
| zcat9 | nsga2-farthest vs gces | 11/0/10 | 13/0/8 | 11/2/8 |
| zcat9 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat9 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat9 | nsga2-sector-farthest vs nsgaii | 20/0/1 | 20/0/1 | 19/2/0 |
| zcat9 | nsga2-sector-farthest vs nsga2-farthest | 9/0/12 | 9/0/12 | 9/0/12 |
| zcat9 | nsga2-sector-farthest vs gces-noGeo | 9/0/12 | 10/0/11 | 8/3/10 |
| zcat9 | nsga2-sector-farthest vs gces | 10/0/11 | 11/0/10 | 10/1/10 |
| zcat9 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat10 | nsga2-farthest vs nsgaii | 19/0/2 | 20/0/1 | 18/3/0 |
| zcat10 | gces-noGeo vs nsgaii | 17/0/4 | 16/0/5 | 16/1/4 |
| zcat10 | gces vs nsgaii | 19/0/2 | 19/0/2 | 19/0/2 |
| zcat10 | nsga2-farthest vs gces-noGeo | 14/0/7 | 14/0/7 | 13/2/6 |
| zcat10 | nsga2-farthest vs gces | 11/0/10 | 9/0/12 | 9/2/10 |
| zcat10 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat10 | nsga2-hvref-farthest vs gces | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat10 | nsga2-hvref-farthest vs nsga2-hvfarthest | 2/0/19 | 2/0/19 | 2/0/19 |
| zcat10 | nsga2-sector-farthest vs nsgaii | 18/0/3 | 19/0/2 | 18/1/2 |
| zcat10 | nsga2-sector-farthest vs nsga2-farthest | 7/0/14 | 8/0/13 | 7/1/13 |
| zcat10 | nsga2-sector-farthest vs gces-noGeo | 12/0/9 | 11/0/10 | 11/1/9 |
| zcat10 | nsga2-sector-farthest vs gces | 11/0/10 | 11/0/10 | 11/0/10 |
| zcat10 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat11 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-farthest vs gces-noGeo | 13/0/8 | 10/0/11 | 10/3/8 |
| zcat11 | nsga2-farthest vs gces | 11/0/10 | 10/0/11 | 10/1/10 |
| zcat11 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat11 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat11 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat11 | nsga2-hvfarthest vs gces | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat11 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-hvref-farthest vs nsga2-hvfarthest | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat11 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat11 | nsga2-sector-farthest vs nsga2-farthest | 14/0/7 | 14/0/7 | 13/2/6 |
| zcat11 | nsga2-sector-farthest vs gces-noGeo | 17/0/4 | 16/0/5 | 16/1/4 |
| zcat11 | nsga2-sector-farthest vs gces | 16/0/5 | 15/0/6 | 15/1/5 |
| zcat11 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-farthest vs gces-noGeo | 12/0/9 | 13/0/8 | 10/5/6 |
| zcat12 | nsga2-farthest vs gces | 13/0/8 | 15/0/6 | 13/2/6 |
| zcat12 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-hvref-farthest vs gces-noGeo | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat12 | nsga2-hvref-farthest vs gces | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat12 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat12 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat12 | nsga2-sector-farthest vs nsga2-farthest | 13/0/8 | 13/0/8 | 11/4/6 |
| zcat12 | nsga2-sector-farthest vs gces-noGeo | 14/0/7 | 16/0/5 | 14/2/5 |
| zcat12 | nsga2-sector-farthest vs gces | 13/0/8 | 15/0/6 | 13/2/6 |
| zcat12 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat13 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | gces-noGeo vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-farthest vs gces-noGeo | 11/0/10 | 11/0/10 | 11/0/10 |
| zcat13 | nsga2-farthest vs gces | 11/0/10 | 12/0/9 | 11/1/9 |
| zcat13 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat13 | nsga2-hvfarthest vs nsga2-farthest | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat13 | nsga2-hvfarthest vs gces-noGeo | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat13 | nsga2-hvfarthest vs gces | 20/0/1 | 20/0/1 | 20/0/1 |
| zcat13 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-hvref-farthest vs nsga2-hvfarthest | 1/0/20 | 1/0/20 | 1/0/20 |
| zcat13 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat13 | nsga2-sector-farthest vs nsga2-farthest | 11/0/10 | 11/0/10 | 11/0/10 |
| zcat13 | nsga2-sector-farthest vs gces-noGeo | 13/0/8 | 13/0/8 | 11/4/6 |
| zcat13 | nsga2-sector-farthest vs gces | 14/0/7 | 16/0/5 | 14/2/5 |
| zcat13 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-farthest vs nsgaii | 8/0/13 | 12/0/9 | 8/4/9 |
| zcat14 | gces-noGeo vs nsgaii | 10/0/11 | 14/0/7 | 10/4/7 |
| zcat14 | gces vs nsgaii | 8/0/13 | 13/0/8 | 8/5/8 |
| zcat14 | nsga2-farthest vs gces-noGeo | 7/0/14 | 9/0/12 | 7/2/12 |
| zcat14 | nsga2-farthest vs gces | 7/0/14 | 8/0/13 | 4/7/10 |
| zcat14 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat14 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat14 | nsga2-sector-farthest vs nsgaii | 11/0/10 | 14/0/7 | 10/5/6 |
| zcat14 | nsga2-sector-farthest vs nsga2-farthest | 16/0/5 | 15/0/6 | 14/3/4 |
| zcat14 | nsga2-sector-farthest vs gces-noGeo | 14/0/7 | 11/0/10 | 11/3/7 |
| zcat14 | nsga2-sector-farthest vs gces | 14/0/7 | 12/0/9 | 12/2/7 |
| zcat14 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat15 | nsga2-farthest vs nsgaii | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat15 | gces-noGeo vs nsgaii | 18/0/3 | 20/0/1 | 18/2/1 |
| zcat15 | gces vs nsgaii | 18/0/3 | 19/0/2 | 17/3/1 |
| zcat15 | nsga2-farthest vs gces-noGeo | 9/0/12 | 6/0/15 | 5/5/11 |
| zcat15 | nsga2-farthest vs gces | 12/0/9 | 7/0/14 | 6/7/8 |
| zcat15 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 19/0/2 | 19/2/0 |
| zcat15 | nsga2-hvfarthest vs gces | 21/0/0 | 19/0/2 | 19/2/0 |
| zcat15 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat15 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat15 | nsga2-hvref-farthest vs gces | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat15 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 1/0/20 | 0/1/20 |
| zcat15 | nsga2-sector-farthest vs nsgaii | 19/0/2 | 21/0/0 | 19/2/0 |
| zcat15 | nsga2-sector-farthest vs nsga2-farthest | 12/0/9 | 14/0/7 | 11/4/6 |
| zcat15 | nsga2-sector-farthest vs gces-noGeo | 13/0/8 | 8/0/13 | 7/7/7 |
| zcat15 | nsga2-sector-farthest vs gces | 10/0/11 | 9/0/12 | 8/3/10 |
| zcat15 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-farthest vs nsgaii | 9/0/12 | 6/0/15 | 6/3/12 |
| zcat16 | gces-noGeo vs nsgaii | 9/0/12 | 9/0/12 | 7/4/10 |
| zcat16 | gces vs nsgaii | 11/0/10 | 6/0/15 | 6/5/10 |
| zcat16 | nsga2-farthest vs gces-noGeo | 9/0/12 | 9/0/12 | 9/0/12 |
| zcat16 | nsga2-farthest vs gces | 5/0/16 | 8/0/13 | 5/3/13 |
| zcat16 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat16 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat16 | nsga2-sector-farthest vs nsgaii | 10/0/11 | 8/0/13 | 6/6/9 |
| zcat16 | nsga2-sector-farthest vs nsga2-farthest | 13/0/8 | 12/0/9 | 10/5/6 |
| zcat16 | nsga2-sector-farthest vs gces-noGeo | 12/0/9 | 12/0/9 | 12/0/9 |
| zcat16 | nsga2-sector-farthest vs gces | 8/0/13 | 7/0/14 | 7/1/13 |
| zcat16 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | gces-noGeo vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat17 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-farthest vs gces-noGeo | 12/0/9 | 13/0/8 | 10/5/6 |
| zcat17 | nsga2-farthest vs gces | 9/0/12 | 12/0/9 | 6/9/6 |
| zcat17 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat17 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat17 | nsga2-sector-farthest vs nsga2-farthest | 16/0/5 | 7/0/14 | 7/9/5 |
| zcat17 | nsga2-sector-farthest vs gces-noGeo | 16/0/5 | 13/0/8 | 11/7/3 |
| zcat17 | nsga2-sector-farthest vs gces | 12/0/9 | 5/0/16 | 4/9/8 |
| zcat17 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-farthest vs nsgaii | 20/0/1 | 20/0/1 | 19/2/0 |
| zcat18 | gces-noGeo vs nsgaii | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat18 | gces vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-farthest vs gces-noGeo | 13/0/8 | 11/0/10 | 10/4/7 |
| zcat18 | nsga2-farthest vs gces | 8/0/13 | 9/0/12 | 6/5/10 |
| zcat18 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat18 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat18 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat18 | nsga2-sector-farthest vs nsga2-farthest | 14/0/7 | 15/0/6 | 13/3/5 |
| zcat18 | nsga2-sector-farthest vs gces-noGeo | 17/0/4 | 17/0/4 | 16/2/3 |
| zcat18 | nsga2-sector-farthest vs gces | 14/0/7 | 15/0/6 | 12/5/4 |
| zcat18 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | gces-noGeo vs nsgaii | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat19 | gces vs nsgaii | 20/0/1 | 21/0/0 | 20/1/0 |
| zcat19 | nsga2-farthest vs gces-noGeo | 8/0/13 | 8/0/13 | 7/2/12 |
| zcat19 | nsga2-farthest vs gces | 14/0/7 | 12/0/9 | 12/2/7 |
| zcat19 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvfarthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs gces | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat19 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat19 | nsga2-sector-farthest vs nsga2-farthest | 10/0/11 | 13/0/8 | 9/5/7 |
| zcat19 | nsga2-sector-farthest vs gces-noGeo | 12/0/9 | 12/0/9 | 10/4/7 |
| zcat19 | nsga2-sector-farthest vs gces | 15/0/6 | 14/0/7 | 14/1/6 |
| zcat19 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 0/0/21 | 0/0/21 |
| zcat20 | nsga2-farthest vs nsgaii | 20/0/1 | 18/0/3 | 17/4/0 |
| zcat20 | gces-noGeo vs nsgaii | 19/0/2 | 18/0/3 | 17/3/1 |
| zcat20 | gces vs nsgaii | 20/0/1 | 19/0/2 | 18/3/0 |
| zcat20 | nsga2-farthest vs gces-noGeo | 11/0/10 | 9/0/12 | 7/6/8 |
| zcat20 | nsga2-farthest vs gces | 12/0/9 | 13/0/8 | 9/7/5 |
| zcat20 | nsga2-hvfarthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat20 | nsga2-hvfarthest vs nsga2-farthest | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat20 | nsga2-hvfarthest vs gces-noGeo | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat20 | nsga2-hvfarthest vs gces | 21/0/0 | 20/0/1 | 20/1/0 |
| zcat20 | nsga2-hvref-farthest vs nsgaii | 21/0/0 | 21/0/0 | 21/0/0 |
| zcat20 | nsga2-hvref-farthest vs nsga2-farthest | 21/0/0 | 17/0/4 | 17/4/0 |
| zcat20 | nsga2-hvref-farthest vs gces-noGeo | 21/0/0 | 15/0/6 | 15/6/0 |
| zcat20 | nsga2-hvref-farthest vs gces | 21/0/0 | 18/0/3 | 18/3/0 |
| zcat20 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/0/21 | 4/0/17 | 0/4/17 |
| zcat20 | nsga2-sector-farthest vs nsgaii | 21/0/0 | 19/0/2 | 19/2/0 |
| zcat20 | nsga2-sector-farthest vs nsga2-farthest | 13/0/8 | 15/0/6 | 10/8/3 |
| zcat20 | nsga2-sector-farthest vs gces-noGeo | 13/0/8 | 10/0/11 | 9/5/7 |
| zcat20 | nsga2-sector-farthest vs gces | 15/0/6 | 15/0/6 | 10/10/1 |
| zcat20 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/0/21 | 5/0/16 | 0/5/16 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0.064773 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | gces-noGeo vs nsgaii | 0.061767 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | gces vs nsgaii | 0.060073 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-farthest vs gces-noGeo | 0.003005 | 13/0/8 | 0.272210 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces | 0.004699 | 14/0/7 | 0.064650 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsgaii | 0.112333 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 0.047561 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 0.050566 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvfarthest vs gces | 0.052260 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0.107918 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0.043145 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0.046151 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs gces | 0.047844 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.004416 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-sector-farthest vs nsgaii | 0.063816 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-sector-farthest vs nsga2-farthest | -0.000956 | 10/0/11 | 0.972858 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces-noGeo | 0.002049 | 13/0/8 | 0.473334 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces | 0.003743 | 14/0/7 | 0.146964 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.044102 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-farthest vs nsgaii | 0.183721 | 19/0/2 | 0.000013 | 0.001589 | yes |
| zcat10 | gces-noGeo vs nsgaii | 0.124717 | 17/0/4 | 0.000241 | 0.026058 | yes |
| zcat10 | gces vs nsgaii | 0.166721 | 19/0/2 | 0.000024 | 0.002766 | yes |
| zcat10 | nsga2-farthest vs gces-noGeo | 0.059005 | 14/0/7 | 0.355416 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces | 0.017000 | 11/0/10 | 0.972858 | 1.000000 | no |
| zcat10 | nsga2-hvfarthest vs nsgaii | 0.514332 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvfarthest vs nsga2-farthest | 0.330611 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvfarthest vs gces-noGeo | 0.389616 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvfarthest vs gces | 0.347611 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs nsgaii | 0.410969 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-farthest | 0.227248 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs gces-noGeo | 0.286252 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs gces | 0.244248 | 20/0/1 | 0.000003 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.103363 | 2/0/19 | 0.000024 | 0.002766 | yes |
| zcat10 | nsga2-sector-farthest vs nsgaii | 0.123933 | 18/0/3 | 0.000013 | 0.001589 | yes |
| zcat10 | nsga2-sector-farthest vs nsga2-farthest | -0.059788 | 7/0/14 | 0.373725 | 1.000000 | no |
| zcat10 | nsga2-sector-farthest vs gces-noGeo | -0.000783 | 12/0/9 | 0.759288 | 1.000000 | no |
| zcat10 | nsga2-sector-farthest vs gces | -0.042788 | 11/0/10 | 0.494802 | 1.000000 | no |
| zcat10 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.287035 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-farthest vs nsgaii | 0.130409 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | gces-noGeo vs nsgaii | 0.122258 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | gces vs nsgaii | 0.115734 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-farthest vs gces-noGeo | 0.008151 | 13/0/8 | 0.494802 | 1.000000 | no |
| zcat11 | nsga2-farthest vs gces | 0.014675 | 11/0/10 | 0.539189 | 1.000000 | no |
| zcat11 | nsga2-hvfarthest vs nsgaii | 0.267781 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvfarthest vs nsga2-farthest | 0.137372 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvfarthest vs gces-noGeo | 0.145524 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvfarthest vs gces | 0.152048 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs nsgaii | 0.242497 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-farthest | 0.112088 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs gces-noGeo | 0.120239 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs gces | 0.126763 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.025284 | 1/0/20 | 0.000067 | 0.007477 | yes |
| zcat11 | nsga2-sector-farthest vs nsgaii | 0.144836 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-sector-farthest vs nsga2-farthest | 0.014427 | 14/0/7 | 0.015780 | 1.000000 | no |
| zcat11 | nsga2-sector-farthest vs gces-noGeo | 0.022578 | 17/0/4 | 0.002857 | 0.280006 | no |
| zcat11 | nsga2-sector-farthest vs gces | 0.029102 | 16/0/5 | 0.004285 | 0.415631 | no |
| zcat11 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.097661 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-farthest vs nsgaii | 0.414222 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | gces-noGeo vs nsgaii | 0.397674 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | gces vs nsgaii | 0.396262 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-farthest vs gces-noGeo | 0.016548 | 12/0/9 | 0.228986 | 1.000000 | no |
| zcat12 | nsga2-farthest vs gces | 0.017960 | 13/0/8 | 0.215680 | 1.000000 | no |
| zcat12 | nsga2-hvfarthest vs nsgaii | 0.694617 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvfarthest vs nsga2-farthest | 0.280395 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvfarthest vs gces-noGeo | 0.296943 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvfarthest vs gces | 0.298355 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs nsgaii | 0.600004 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-farthest | 0.185782 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs gces-noGeo | 0.202330 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs gces | 0.203741 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.094613 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-sector-farthest vs nsgaii | 0.418236 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-sector-farthest vs nsga2-farthest | 0.004013 | 13/0/8 | 0.539189 | 1.000000 | no |
| zcat12 | nsga2-sector-farthest vs gces-noGeo | 0.020562 | 14/0/7 | 0.128078 | 1.000000 | no |
| zcat12 | nsga2-sector-farthest vs gces | 0.021973 | 13/0/8 | 0.157134 | 1.000000 | no |
| zcat12 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.181768 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-farthest vs nsgaii | 0.152985 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | gces-noGeo vs nsgaii | 0.153464 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | gces vs nsgaii | 0.148454 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-farthest vs gces-noGeo | -0.000479 | 11/0/10 | 0.392584 | 1.000000 | no |
| zcat13 | nsga2-farthest vs gces | 0.004532 | 11/0/10 | 0.287734 | 1.000000 | no |
| zcat13 | nsga2-hvfarthest vs nsgaii | 0.244186 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvfarthest vs nsga2-farthest | 0.091201 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat13 | nsga2-hvfarthest vs gces-noGeo | 0.090722 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat13 | nsga2-hvfarthest vs gces | 0.095733 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs nsgaii | 0.228180 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-farthest | 0.075195 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs gces-noGeo | 0.074716 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs gces | 0.079727 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.016006 | 1/0/20 | 0.000426 | 0.045187 | yes |
| zcat13 | nsga2-sector-farthest vs nsgaii | 0.157736 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-sector-farthest vs nsga2-farthest | 0.004751 | 11/0/10 | 0.838194 | 1.000000 | no |
| zcat13 | nsga2-sector-farthest vs gces-noGeo | 0.004272 | 13/0/8 | 0.119342 | 1.000000 | no |
| zcat13 | nsga2-sector-farthest vs gces | 0.009283 | 14/0/7 | 0.157134 | 1.000000 | no |
| zcat13 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.070444 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-farthest vs nsgaii | -0.007781 | 8/0/13 | 0.516761 | 1.000000 | no |
| zcat14 | gces-noGeo vs nsgaii | -0.004676 | 10/0/11 | 0.682714 | 1.000000 | no |
| zcat14 | gces vs nsgaii | 0.004061 | 8/0/13 | 0.972858 | 1.000000 | no |
| zcat14 | nsga2-farthest vs gces-noGeo | -0.003105 | 7/0/14 | 0.095799 | 1.000000 | no |
| zcat14 | nsga2-farthest vs gces | -0.011842 | 7/0/14 | 0.054693 | 1.000000 | no |
| zcat14 | nsga2-hvfarthest vs nsgaii | 0.144737 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvfarthest vs nsga2-farthest | 0.152518 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvfarthest vs gces-noGeo | 0.149413 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvfarthest vs gces | 0.140676 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs nsgaii | 0.099512 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-farthest | 0.107293 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs gces-noGeo | 0.104188 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs gces | 0.095451 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.045225 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-sector-farthest vs nsgaii | 0.015351 | 11/0/10 | 0.657827 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs nsga2-farthest | 0.023132 | 16/0/5 | 0.005542 | 0.532013 | no |
| zcat14 | nsga2-sector-farthest vs gces-noGeo | 0.020027 | 14/0/7 | 0.167807 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs gces | 0.011290 | 14/0/7 | 0.167807 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.084161 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-farthest vs nsgaii | 0.039113 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat15 | gces-noGeo vs nsgaii | 0.037813 | 18/0/3 | 0.000197 | 0.021518 | yes |
| zcat15 | gces vs nsgaii | 0.041573 | 18/0/3 | 0.000241 | 0.026058 | yes |
| zcat15 | nsga2-farthest vs gces-noGeo | 0.001300 | 9/0/12 | 0.473334 | 1.000000 | no |
| zcat15 | nsga2-farthest vs gces | -0.002459 | 12/0/9 | 0.516761 | 1.000000 | no |
| zcat15 | nsga2-hvfarthest vs nsgaii | 0.138323 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvfarthest vs nsga2-farthest | 0.099210 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvfarthest vs gces-noGeo | 0.100510 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvfarthest vs gces | 0.096751 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs nsgaii | 0.107721 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-farthest | 0.068607 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs gces-noGeo | 0.069907 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs gces | 0.066148 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.030602 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-sector-farthest vs nsgaii | 0.040581 | 19/0/2 | 0.000084 | 0.009232 | yes |
| zcat15 | nsga2-sector-farthest vs nsga2-farthest | 0.001468 | 12/0/9 | 0.082195 | 1.000000 | no |
| zcat15 | nsga2-sector-farthest vs gces-noGeo | 0.002768 | 13/0/8 | 0.473334 | 1.000000 | no |
| zcat15 | nsga2-sector-farthest vs gces | -0.000991 | 10/0/11 | 0.733470 | 1.000000 | no |
| zcat15 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.067139 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-farthest vs nsgaii | -0.015742 | 9/0/12 | 0.562075 | 1.000000 | no |
| zcat16 | gces-noGeo vs nsgaii | 0.006566 | 9/0/12 | 0.918694 | 1.000000 | no |
| zcat16 | gces vs nsgaii | 0.003656 | 11/0/10 | 0.838194 | 1.000000 | no |
| zcat16 | nsga2-farthest vs gces-noGeo | -0.022308 | 9/0/12 | 0.355416 | 1.000000 | no |
| zcat16 | nsga2-farthest vs gces | -0.019398 | 5/0/16 | 0.128078 | 1.000000 | no |
| zcat16 | nsga2-hvfarthest vs nsgaii | 0.099829 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvfarthest vs nsga2-farthest | 0.115571 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvfarthest vs gces-noGeo | 0.093263 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvfarthest vs gces | 0.096173 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs nsgaii | 0.079162 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-farthest | 0.094904 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs gces-noGeo | 0.072596 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs gces | 0.075506 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.020667 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-sector-farthest vs nsgaii | -0.005675 | 10/0/11 | 0.733470 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs nsga2-farthest | 0.010067 | 13/0/8 | 0.082195 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs gces-noGeo | -0.012241 | 12/0/9 | 0.562075 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs gces | -0.009331 | 8/0/13 | 0.945745 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.084837 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-farthest vs nsgaii | 0.066752 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | gces-noGeo vs nsgaii | 0.065383 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | gces vs nsgaii | 0.072764 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-farthest vs gces-noGeo | 0.001369 | 12/0/9 | 0.494802 | 1.000000 | no |
| zcat17 | nsga2-farthest vs gces | -0.006012 | 9/0/12 | 0.082195 | 1.000000 | no |
| zcat17 | nsga2-hvfarthest vs nsgaii | 0.118585 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvfarthest vs nsga2-farthest | 0.051832 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvfarthest vs gces-noGeo | 0.053202 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvfarthest vs gces | 0.045821 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs nsgaii | 0.106065 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-farthest | 0.039312 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs gces-noGeo | 0.040681 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs gces | 0.033301 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.012520 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-sector-farthest vs nsgaii | 0.077040 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-sector-farthest vs nsga2-farthest | 0.010288 | 16/0/5 | 0.009016 | 0.847507 | no |
| zcat17 | nsga2-sector-farthest vs gces-noGeo | 0.011657 | 16/0/5 | 0.000510 | 0.053062 | no |
| zcat17 | nsga2-sector-farthest vs gces | 0.004276 | 12/0/9 | 0.157134 | 1.000000 | no |
| zcat17 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.029025 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-farthest vs nsgaii | 0.055066 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat18 | gces-noGeo vs nsgaii | 0.047481 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat18 | gces vs nsgaii | 0.059602 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-farthest vs gces-noGeo | 0.007584 | 13/0/8 | 0.157134 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces | -0.004537 | 8/0/13 | 0.202917 | 1.000000 | no |
| zcat18 | nsga2-hvfarthest vs nsgaii | 0.154063 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvfarthest vs nsga2-farthest | 0.098997 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvfarthest vs gces-noGeo | 0.106581 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvfarthest vs gces | 0.094461 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs nsgaii | 0.129848 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-farthest | 0.074783 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs gces-noGeo | 0.082367 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs gces | 0.070246 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.024215 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-sector-farthest vs nsgaii | 0.065037 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-sector-farthest vs nsga2-farthest | 0.009972 | 14/0/7 | 0.146964 | 1.000000 | no |
| zcat18 | nsga2-sector-farthest vs gces-noGeo | 0.017556 | 17/0/4 | 0.017547 | 1.000000 | no |
| zcat18 | nsga2-sector-farthest vs gces | 0.005435 | 14/0/7 | 0.202917 | 1.000000 | no |
| zcat18 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.064811 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-farthest vs nsgaii | 0.164404 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | gces-noGeo vs nsgaii | 0.167027 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat19 | gces vs nsgaii | 0.146085 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat19 | nsga2-farthest vs gces-noGeo | -0.002622 | 8/0/13 | 0.657827 | 1.000000 | no |
| zcat19 | nsga2-farthest vs gces | 0.018319 | 14/0/7 | 0.355416 | 1.000000 | no |
| zcat19 | nsga2-hvfarthest vs nsgaii | 0.319019 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvfarthest vs nsga2-farthest | 0.154615 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvfarthest vs gces-noGeo | 0.151992 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvfarthest vs gces | 0.172934 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs nsgaii | 0.282859 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-farthest | 0.118454 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs gces-noGeo | 0.115832 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs gces | 0.136774 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.036160 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-sector-farthest vs nsgaii | 0.177601 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-sector-farthest vs nsga2-farthest | 0.013197 | 10/0/11 | 0.539189 | 1.000000 | no |
| zcat19 | nsga2-sector-farthest vs gces-noGeo | 0.010574 | 12/0/9 | 0.228986 | 1.000000 | no |
| zcat19 | nsga2-sector-farthest vs gces | 0.031516 | 15/0/6 | 0.050192 | 1.000000 | no |
| zcat19 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.105258 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-farthest vs nsgaii | 0.013742 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | gces-noGeo vs nsgaii | 0.011562 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | gces vs nsgaii | 0.012030 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-farthest vs gces-noGeo | 0.002181 | 18/0/3 | 0.002151 | 0.215149 | no |
| zcat2 | nsga2-farthest vs gces | 0.001712 | 16/0/5 | 0.000510 | 0.053062 | no |
| zcat2 | nsga2-hvfarthest vs nsgaii | 0.023662 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvfarthest vs nsga2-farthest | 0.009919 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvfarthest vs gces-noGeo | 0.012100 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvfarthest vs gces | 0.011631 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs nsgaii | 0.020652 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-farthest | 0.006909 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs gces-noGeo | 0.009090 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs gces | 0.008621 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.003010 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-sector-farthest vs nsgaii | 0.012571 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-sector-farthest vs nsga2-farthest | -0.001172 | 4/0/17 | 0.001002 | 0.102236 | no |
| zcat2 | nsga2-sector-farthest vs gces-noGeo | 0.001009 | 15/0/6 | 0.059507 | 1.000000 | no |
| zcat2 | nsga2-sector-farthest vs gces | 0.000540 | 12/0/9 | 0.190687 | 1.000000 | no |
| zcat2 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.008081 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-farthest vs nsgaii | 0.040569 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat20 | gces-noGeo vs nsgaii | 0.051434 | 19/0/2 | 0.000067 | 0.007477 | yes |
| zcat20 | gces vs nsgaii | 0.035412 | 20/0/1 | 0.000024 | 0.002766 | yes |
| zcat20 | nsga2-farthest vs gces-noGeo | -0.010865 | 11/0/10 | 0.707934 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces | 0.005157 | 12/0/9 | 0.190687 | 1.000000 | no |
| zcat20 | nsga2-hvfarthest vs nsgaii | 0.133237 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvfarthest vs nsga2-farthest | 0.092669 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvfarthest vs gces-noGeo | 0.081804 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvfarthest vs gces | 0.097825 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs nsgaii | 0.104395 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-farthest | 0.063827 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs gces-noGeo | 0.052961 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs gces | 0.068983 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.028842 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-sector-farthest vs nsgaii | 0.046527 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-sector-farthest vs nsga2-farthest | 0.005959 | 13/0/8 | 0.215680 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs gces-noGeo | -0.004906 | 13/0/8 | 0.272210 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs gces | 0.011115 | 15/0/6 | 0.031919 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.057868 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-farthest vs nsgaii | 0.142059 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | gces-noGeo vs nsgaii | 0.147232 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | gces vs nsgaii | 0.137581 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat3 | nsga2-farthest vs gces-noGeo | -0.005172 | 9/0/12 | 0.891732 | 1.000000 | no |
| zcat3 | nsga2-farthest vs gces | 0.004478 | 12/0/9 | 0.287734 | 1.000000 | no |
| zcat3 | nsga2-hvfarthest vs nsgaii | 0.301889 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvfarthest vs nsga2-farthest | 0.159830 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvfarthest vs gces-noGeo | 0.154658 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvfarthest vs gces | 0.164308 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs nsgaii | 0.273847 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-farthest | 0.131788 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs gces-noGeo | 0.126616 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs gces | 0.136266 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.028042 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-sector-farthest vs nsgaii | 0.157602 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-sector-farthest vs nsga2-farthest | 0.015543 | 13/0/8 | 0.119342 | 1.000000 | no |
| zcat3 | nsga2-sector-farthest vs gces-noGeo | 0.010371 | 11/0/10 | 0.242842 | 1.000000 | no |
| zcat3 | nsga2-sector-farthest vs gces | 0.020021 | 15/0/6 | 0.019473 | 1.000000 | no |
| zcat3 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.116245 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-farthest vs nsgaii | 0.107213 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | gces-noGeo vs nsgaii | 0.114458 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | gces vs nsgaii | 0.102250 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-farthest vs gces-noGeo | -0.007245 | 11/0/10 | 1.000000 | 1.000000 | no |
| zcat4 | nsga2-farthest vs gces | 0.004963 | 13/0/8 | 0.242842 | 1.000000 | no |
| zcat4 | nsga2-hvfarthest vs nsgaii | 0.248480 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvfarthest vs nsga2-farthest | 0.141266 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvfarthest vs gces-noGeo | 0.134022 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvfarthest vs gces | 0.146230 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs nsgaii | 0.223913 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-farthest | 0.116700 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs gces-noGeo | 0.109456 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs gces | 0.121663 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.024566 | 1/0/20 | 0.000003 | 0.000362 | yes |
| zcat4 | nsga2-sector-farthest vs nsgaii | 0.130859 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-sector-farthest vs nsga2-farthest | 0.023646 | 15/0/6 | 0.012691 | 1.000000 | no |
| zcat4 | nsga2-sector-farthest vs gces-noGeo | 0.016401 | 15/0/6 | 0.009016 | 0.847507 | no |
| zcat4 | nsga2-sector-farthest vs gces | 0.028609 | 18/0/3 | 0.001600 | 0.161627 | no |
| zcat4 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.093054 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-farthest vs nsgaii | 0.025511 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | gces-noGeo vs nsgaii | 0.024521 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | gces vs nsgaii | 0.023472 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-farthest vs gces-noGeo | 0.000991 | 14/0/7 | 0.157134 | 1.000000 | no |
| zcat5 | nsga2-farthest vs gces | 0.002039 | 12/0/9 | 0.082195 | 1.000000 | no |
| zcat5 | nsga2-hvfarthest vs nsgaii | 0.057827 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvfarthest vs nsga2-farthest | 0.032316 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvfarthest vs gces-noGeo | 0.033306 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvfarthest vs gces | 0.034355 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs nsgaii | 0.048624 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-farthest | 0.023113 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs gces-noGeo | 0.024104 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs gces | 0.025153 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.009202 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-sector-farthest vs nsgaii | 0.032557 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-sector-farthest vs nsga2-farthest | 0.007046 | 19/0/2 | 0.000013 | 0.001589 | yes |
| zcat5 | nsga2-sector-farthest vs gces-noGeo | 0.008036 | 20/0/1 | 0.000010 | 0.001154 | yes |
| zcat5 | nsga2-sector-farthest vs gces | 0.009085 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.016068 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-farthest vs nsgaii | 0.095076 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | gces-noGeo vs nsgaii | 0.093045 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | gces vs nsgaii | 0.089339 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-farthest vs gces-noGeo | 0.002031 | 9/0/12 | 0.811678 | 1.000000 | no |
| zcat6 | nsga2-farthest vs gces | 0.005737 | 14/0/7 | 0.103214 | 1.000000 | no |
| zcat6 | nsga2-hvfarthest vs nsgaii | 0.210857 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvfarthest vs nsga2-farthest | 0.115781 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvfarthest vs gces-noGeo | 0.117812 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvfarthest vs gces | 0.121518 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs nsgaii | 0.205292 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-farthest | 0.110216 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs gces-noGeo | 0.112248 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs gces | 0.115954 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.005564 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-sector-farthest vs nsgaii | 0.109818 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-sector-farthest vs nsga2-farthest | 0.014742 | 15/0/6 | 0.015780 | 1.000000 | no |
| zcat6 | nsga2-sector-farthest vs gces-noGeo | 0.016773 | 17/0/4 | 0.009016 | 0.847507 | no |
| zcat6 | nsga2-sector-farthest vs gces | 0.020479 | 18/0/3 | 0.000426 | 0.045187 | yes |
| zcat6 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.095475 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-farthest vs nsgaii | 0.061108 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | gces-noGeo vs nsgaii | 0.059390 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | gces vs nsgaii | 0.058070 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-farthest vs gces-noGeo | 0.001719 | 10/0/11 | 0.972858 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces | 0.003039 | 12/0/9 | 0.539189 | 1.000000 | no |
| zcat7 | nsga2-hvfarthest vs nsgaii | 0.151222 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvfarthest vs nsga2-farthest | 0.090114 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvfarthest vs gces-noGeo | 0.091832 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvfarthest vs gces | 0.093153 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs nsgaii | 0.123390 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-farthest | 0.062282 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs gces-noGeo | 0.064000 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs gces | 0.065320 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.027832 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-sector-farthest vs nsgaii | 0.074489 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat7 | nsga2-sector-farthest vs nsga2-farthest | 0.013380 | 13/0/8 | 0.026331 | 1.000000 | no |
| zcat7 | nsga2-sector-farthest vs gces-noGeo | 0.015099 | 15/0/6 | 0.006281 | 0.596685 | no |
| zcat7 | nsga2-sector-farthest vs gces | 0.016419 | 15/0/6 | 0.002482 | 0.245759 | no |
| zcat7 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.048901 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-farthest vs nsgaii | 0.169852 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | gces-noGeo vs nsgaii | 0.165722 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | gces vs nsgaii | 0.163794 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-farthest vs gces-noGeo | 0.004130 | 11/0/10 | 0.392584 | 1.000000 | no |
| zcat8 | nsga2-farthest vs gces | 0.006058 | 13/0/8 | 0.452368 | 1.000000 | no |
| zcat8 | nsga2-hvfarthest vs nsgaii | 0.589909 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvfarthest vs nsga2-farthest | 0.420057 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvfarthest vs gces-noGeo | 0.424187 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvfarthest vs gces | 0.426115 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs nsgaii | 0.351575 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-farthest | 0.181723 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs gces-noGeo | 0.185853 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs gces | 0.187781 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.238334 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-sector-farthest vs nsgaii | 0.182958 | 20/1/0 | 0.000064 | 0.007227 | yes |
| zcat8 | nsga2-sector-farthest vs nsga2-farthest | 0.013105 | 10/0/11 | 0.864887 | 1.000000 | no |
| zcat8 | nsga2-sector-farthest vs gces-noGeo | 0.017235 | 11/0/10 | 0.633297 | 1.000000 | no |
| zcat8 | nsga2-sector-farthest vs gces | 0.019163 | 12/0/9 | 0.562075 | 1.000000 | no |
| zcat8 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.168618 | 1/0/20 | 0.000002 | 0.000362 | yes |
| zcat9 | nsga2-farthest vs nsgaii | 0.096189 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | gces-noGeo vs nsgaii | 0.073669 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat9 | gces vs nsgaii | 0.088210 | 19/0/2 | 0.000010 | 0.001154 | yes |
| zcat9 | nsga2-farthest vs gces-noGeo | 0.022520 | 12/0/9 | 0.355416 | 1.000000 | no |
| zcat9 | nsga2-farthest vs gces | 0.007979 | 11/0/10 | 0.411982 | 1.000000 | no |
| zcat9 | nsga2-hvfarthest vs nsgaii | 0.305587 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvfarthest vs nsga2-farthest | 0.209399 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvfarthest vs gces-noGeo | 0.231918 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvfarthest vs gces | 0.217378 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs nsgaii | 0.260730 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-farthest | 0.164541 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs gces-noGeo | 0.187061 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs gces | 0.172521 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.044857 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-sector-farthest vs nsgaii | 0.075094 | 20/0/1 | 0.000003 | 0.000362 | yes |
| zcat9 | nsga2-sector-farthest vs nsga2-farthest | -0.021095 | 9/0/12 | 0.785365 | 1.000000 | no |
| zcat9 | nsga2-sector-farthest vs gces-noGeo | 0.001424 | 9/0/12 | 0.838194 | 1.000000 | no |
| zcat9 | nsga2-sector-farthest vs gces | -0.013116 | 10/0/11 | 0.838194 | 1.000000 | no |
| zcat9 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.185637 | 0/0/21 | 0.000001 | 0.000362 | yes |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | -0.032759 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | gces-noGeo vs nsgaii | -0.031362 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | gces vs nsgaii | -0.030114 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-farthest vs gces-noGeo | -0.001397 | 10/0/11 | 0.972858 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces | -0.002645 | 14/0/7 | 0.272210 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsgaii | -0.073076 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | -0.040316 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | -0.041714 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvfarthest vs gces | -0.042962 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs nsgaii | -0.069857 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | -0.037098 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | -0.038495 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs gces | -0.039743 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.003219 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-sector-farthest vs nsgaii | -0.029522 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat1 | nsga2-sector-farthest vs nsga2-farthest | 0.003237 | 9/0/12 | 0.609149 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces-noGeo | 0.001839 | 9/0/12 | 0.657827 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces | 0.000592 | 9/0/12 | 0.838194 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.040335 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-farthest vs nsgaii | -0.042733 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat10 | gces-noGeo vs nsgaii | -0.029197 | 16/0/5 | 0.000354 | 0.047057 | yes |
| zcat10 | gces vs nsgaii | -0.043901 | 19/0/2 | 0.000007 | 0.000968 | yes |
| zcat10 | nsga2-farthest vs gces-noGeo | -0.013536 | 14/0/7 | 0.287734 | 1.000000 | no |
| zcat10 | nsga2-farthest vs gces | 0.001168 | 9/0/12 | 0.609149 | 1.000000 | no |
| zcat10 | nsga2-hvfarthest vs nsgaii | -0.096284 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvfarthest vs nsga2-farthest | -0.053551 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvfarthest vs gces-noGeo | -0.067087 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvfarthest vs gces | -0.052383 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs nsgaii | -0.086188 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-farthest | -0.043455 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs gces-noGeo | -0.056991 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs gces | -0.042287 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat10 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.010095 | 2/0/19 | 0.000067 | 0.009413 | yes |
| zcat10 | nsga2-sector-farthest vs nsgaii | -0.033731 | 19/0/2 | 0.000007 | 0.000968 | yes |
| zcat10 | nsga2-sector-farthest vs nsga2-farthest | 0.009002 | 8/0/13 | 0.494802 | 1.000000 | no |
| zcat10 | nsga2-sector-farthest vs gces-noGeo | -0.004534 | 11/0/10 | 0.682714 | 1.000000 | no |
| zcat10 | nsga2-sector-farthest vs gces | 0.010170 | 11/0/10 | 0.373725 | 1.000000 | no |
| zcat10 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.052457 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-farthest vs nsgaii | -0.089914 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | gces-noGeo vs nsgaii | -0.088546 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | gces vs nsgaii | -0.079330 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-farthest vs gces-noGeo | -0.001368 | 10/0/11 | 0.918694 | 1.000000 | no |
| zcat11 | nsga2-farthest vs gces | -0.010584 | 10/0/11 | 0.609149 | 1.000000 | no |
| zcat11 | nsga2-hvfarthest vs nsgaii | -0.165559 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat11 | nsga2-hvfarthest vs nsga2-farthest | -0.075645 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat11 | nsga2-hvfarthest vs gces-noGeo | -0.077012 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat11 | nsga2-hvfarthest vs gces | -0.086229 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat11 | nsga2-hvref-farthest vs nsgaii | -0.151037 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-farthest | -0.061123 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs gces-noGeo | -0.062491 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs gces | -0.071708 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.014521 | 1/0/20 | 0.000426 | 0.055844 | no |
| zcat11 | nsga2-sector-farthest vs nsgaii | -0.095387 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat11 | nsga2-sector-farthest vs nsga2-farthest | -0.005473 | 14/0/7 | 0.010125 | 1.000000 | no |
| zcat11 | nsga2-sector-farthest vs gces-noGeo | -0.006841 | 16/0/5 | 0.003279 | 0.380333 | no |
| zcat11 | nsga2-sector-farthest vs gces | -0.016057 | 15/0/6 | 0.006281 | 0.703461 | no |
| zcat11 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.055650 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-farthest vs nsgaii | -0.243749 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | gces-noGeo vs nsgaii | -0.233688 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | gces vs nsgaii | -0.233726 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-farthest vs gces-noGeo | -0.010061 | 13/0/8 | 0.103214 | 1.000000 | no |
| zcat12 | nsga2-farthest vs gces | -0.010023 | 15/0/6 | 0.075980 | 1.000000 | no |
| zcat12 | nsga2-hvfarthest vs nsgaii | -0.347607 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvfarthest vs nsga2-farthest | -0.103858 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvfarthest vs gces-noGeo | -0.113919 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvfarthest vs gces | -0.113881 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs nsgaii | -0.321830 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-farthest | -0.078080 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs gces-noGeo | -0.088142 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs gces | -0.088104 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat12 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.025777 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-sector-farthest vs nsgaii | -0.247294 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat12 | nsga2-sector-farthest vs nsga2-farthest | -0.003544 | 13/0/8 | 0.516761 | 1.000000 | no |
| zcat12 | nsga2-sector-farthest vs gces-noGeo | -0.013606 | 16/0/5 | 0.021571 | 1.000000 | no |
| zcat12 | nsga2-sector-farthest vs gces | -0.013568 | 15/0/6 | 0.026331 | 1.000000 | no |
| zcat12 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.074536 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-farthest vs nsgaii | -0.122093 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | gces-noGeo vs nsgaii | -0.122698 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | gces vs nsgaii | -0.120516 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-farthest vs gces-noGeo | 0.000604 | 11/0/10 | 0.539189 | 1.000000 | no |
| zcat13 | nsga2-farthest vs gces | -0.001577 | 12/0/9 | 0.373725 | 1.000000 | no |
| zcat13 | nsga2-hvfarthest vs nsgaii | -0.186301 | 20/0/1 | 0.000197 | 0.027045 | yes |
| zcat13 | nsga2-hvfarthest vs nsga2-farthest | -0.064208 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat13 | nsga2-hvfarthest vs gces-noGeo | -0.063604 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat13 | nsga2-hvfarthest vs gces | -0.065785 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat13 | nsga2-hvref-farthest vs nsgaii | -0.175021 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-farthest | -0.052927 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs gces-noGeo | -0.052323 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs gces | -0.054505 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.011281 | 1/0/20 | 0.000426 | 0.055844 | no |
| zcat13 | nsga2-sector-farthest vs nsgaii | -0.125059 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat13 | nsga2-sector-farthest vs nsga2-farthest | -0.002966 | 11/0/10 | 0.585402 | 1.000000 | no |
| zcat13 | nsga2-sector-farthest vs gces-noGeo | -0.002362 | 13/0/8 | 0.095799 | 1.000000 | no |
| zcat13 | nsga2-sector-farthest vs gces | -0.004543 | 16/0/5 | 0.119342 | 1.000000 | no |
| zcat13 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.049961 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-farthest vs nsgaii | -0.023016 | 12/0/9 | 0.075980 | 1.000000 | no |
| zcat14 | gces-noGeo vs nsgaii | -0.028433 | 14/0/7 | 0.038438 | 1.000000 | no |
| zcat14 | gces vs nsgaii | -0.027010 | 13/0/8 | 0.050192 | 1.000000 | no |
| zcat14 | nsga2-farthest vs gces-noGeo | 0.005417 | 9/0/12 | 0.337660 | 1.000000 | no |
| zcat14 | nsga2-farthest vs gces | 0.003995 | 8/0/13 | 0.337660 | 1.000000 | no |
| zcat14 | nsga2-hvfarthest vs nsgaii | -0.134531 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvfarthest vs nsga2-farthest | -0.111515 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvfarthest vs gces-noGeo | -0.106099 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvfarthest vs gces | -0.107521 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs nsgaii | -0.099094 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-farthest | -0.076078 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs gces-noGeo | -0.070661 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs gces | -0.072084 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.035437 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat14 | nsga2-sector-farthest vs nsgaii | -0.031192 | 14/0/7 | 0.011347 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs nsga2-farthest | -0.008177 | 15/0/6 | 0.050192 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs gces-noGeo | -0.002760 | 11/0/10 | 0.337660 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs gces | -0.004182 | 12/0/9 | 0.733470 | 1.000000 | no |
| zcat14 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.067902 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-farthest vs nsgaii | -0.094665 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | gces-noGeo vs nsgaii | -0.107774 | 20/0/1 | 0.000007 | 0.000968 | yes |
| zcat15 | gces vs nsgaii | -0.107763 | 19/0/2 | 0.000197 | 0.027045 | yes |
| zcat15 | nsga2-farthest vs gces-noGeo | 0.013109 | 6/0/15 | 0.004879 | 0.556206 | no |
| zcat15 | nsga2-farthest vs gces | 0.013098 | 7/0/14 | 0.035056 | 1.000000 | no |
| zcat15 | nsga2-hvfarthest vs nsgaii | -0.249161 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvfarthest vs nsga2-farthest | -0.154496 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvfarthest vs gces-noGeo | -0.141387 | 19/0/2 | 0.008010 | 0.881090 | no |
| zcat15 | nsga2-hvfarthest vs gces | -0.141398 | 19/0/2 | 0.006281 | 0.703461 | no |
| zcat15 | nsga2-hvref-farthest vs nsgaii | -0.196337 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-farthest | -0.101672 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-hvref-farthest vs gces-noGeo | -0.088564 | 20/0/1 | 0.000426 | 0.055844 | no |
| zcat15 | nsga2-hvref-farthest vs gces | -0.088575 | 20/0/1 | 0.000354 | 0.047057 | yes |
| zcat15 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.052824 | 1/0/20 | 0.000426 | 0.055844 | no |
| zcat15 | nsga2-sector-farthest vs nsgaii | -0.091400 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat15 | nsga2-sector-farthest vs nsga2-farthest | 0.003266 | 14/0/7 | 0.042080 | 1.000000 | no |
| zcat15 | nsga2-sector-farthest vs gces-noGeo | 0.016374 | 8/0/13 | 0.157134 | 1.000000 | no |
| zcat15 | nsga2-sector-farthest vs gces | 0.016363 | 9/0/12 | 0.516761 | 1.000000 | no |
| zcat15 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.104938 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-farthest vs nsgaii | 0.012325 | 6/0/15 | 0.137283 | 1.000000 | no |
| zcat16 | gces-noGeo vs nsgaii | 0.007440 | 9/0/12 | 0.178988 | 1.000000 | no |
| zcat16 | gces vs nsgaii | 0.004837 | 6/0/15 | 0.257248 | 1.000000 | no |
| zcat16 | nsga2-farthest vs gces-noGeo | 0.004885 | 9/0/12 | 0.494802 | 1.000000 | no |
| zcat16 | nsga2-farthest vs gces | 0.007489 | 8/0/13 | 0.320457 | 1.000000 | no |
| zcat16 | nsga2-hvfarthest vs nsgaii | -0.072198 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvfarthest vs nsga2-farthest | -0.084523 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvfarthest vs gces-noGeo | -0.079638 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvfarthest vs gces | -0.077034 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs nsgaii | -0.053553 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-farthest | -0.065879 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs gces-noGeo | -0.060993 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs gces | -0.058390 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.018644 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat16 | nsga2-sector-farthest vs nsgaii | 0.008212 | 8/0/13 | 0.303816 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs nsga2-farthest | -0.004113 | 12/0/9 | 0.411982 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs gces-noGeo | 0.000772 | 12/0/9 | 0.891732 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs gces | 0.003376 | 7/0/14 | 0.733470 | 1.000000 | no |
| zcat16 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.061766 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-farthest vs nsgaii | -0.096774 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | gces-noGeo vs nsgaii | -0.086898 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat17 | gces vs nsgaii | -0.098625 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-farthest vs gces-noGeo | -0.009876 | 13/0/8 | 0.064650 | 1.000000 | no |
| zcat17 | nsga2-farthest vs gces | 0.001851 | 12/0/9 | 0.838194 | 1.000000 | no |
| zcat17 | nsga2-hvfarthest vs nsgaii | -0.152653 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvfarthest vs nsga2-farthest | -0.055879 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvfarthest vs gces-noGeo | -0.065755 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvfarthest vs gces | -0.054028 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs nsgaii | -0.128856 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-farthest | -0.032082 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs gces-noGeo | -0.041958 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs gces | -0.030231 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.023797 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-sector-farthest vs nsgaii | -0.086266 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat17 | nsga2-sector-farthest vs nsga2-farthest | 0.010507 | 7/0/14 | 0.045993 | 1.000000 | no |
| zcat17 | nsga2-sector-farthest vs gces-noGeo | 0.000631 | 13/0/8 | 0.945745 | 1.000000 | no |
| zcat17 | nsga2-sector-farthest vs gces | 0.012358 | 5/0/16 | 0.026331 | 1.000000 | no |
| zcat17 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.042590 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-farthest vs nsgaii | -0.085657 | 20/0/1 | 0.000067 | 0.009413 | yes |
| zcat18 | gces-noGeo vs nsgaii | -0.079917 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | gces vs nsgaii | -0.090273 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-farthest vs gces-noGeo | -0.005740 | 11/0/10 | 0.785365 | 1.000000 | no |
| zcat18 | nsga2-farthest vs gces | 0.004616 | 9/0/12 | 0.303816 | 1.000000 | no |
| zcat18 | nsga2-hvfarthest vs nsgaii | -0.164496 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvfarthest vs nsga2-farthest | -0.078839 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvfarthest vs gces-noGeo | -0.084579 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvfarthest vs gces | -0.074223 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs nsgaii | -0.150625 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-farthest | -0.064968 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs gces-noGeo | -0.070708 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs gces | -0.060352 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.013872 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat18 | nsga2-sector-farthest vs nsgaii | -0.093065 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat18 | nsga2-sector-farthest vs nsga2-farthest | -0.007407 | 15/0/6 | 0.190687 | 1.000000 | no |
| zcat18 | nsga2-sector-farthest vs gces-noGeo | -0.013148 | 17/0/4 | 0.075980 | 1.000000 | no |
| zcat18 | nsga2-sector-farthest vs gces | -0.002791 | 15/0/6 | 0.190687 | 1.000000 | no |
| zcat18 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.057560 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-farthest vs nsgaii | -0.141771 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | gces-noGeo vs nsgaii | -0.142579 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | gces vs nsgaii | -0.128462 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-farthest vs gces-noGeo | 0.000807 | 8/0/13 | 0.320457 | 1.000000 | no |
| zcat19 | nsga2-farthest vs gces | -0.013309 | 12/0/9 | 0.733470 | 1.000000 | no |
| zcat19 | nsga2-hvfarthest vs nsgaii | -0.247415 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvfarthest vs nsga2-farthest | -0.105644 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvfarthest vs gces-noGeo | -0.104837 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvfarthest vs gces | -0.118954 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs nsgaii | -0.224736 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-farthest | -0.082965 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs gces-noGeo | -0.082158 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs gces | -0.096274 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.022679 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-sector-farthest vs nsgaii | -0.147649 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat19 | nsga2-sector-farthest vs nsga2-farthest | -0.005878 | 13/0/8 | 0.320457 | 1.000000 | no |
| zcat19 | nsga2-sector-farthest vs gces-noGeo | -0.005071 | 12/0/9 | 0.373725 | 1.000000 | no |
| zcat19 | nsga2-sector-farthest vs gces | -0.019188 | 14/0/7 | 0.064650 | 1.000000 | no |
| zcat19 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.077087 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-farthest vs nsgaii | -0.034780 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | gces-noGeo vs nsgaii | -0.033056 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | gces vs nsgaii | -0.036068 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-farthest vs gces-noGeo | -0.001724 | 12/0/9 | 0.373725 | 1.000000 | no |
| zcat2 | nsga2-farthest vs gces | 0.001287 | 12/0/9 | 0.431911 | 1.000000 | no |
| zcat2 | nsga2-hvfarthest vs nsgaii | -0.062696 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvfarthest vs nsga2-farthest | -0.027916 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvfarthest vs gces-noGeo | -0.029640 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvfarthest vs gces | -0.026629 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs nsgaii | -0.052936 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-farthest | -0.018156 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs gces-noGeo | -0.019880 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs gces | -0.016868 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.009760 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-sector-farthest vs nsgaii | -0.031414 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat2 | nsga2-sector-farthest vs nsga2-farthest | 0.003366 | 3/0/18 | 0.003753 | 0.431561 | no |
| zcat2 | nsga2-sector-farthest vs gces-noGeo | 0.001641 | 5/0/16 | 0.035056 | 1.000000 | no |
| zcat2 | nsga2-sector-farthest vs gces | 0.004653 | 7/0/14 | 0.202917 | 1.000000 | no |
| zcat2 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.021521 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-farthest vs nsgaii | -0.077760 | 18/0/3 | 0.004879 | 0.556206 | no |
| zcat20 | gces-noGeo vs nsgaii | -0.083661 | 18/0/3 | 0.000067 | 0.009413 | yes |
| zcat20 | gces vs nsgaii | -0.047732 | 19/0/2 | 0.000852 | 0.100492 | no |
| zcat20 | nsga2-farthest vs gces-noGeo | 0.005900 | 9/0/12 | 0.215680 | 1.000000 | no |
| zcat20 | nsga2-farthest vs gces | -0.030028 | 13/0/8 | 0.473334 | 1.000000 | no |
| zcat20 | nsga2-hvfarthest vs nsgaii | -0.163837 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvfarthest vs nsga2-farthest | -0.086077 | 20/0/1 | 0.000241 | 0.032573 | yes |
| zcat20 | nsga2-hvfarthest vs gces-noGeo | -0.080176 | 18/0/3 | 0.000607 | 0.072291 | no |
| zcat20 | nsga2-hvfarthest vs gces | -0.116105 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs nsgaii | -0.142404 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat20 | nsga2-hvref-farthest vs nsga2-farthest | -0.064644 | 17/0/4 | 0.000105 | 0.014477 | yes |
| zcat20 | nsga2-hvref-farthest vs gces-noGeo | -0.058743 | 15/0/6 | 0.012691 | 1.000000 | no |
| zcat20 | nsga2-hvref-farthest vs gces | -0.094672 | 18/0/3 | 0.000510 | 0.061736 | no |
| zcat20 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.021433 | 4/0/17 | 0.050192 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs nsgaii | -0.093319 | 19/0/2 | 0.000293 | 0.039232 | yes |
| zcat20 | nsga2-sector-farthest vs nsga2-farthest | -0.015559 | 15/0/6 | 0.082195 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs gces-noGeo | -0.009659 | 10/0/11 | 0.811678 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs gces | -0.045587 | 15/0/6 | 0.011347 | 1.000000 | no |
| zcat20 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.049085 | 5/0/16 | 0.011347 | 1.000000 | no |
| zcat3 | nsga2-farthest vs nsgaii | -0.109046 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | gces-noGeo vs nsgaii | -0.111404 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | gces vs nsgaii | -0.103603 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat3 | nsga2-farthest vs gces-noGeo | 0.002358 | 9/0/12 | 0.972858 | 1.000000 | no |
| zcat3 | nsga2-farthest vs gces | -0.005443 | 12/0/9 | 0.242842 | 1.000000 | no |
| zcat3 | nsga2-hvfarthest vs nsgaii | -0.195723 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvfarthest vs nsga2-farthest | -0.086677 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvfarthest vs gces-noGeo | -0.084319 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvfarthest vs gces | -0.092120 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs nsgaii | -0.181869 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-farthest | -0.072822 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs gces-noGeo | -0.070465 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs gces | -0.078266 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.013854 | 1/0/20 | 0.000002 | 0.000362 | yes |
| zcat3 | nsga2-sector-farthest vs nsgaii | -0.121154 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat3 | nsga2-sector-farthest vs nsga2-farthest | -0.012107 | 11/0/10 | 0.303816 | 1.000000 | no |
| zcat3 | nsga2-sector-farthest vs gces-noGeo | -0.009750 | 10/0/11 | 0.257248 | 1.000000 | no |
| zcat3 | nsga2-sector-farthest vs gces | -0.017551 | 16/0/5 | 0.029015 | 1.000000 | no |
| zcat3 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.060715 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-farthest vs nsgaii | -0.099050 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | gces-noGeo vs nsgaii | -0.104370 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | gces vs nsgaii | -0.100127 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-farthest vs gces-noGeo | 0.005320 | 11/0/10 | 0.945745 | 1.000000 | no |
| zcat4 | nsga2-farthest vs gces | 0.001077 | 13/0/8 | 0.431911 | 1.000000 | no |
| zcat4 | nsga2-hvfarthest vs nsgaii | -0.198097 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvfarthest vs nsga2-farthest | -0.099047 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvfarthest vs gces-noGeo | -0.093728 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvfarthest vs gces | -0.097970 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs nsgaii | -0.185100 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-farthest | -0.086050 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs gces-noGeo | -0.080731 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs gces | -0.084973 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.012997 | 2/0/19 | 0.000005 | 0.000696 | yes |
| zcat4 | nsga2-sector-farthest vs nsgaii | -0.112597 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat4 | nsga2-sector-farthest vs nsga2-farthest | -0.013547 | 14/0/7 | 0.070137 | 1.000000 | no |
| zcat4 | nsga2-sector-farthest vs gces-noGeo | -0.008228 | 12/0/9 | 0.075980 | 1.000000 | no |
| zcat4 | nsga2-sector-farthest vs gces | -0.012470 | 16/0/5 | 0.023854 | 1.000000 | no |
| zcat4 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.072503 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-farthest vs nsgaii | -0.046607 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | gces-noGeo vs nsgaii | -0.046170 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | gces vs nsgaii | -0.045153 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-farthest vs gces-noGeo | -0.000436 | 11/0/10 | 0.494802 | 1.000000 | no |
| zcat5 | nsga2-farthest vs gces | -0.001453 | 12/0/9 | 0.707934 | 1.000000 | no |
| zcat5 | nsga2-hvfarthest vs nsgaii | -0.086750 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvfarthest vs nsga2-farthest | -0.040143 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvfarthest vs gces-noGeo | -0.040579 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvfarthest vs gces | -0.041596 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs nsgaii | -0.072893 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-farthest | -0.026287 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs gces-noGeo | -0.026723 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs gces | -0.027740 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.013856 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-sector-farthest vs nsgaii | -0.046222 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat5 | nsga2-sector-farthest vs nsga2-farthest | 0.000385 | 11/0/10 | 0.918694 | 1.000000 | no |
| zcat5 | nsga2-sector-farthest vs gces-noGeo | -0.000052 | 11/0/10 | 0.452368 | 1.000000 | no |
| zcat5 | nsga2-sector-farthest vs gces | -0.001069 | 14/0/7 | 0.609149 | 1.000000 | no |
| zcat5 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.026671 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-farthest vs nsgaii | -0.071471 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | gces-noGeo vs nsgaii | -0.070886 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | gces vs nsgaii | -0.069594 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-farthest vs gces-noGeo | -0.000585 | 12/0/9 | 0.562075 | 1.000000 | no |
| zcat6 | nsga2-farthest vs gces | -0.001877 | 13/0/8 | 0.128078 | 1.000000 | no |
| zcat6 | nsga2-hvfarthest vs nsgaii | -0.115998 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvfarthest vs nsga2-farthest | -0.044527 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvfarthest vs gces-noGeo | -0.045112 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvfarthest vs gces | -0.046404 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs nsgaii | -0.113665 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-farthest | -0.042194 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs gces-noGeo | -0.042779 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs gces | -0.044071 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.002333 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-sector-farthest vs nsgaii | -0.070604 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat6 | nsga2-sector-farthest vs nsga2-farthest | 0.000867 | 9/0/12 | 0.759288 | 1.000000 | no |
| zcat6 | nsga2-sector-farthest vs gces-noGeo | 0.000282 | 10/0/11 | 0.945745 | 1.000000 | no |
| zcat6 | nsga2-sector-farthest vs gces | -0.001010 | 12/0/9 | 0.373725 | 1.000000 | no |
| zcat6 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.043061 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-farthest vs nsgaii | -0.083015 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | gces-noGeo vs nsgaii | -0.085810 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | gces vs nsgaii | -0.081385 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-farthest vs gces-noGeo | 0.002796 | 9/0/12 | 0.178988 | 1.000000 | no |
| zcat7 | nsga2-farthest vs gces | -0.001630 | 12/0/9 | 1.000000 | 1.000000 | no |
| zcat7 | nsga2-hvfarthest vs nsgaii | -0.160312 | 18/0/3 | 0.000013 | 0.001896 | yes |
| zcat7 | nsga2-hvfarthest vs nsga2-farthest | -0.077297 | 18/0/3 | 0.042080 | 1.000000 | no |
| zcat7 | nsga2-hvfarthest vs gces-noGeo | -0.074502 | 18/0/3 | 0.054693 | 1.000000 | no |
| zcat7 | nsga2-hvfarthest vs gces | -0.078927 | 18/0/3 | 0.045993 | 1.000000 | no |
| zcat7 | nsga2-hvref-farthest vs nsgaii | -0.142510 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-farthest | -0.059495 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs gces-noGeo | -0.056700 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs gces | -0.061125 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.017802 | 3/0/18 | 0.054693 | 1.000000 | no |
| zcat7 | nsga2-sector-farthest vs nsgaii | -0.095717 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat7 | nsga2-sector-farthest vs nsga2-farthest | -0.012703 | 17/0/4 | 0.002151 | 0.251724 | no |
| zcat7 | nsga2-sector-farthest vs gces-noGeo | -0.009907 | 14/0/7 | 0.023854 | 1.000000 | no |
| zcat7 | nsga2-sector-farthest vs gces | -0.014332 | 17/0/4 | 0.000510 | 0.061736 | no |
| zcat7 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.046793 | 1/0/20 | 0.000002 | 0.000362 | yes |
| zcat8 | nsga2-farthest vs nsgaii | -0.097028 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | gces-noGeo vs nsgaii | -0.091964 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | gces vs nsgaii | -0.101994 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-farthest vs gces-noGeo | -0.005064 | 14/0/7 | 0.303816 | 1.000000 | no |
| zcat8 | nsga2-farthest vs gces | 0.004966 | 13/0/8 | 0.228986 | 1.000000 | no |
| zcat8 | nsga2-hvfarthest vs nsgaii | -0.183357 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvfarthest vs nsga2-farthest | -0.086330 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvfarthest vs gces-noGeo | -0.091394 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvfarthest vs gces | -0.081363 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs nsgaii | -0.164056 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-farthest | -0.067028 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs gces-noGeo | -0.072092 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs gces | -0.062061 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.019302 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-sector-farthest vs nsgaii | -0.090686 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat8 | nsga2-sector-farthest vs nsga2-farthest | 0.006342 | 8/0/13 | 0.303816 | 1.000000 | no |
| zcat8 | nsga2-sector-farthest vs gces-noGeo | 0.001278 | 10/0/11 | 0.864887 | 1.000000 | no |
| zcat8 | nsga2-sector-farthest vs gces | 0.011309 | 12/0/9 | 0.759288 | 1.000000 | no |
| zcat8 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.073370 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-farthest vs nsgaii | -0.036548 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | gces-noGeo vs nsgaii | -0.032124 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | gces vs nsgaii | -0.033989 | 20/0/1 | 0.000003 | 0.000421 | yes |
| zcat9 | nsga2-farthest vs gces-noGeo | -0.004424 | 14/0/7 | 0.228986 | 1.000000 | no |
| zcat9 | nsga2-farthest vs gces | -0.002559 | 13/0/8 | 0.373725 | 1.000000 | no |
| zcat9 | nsga2-hvfarthest vs nsgaii | -0.094742 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvfarthest vs nsga2-farthest | -0.058195 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvfarthest vs gces-noGeo | -0.062619 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvfarthest vs gces | -0.060754 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs nsgaii | -0.083564 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-farthest | -0.047016 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs gces-noGeo | -0.051440 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs gces | -0.049575 | 21/0/0 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.011178 | 0/0/21 | 0.000001 | 0.000362 | yes |
| zcat9 | nsga2-sector-farthest vs nsgaii | -0.033615 | 20/0/1 | 0.000002 | 0.000362 | yes |
| zcat9 | nsga2-sector-farthest vs nsga2-farthest | 0.002932 | 9/0/12 | 0.785365 | 1.000000 | no |
| zcat9 | nsga2-sector-farthest vs gces-noGeo | -0.001492 | 10/0/11 | 0.759288 | 1.000000 | no |
| zcat9 | nsga2-sector-farthest vs gces | 0.000373 | 11/0/10 | 0.657827 | 1.000000 | no |
| zcat9 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.049949 | 0/0/21 | 0.000001 | 0.000362 | yes |
