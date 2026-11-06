# Data

Source: `BELLoney/FGAS@b450898434d8f9ef78b9227b9b5d142a10af1fb6` (pinned commit), downloaded by `scripts/download_data.py`.
Full SHA-256 checksums: `data/checksums.json`. Raw files are git-ignored.

Layout of every file: a single matrix `trandata` (n x (m+1)); columns 1..m are features,
the last column is the binary label (1 = outlier).

Values in parentheses are Table 1 of the paper.

| # | Abbr | File | #attr | #obj | #outliers | NaNs | const cols | dup rows | raw range | vs Table 1 | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Card | cardio.mat | 21 (21) | 1831 (1831) | 176 (176) | 0 | 0 | 9 | [0, 1] | OK | `7392e978abb0` |
| 2 | Diab | diabetes_tested_positive_26_variant1.mat | 8 (8) | 526 (526) | 26 (26) | 0 | 0 | 0 | [0, 1] | OK | `736a5001fd30` |
| 3 | Ecoli | ecoli.mat | 7 (7) | 336 (336) | 9 (9) | 0 | 0 | 0 | [0, 1] | OK | `4d9749d28767` |
| 4 | Ionos | ionosphere_b_24_variant1.mat | 34 (34) | 249 (249) | 24 (24) | 0 | 1 | 0 | [0, 1] | OK | `ab932598cbbb` |
| 5 | Iris | iris_Irisvirginica_11_variant1.mat | 4 (4) | 111 (111) | 11 (11) | 0 | 0 | 2 | [0, 1] | OK | `32fdd9f9610c` |
| 6 | Musk | musk.mat | 166 (166) | 3062 (3062) | 97 (97) | 0 | 0 | 0 | [0, 1] | OK | `16bd407c657b` |
| 7 | Page | pageblocks_1_258_variant1.mat | 10 (10) | 5171 (5171) | 258 (258) | 0 | 0 | 27 | [0, 1] | OK | `a35c074b06be` |
| 8 | Pima | pima_TRUE_55_variant1.mat | 9 (9) | 555 (555) | 55 (55) | 0 | 0 | 0 | [0, 1] | OK | `d74688c80199` |
| 9 | Sate | satellite.mat | 36 (36) | 6435 (6435) | 2036 (2036) | 0 | 0 | 0 | [0, 1] | OK | `8a595ca8f939` |
| 10 | Sonar | sonar_M_10_variant1.mat | 60 (60) | 107 (107) | 10 (10) | 0 | 0 | 0 | [0, 1] | OK | `c19d7e400cb7` |
| 11 | Thyro | thyroid.mat | 6 (6) | 3772 (3772) | 93 (93) | 0 | 0 | 116 | [0, 1] | OK | `5b30bac96b39` |
| 12 | Wbc | wbc_malignant_39_variant1.mat | 9 (9) | 483 (483) | 39 (39) | 0 | 0 | 231 | [0, 1] | OK | `c37def19a6c4` |
| 13 | Wdbc | wdbc_M_39_variant1.mat | 31 (31) | 396 (396) | 39 (39) | 0 | 0 | 0 | [0, 1] | OK | `9edc72ac5bdf` |
| 14 | Wine | wine.mat | 13 (13) | 129 (129) | 10 (10) | 0 | 0 | 0 | [0, 1] | OK | `2980f8d9a584` |
| 15 | Wpbc | wpbc_variant1.mat | 33 (33) | 198 (198) | 47 (47) | 0 | 0 | 0 | [0, 1] | OK | `95cd8f8397f2` |
