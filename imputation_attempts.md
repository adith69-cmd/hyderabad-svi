# UDISE Imputation Attempts — Documented Failure

## Goal
Fill missing UDISE 2021-22 (63 wards) and 2023-24 (59 wards) values so
the pressure model has full infrastructure coverage.

## Attempt 1 — KNN with socioeconomic features
Features: census literacy, non-worker rate, SC/ST share, child share,
population, WorldPop growth, amenity count.
Result: **all 10 targets negative LOO R²** (range −0.32 to −0.05).
Interpretation: socioeconomic features do not predict school counts.

## Attempt 2 — KNN + GradientBoosting, spatial features added
Features: ward area (km²), distance from city center, population
density, plus all features from Attempt 1.
Result: **all 30 substantive targets negative LOO R²** (range −0.76 to +0.05).
Gradient Boosting outperformed KNN but still worse than predicting the mean.

## Interpretation
UDISE ward totals are driven by physical area, administrative history,
and educational planning decisions that are not captured in Census 2011
socioeconomic variables or WorldPop population dynamics. This is a
structural fact about the data, not a modeling failure.

## Secondary finding — informative missingness
The binary `_missing` indicator for UDISE columns is predictable from
features with GBM LOO R² = 0.567. This means the 63 wards without UDISE
records are **systematically different** from the 87 wards with records.
The missingness is not random — it correlates with the socioeconomic
profile that also drives SVI.

Implication: imputing UDISE values for those wards would introduce
systematic bias, not just noise. We do not impute.

## Final decision
The 63 wards without UDISE use neutral pressure imputation
(`udise_improvement_z = 0`) with a wider uncertainty band. This is
documented as a limitation, not hidden.