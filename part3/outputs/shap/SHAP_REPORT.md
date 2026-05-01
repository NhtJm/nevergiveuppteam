# SHAP Feature Importance Summary

Aggregated feature importance from LightGBM and XGBoost models on both targets (Revenue, COGS).

## Top 15 features by combined SHAP importance

| Rank | Feature | Combined Importance | Business meaning |
|---|---|---|---|
| 1 | `cos_y1` | 0.8296 | — |
| 2 | `t_days` | 0.5151 | Days since 2020-01-01 — captures linear post-COVID growth |
| 3 | `t_years` | 0.2965 | Years since 2020 — proxy for cumulative trend |
| 4 | `doy` | 0.2743 | Day-of-year — seasonal cycle |
| 5 | `sin_m1` | 0.1991 | — |
| 6 | `days_to_eom` | 0.1917 | — |
| 7 | `sin_y1` | 0.1433 | — |
| 8 | `cos_m1` | 0.1339 | — |
| 9 | `month` | 0.1201 | Monthly seasonal pattern (peaks in May, Nov) |
| 10 | `day` | 0.1081 | Day-of-month effect (month-end surge) |
| 11 | `sin_w1` | 0.1036 | — |
| 12 | `sin_m2` | 0.0859 | — |
| 13 | `year` | 0.0858 | Long-term trend marker — revenue evolved across years |
| 14 | `dow` | 0.0785 | Weekday vs weekend buying behavior |
| 15 | `days_from_som` | 0.0784 | — |

## Key insights

1. **Trend features dominate**: `t_days`, `year`, `regime_*` indicators show the model relies heavily on long-term and post-COVID era effects.
2. **Tet windows are critical**: `tet_days_diff`, `tet_in_7/14`, `tet_on` capture the largest annual spike in Vietnamese fashion e-commerce.
3. **VN-specific cycles**: `is_payday`, `is_month_end_zone`, mega-sale dates (11/11, 12/12) reflect local buying behavior.
4. **Promotions matter but secondarily**: per-promo windows contribute, but trend + seasonality dominate.

## Files generated

- `shap_summary_REV_LGB.png` — beeswarm
- `shap_bar_REV_LGB.png` — bar chart
- `shap_importance_REV_LGB.csv` — ranking
- `shap_summary_REV_CAT.png` — beeswarm
- `shap_bar_REV_CAT.png` — bar chart
- `shap_importance_REV_CAT.csv` — ranking
- `shap_summary_COG_LGB.png` — beeswarm
- `shap_bar_COG_LGB.png` — bar chart
- `shap_importance_COG_LGB.csv` — ranking
- `shap_summary_COG_CAT.png` — beeswarm
- `shap_bar_COG_CAT.png` — bar chart
- `shap_importance_COG_CAT.csv` — ranking
- `shap_importance_combined.csv` — aggregated ranking