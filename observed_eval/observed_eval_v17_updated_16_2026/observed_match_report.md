# v17_recency_all_matches_model observed World Cup evaluation

- Matches: 16
- Result accuracy: 37.5% (95% CI 12.5% to 62.5%)
- Result log loss: 1.216 (95% CI 0.895 to 1.548)
- Three-way Brier score: 0.757 (95% CI 0.534 to 0.970)
- Ranked probability score: 0.204 (95% CI 0.150 to 0.264)
- Goal MAE: 0.941 (95% CI 0.640 to 1.298)
- Goal RMSE: 1.463 (95% CI 0.890 to 1.954)
- Exact-score accuracy: 18.8%
- Log-loss skill vs uniform forecast: -10.7%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 16 | 0.375 | 1.216 | 0.757 | 0.204 | 0.941 |
| actual_result | draw | 8 | 0.000 | 1.694 | 1.100 | 0.228 | 0.834 |
| actual_result | team_a_win | 7 | 0.714 | 0.763 | 0.432 | 0.190 | 1.092 |
| actual_result | team_b_win | 1 | 1.000 | 0.568 | 0.282 | 0.116 | 0.737 |
| stage | Group Stage | 16 | 0.375 | 1.216 | 0.757 | 0.204 | 0.941 |
| group | Group A | 2 | 1.000 | 0.465 | 0.214 | 0.084 | 0.258 |
| group | Group B | 2 | 0.000 | 1.685 | 1.158 | 0.248 | 0.720 |
| group | Group C | 2 | 0.500 | 0.939 | 0.552 | 0.130 | 0.513 |
| group | Group D | 2 | 1.000 | 0.797 | 0.453 | 0.193 | 1.066 |
| group | Group E | 2 | 0.500 | 0.858 | 0.506 | 0.236 | 1.479 |
| group | Group F | 2 | 0.000 | 1.217 | 0.747 | 0.220 | 1.352 |
| group | Group G | 2 | 0.000 | 1.565 | 1.051 | 0.213 | 0.586 |
| group | Group H | 2 | 0.000 | 2.205 | 1.370 | 0.312 | 1.552 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 32 | 5.556 | 6.706 | -0.157 | 0.301 |
| shots_on_target | 32 | 2.210 | 2.787 | 0.307 | 0.305 |
| possession | 32 | 6.762 | 8.565 | 0.000 | 0.635 |
| fouls | 32 | 5.325 | 6.327 | 3.458 | -0.002 |
| yellow_cards | 32 | 1.079 | 1.254 | 0.391 | 0.204 |
| red_cards | 32 | 0.228 | 0.636 | -0.125 | 0.194 |
