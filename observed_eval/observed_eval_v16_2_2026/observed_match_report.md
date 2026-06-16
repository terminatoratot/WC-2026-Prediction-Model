# v16_bayesian_bivariate_model observed World Cup evaluation

- Matches: 12
- Result accuracy: 50.0% (95% CI 25.0% to 75.0%)
- Result log loss: 1.022 (95% CI 0.752 to 1.291)
- Three-way Brier score: 0.627 (95% CI 0.431 to 0.820)
- Ranked probability score: 0.199 (95% CI 0.133 to 0.277)
- Goal MAE: 0.899 (95% CI 0.617 to 1.224)
- Goal RMSE: 1.355 (95% CI 0.775 to 1.812)
- Exact-score accuracy: 8.3%
- Log-loss skill vs uniform forecast: 7.0%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 12 | 0.500 | 1.022 | 0.627 | 0.199 | 0.899 |
| actual_result | draw | 4 | 0.000 | 1.413 | 0.944 | 0.186 | 0.513 |
| actual_result | team_a_win | 7 | 0.714 | 0.871 | 0.501 | 0.221 | 1.130 |
| actual_result | team_b_win | 1 | 1.000 | 0.516 | 0.245 | 0.096 | 0.819 |
| stage | Group Stage | 12 | 0.500 | 1.022 | 0.627 | 0.199 | 0.899 |
| group | Group A | 2 | 1.000 | 0.547 | 0.274 | 0.108 | 0.428 |
| group | Group B | 2 | 0.000 | 1.493 | 1.031 | 0.215 | 0.553 |
| group | Group C | 2 | 0.500 | 0.893 | 0.532 | 0.123 | 0.536 |
| group | Group D | 2 | 0.500 | 0.992 | 0.597 | 0.261 | 1.169 |
| group | Group E | 2 | 0.500 | 1.052 | 0.610 | 0.287 | 1.417 |
| group | Group F | 2 | 0.500 | 1.157 | 0.720 | 0.201 | 1.290 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 24 | 5.161 | 6.309 | 1.028 | 0.393 |
| shots_on_target | 24 | 1.948 | 2.283 | 0.694 | 0.563 |
| possession | 24 | 7.026 | 8.323 | 0.000 | 0.771 |
| fouls | 24 | 5.054 | 5.860 | 3.175 | 0.107 |
| yellow_cards | 24 | 1.139 | 1.295 | 0.353 | 0.315 |
| red_cards | 24 | 0.294 | 0.734 | -0.177 | 0.168 |
