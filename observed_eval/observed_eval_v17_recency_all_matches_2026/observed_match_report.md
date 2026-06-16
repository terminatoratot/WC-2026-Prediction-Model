# v17_recency_all_matches_model observed World Cup evaluation

- Matches: 12
- Result accuracy: 50.0% (95% CI 25.0% to 75.0%)
- Result log loss: 0.993 (95% CI 0.703 to 1.274)
- Three-way Brier score: 0.605 (95% CI 0.392 to 0.812)
- Ranked probability score: 0.185 (95% CI 0.122 to 0.253)
- Goal MAE: 0.898 (95% CI 0.582 to 1.253)
- Goal RMSE: 1.365 (95% CI 0.775 to 1.862)
- Exact-score accuracy: 25.0%
- Log-loss skill vs uniform forecast: 9.6%

Bootstrap intervals measure sampling uncertainty across this observed match set. With a small number of matches, they should be expected to be wide.

## Subgroups

| dimension | subgroup | n_matches | result_accuracy | mean_result_log_loss | mean_result_brier | mean_result_rps | mean_goal_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 12 | 0.500 | 0.993 | 0.605 | 0.185 | 0.898 |
| actual_result | draw | 4 | 0.000 | 1.504 | 0.989 | 0.194 | 0.599 |
| actual_result | team_a_win | 7 | 0.714 | 0.763 | 0.432 | 0.190 | 1.092 |
| actual_result | team_b_win | 1 | 1.000 | 0.568 | 0.282 | 0.116 | 0.737 |
| stage | Group Stage | 12 | 0.500 | 0.993 | 0.605 | 0.185 | 0.898 |
| group | Group A | 2 | 1.000 | 0.465 | 0.214 | 0.084 | 0.258 |
| group | Group B | 2 | 0.000 | 1.685 | 1.158 | 0.248 | 0.720 |
| group | Group C | 2 | 0.500 | 0.939 | 0.552 | 0.130 | 0.513 |
| group | Group D | 2 | 1.000 | 0.797 | 0.453 | 0.193 | 1.066 |
| group | Group E | 2 | 0.500 | 0.858 | 0.506 | 0.236 | 1.479 |
| group | Group F | 2 | 0.000 | 1.217 | 0.747 | 0.220 | 1.352 |

## Box events

| event | n_team_observations | mae | rmse | bias | correlation |
| --- | --- | --- | --- | --- | --- |
| shots | 24 | 5.161 | 6.309 | 1.028 | 0.393 |
| shots_on_target | 24 | 1.948 | 2.283 | 0.694 | 0.563 |
| possession | 24 | 7.026 | 8.323 | 0.000 | 0.771 |
| fouls | 24 | 5.054 | 5.860 | 3.175 | 0.107 |
| yellow_cards | 24 | 1.139 | 1.295 | 0.353 | 0.315 |
| red_cards | 24 | 0.294 | 0.734 | -0.177 | 0.168 |
